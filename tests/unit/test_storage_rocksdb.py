"""Tests for RocksDB-backed TensorPoW storage."""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tensorpow.chain.blocks import Fruit, tx_merkle_root
from tensorpow.chain.headers import FruitHeader
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.mempool.shard_tree import ROOT_SHARD_ID, ShardTree
from tensorpow.pow.challenge import FORMAT_EPOCH
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.storage import (
    BENCHMARK_MIN_WRITES_PER_SEC,
    COLUMN_DAG,
    COLUMN_FEE_FLOORS,
    COLUMN_MEMPOOL,
    BatchDelete,
    BatchPut,
    RocksDBStore,
    StorageBatch,
    atomic_state_batch,
)
from tensorpow.tx.transaction import Output, Transaction


def test_column_families_and_typed_round_trip(tmp_path: Path) -> None:
    store = RocksDBStore(tmp_path / "db")
    tx = _tx(7)
    fruit = _fruit((tx.to_bytes(),))
    block_hash = fruit.block_hash()
    outpoint = Outpoint(tx.tx_id(), 0)
    utxo = UTXO(
        outpoint=outpoint,
        amount_matoms=7,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=bytes([7]) * 32,
        payload=bytes([7]) * 32,
    )

    store.put_header(block_hash, fruit.header)
    store.put_body(block_hash, fruit)
    store.put_utxo(utxo)
    store.put_mempool_tx(tx)
    store.put_shard_tree(ShardTree())
    store.put_fee_floor(ROOT_SHARD_ID, 123)
    store.flush()
    store.close()

    reopened = RocksDBStore(tmp_path / "db")
    assert reopened.get_header_bytes(block_hash) == fruit.header.serialize()
    assert reopened.get_body_bytes(block_hash) == fruit.serialize()
    assert reopened.get_utxo(outpoint) == utxo
    assert tuple(reopened.utxos()) == (utxo,)
    assert tuple(reopened.mempool_txs()) == (tx,)
    assert reopened.get_shard_tree() == ShardTree()
    assert reopened.fee_floor(ROOT_SHARD_ID) == 123
    reopened.close()


def test_atomic_batch_applies_all_or_none(tmp_path: Path) -> None:
    store = RocksDBStore(tmp_path / "db")
    tx = _tx(1)
    outpoint = Outpoint(tx.tx_id(), 0)
    utxo = UTXO(
        outpoint=outpoint,
        amount_matoms=1,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=bytes([1]) * 32,
        payload=bytes([1]) * 32,
    )
    fruit = _fruit((tx.to_bytes(),))
    batch = atomic_state_batch(
        headers=((fruit.block_hash(), fruit.header),),
        bodies=((fruit.block_hash(), fruit),),
        utxo_puts=(utxo,),
        mempool_puts=(tx,),
    )

    store.write_batch(batch)
    assert store.get_header_bytes(fruit.block_hash()) == fruit.header.serialize()
    assert store.get_utxo(outpoint) == utxo

    store.write_batch(
        StorageBatch(
            puts=(BatchPut(COLUMN_DAG, b"marker", b"ok"),),
            deletes=(
                BatchDelete(COLUMN_MEMPOOL, tx.tx_id()),
                BatchDelete(COLUMN_DAG, b"missing"),
            ),
        )
    )
    assert store.get(COLUMN_DAG, b"marker") == b"ok"
    assert tuple(store.mempool_txs()) == ()
    store.close()


def test_checkpoint_snapshot_and_repair_recover_consistent_state(tmp_path: Path) -> None:
    store = RocksDBStore(tmp_path / "db")
    store.put(COLUMN_DAG, b"k1", b"value-one")
    checkpoint_path = store.create_checkpoint(tmp_path / "checkpoint")
    store.put(COLUMN_DAG, b"k2", b"value-two")
    store.close()

    snapshot = RocksDBStore(checkpoint_path)
    assert snapshot.get(COLUMN_DAG, b"k1") == b"value-one"
    assert snapshot.get(COLUMN_DAG, b"k2") is None
    snapshot.close()

    RocksDBStore.repair(tmp_path / "db")
    recovered = RocksDBStore(tmp_path / "db")
    assert recovered.get(COLUMN_DAG, b"k1") == b"value-one"
    assert recovered.get(COLUMN_DAG, b"k2") == b"value-two"
    recovered.close()


def test_close_releases_handles_when_flush_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "db"
    store = RocksDBStore(db_path)
    store.put(COLUMN_DAG, b"k", b"value")

    def fail_flush(*, sync: bool = True) -> None:
        raise RuntimeError(f"flush failed sync={sync}")

    monkeypatch.setattr(store, "flush", fail_flush)

    with pytest.raises(RuntimeError, match="flush failed"):
        store.close()

    assert store._columns == {}
    assert store._handles == {}
    reopened = RocksDBStore(db_path)
    assert reopened.get(COLUMN_DAG, b"k") == b"value"
    reopened.close()


def test_items_and_typed_views_stream_without_python_sort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RocksDBStore(tmp_path / "db")
    tx = _tx(4)
    outpoint = Outpoint(tx.tx_id(), 0)
    utxo = UTXO(
        outpoint=outpoint,
        amount_matoms=4,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=bytes([4]) * 32,
        payload=bytes([4]) * 32,
    )
    store.put(COLUMN_FEE_FLOORS, b"b", b"second")
    store.put(COLUMN_FEE_FLOORS, b"a", b"first")
    store.put_utxo(utxo)
    store.put_mempool_tx(tx)

    def fail_sorted(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("storage iterators must not use Python sorted()")

    monkeypatch.setattr("builtins.sorted", fail_sorted)

    assert list(store.items(COLUMN_FEE_FLOORS)) == [
        (b"a", b"first"),
        (b"b", b"second"),
    ]
    assert list(store.utxos()) == [utxo]
    assert list(store.mempool_txs()) == [tx]
    store.close()


def test_reopen_after_killed_writer_preserves_batch_consistency(tmp_path: Path) -> None:
    db_path = tmp_path / "db"
    sentinel_path = tmp_path / "writer.started"
    script = textwrap.dedent(
        """
        import os
        import sys
        from tensorpow.storage import COLUMN_DAG, BatchPut, RocksDBStore, StorageBatch

        db_path, sentinel_path = sys.argv[1], sys.argv[2]
        store = RocksDBStore(db_path)
        for batch_index in range(10000):
            batch = StorageBatch(puts=tuple(
                BatchPut(
                    COLUMN_DAG,
                    b"batch:" + batch_index.to_bytes(4, "little") + item.to_bytes(2, "little"),
                    item.to_bytes(8, "little"),
                )
                for item in range(8)
            ))
            store.write_batch(batch)
            if batch_index == 8:
                open(sentinel_path, "wb").close()
        store.close()
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script, str(db_path), str(sentinel_path)])
    deadline = time.monotonic() + 5
    while not sentinel_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=5)

    store = RocksDBStore(db_path)
    items = dict(store.items(COLUMN_DAG))
    for batch_index in range(10000):
        present = [
            b"batch:" + batch_index.to_bytes(4, "little") + item.to_bytes(2, "little") in items
            for item in range(8)
        ]
        assert all(present) or not any(present)
    store.close()


def test_batched_write_benchmark_exceeds_10k_writes_per_second(tmp_path: Path) -> None:
    store = RocksDBStore(tmp_path / "db")
    result = store.benchmark_writes()

    assert result.writes == 10_000
    assert result.writes_per_second >= BENCHMARK_MIN_WRITES_PER_SEC
    store.close()


def _fruit(transactions: tuple[bytes, ...]) -> Fruit:
    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=bytes.fromhex("11" * 32),
        parent_bitmap=b"",
        latest_anchor=bytes.fromhex("22" * 32),
        tx_merkle_root=tx_merkle_root(transactions),
        timestamp_ms=1,
        shard_id=ROOT_SHARD_ID,
        nonce=2,
    )
    return Fruit(header=header, transactions=transactions)


def _tx(seed: int) -> Transaction:
    payload = bytes([seed]) * 32
    return Transaction.coinbase(
        (Output(amount_matoms=seed, template_id=TEMPLATE_PKH, payload=payload),)
    )
