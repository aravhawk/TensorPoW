"""Adversarial regression for spam flooding against shard fee floors."""

from __future__ import annotations

from pathlib import Path

from tensorpow.mempool import MAX_FRUIT_PAYLOAD_BYTES, ROOT_SHARD_ID, Mempool
from tensorpow.node import TensorPowConfig, TensorPowNode
from tensorpow.state.utxo import UTXOSet
from tests.adversarial._helpers import anchor, coinbase_tx, fruit, genesis_anchor, signed_tx, utxo


def test_zero_fee_spam_stays_out_after_empty_fruit_flooding() -> None:
    spam_utxos = tuple(utxo(10_000 + index, amount=1_000) for index in range(32))
    mempool = Mempool(utxo_view=UTXOSet(spam_utxos))

    for _ in range(24):
        mempool.record_confirmed_fruit(
            ROOT_SHARD_ID,
            floor_eligible_fees_matoms=MAX_FRUIT_PAYLOAD_BYTES * 50,
            payload_bytes=MAX_FRUIT_PAYLOAD_BYTES,
        )
    high_floor = mempool.fee_floor(ROOT_SHARD_ID)

    for _ in range(16):
        mempool.record_confirmed_fruit(
            ROOT_SHARD_ID,
            floor_eligible_fees_matoms=0,
            payload_bytes=MAX_FRUIT_PAYLOAD_BYTES,
        )
    floor_after_empty_fruits = mempool.fee_floor(ROOT_SHARD_ID)

    assert high_floor > floor_after_empty_fruits > 0
    for funded in spam_utxos:
        result = mempool.add_tx(signed_tx(funded, fee=0))
        assert result.reason == "below_fee_floor"

    assert len(mempool) == 0
    assert mempool.fee_floor(ROOT_SHARD_ID) == floor_after_empty_fruits


def test_node_rejects_zero_fee_spam_fruit_after_fee_floor_anchor(tmp_path: Path) -> None:
    genesis = genesis_anchor()
    node = TensorPowNode(
        TensorPowConfig(
            data_dir=tmp_path / "spam-node",
            expected_genesis_hash=genesis.block_hash(),
        ),
        pow_verifier=lambda _header, _target, _backend: True,
    )
    try:
        floor_seed = fruit(
            (coinbase_tx(51).to_bytes(),),
            nonce=51,
            timestamp_ms=2,
            latest_anchor=genesis.block_hash(),
        )
        high_floor_anchor = anchor(
            covered_fruit_hashes=(floor_seed.block_hash(),),
            parent_anchor=genesis.block_hash(),
            fee_floor_matoms_per_kb=10_000,
            timestamp_ms=3,
            nonce=52,
        )
        funded = utxo(52_000, amount=1_000)
        zero_fee = signed_tx(funded, fee=0)
        spam_fruit = fruit(
            (coinbase_tx(53).to_bytes(), zero_fee.to_bytes()),
            nonce=53,
            timestamp_ms=4,
            parent_selected=floor_seed.block_hash(),
            latest_anchor=high_floor_anchor.block_hash(),
        )

        assert node.process_anchor(genesis)
        assert node.process_fruit(floor_seed)
        assert node.process_anchor(high_floor_anchor)
        node.store.put_utxo(funded)
        node.utxo_set.add(funded)
        assert node.process_fruit(spam_fruit).reason == "below_fee_floor"
        assert node.get_block(spam_fruit.block_hash()) is None
    finally:
        node.close()
