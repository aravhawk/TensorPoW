"""Tests for Graphene compact fruit relay sketches."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager

import tensorpow.net.graphene as graphene
from tensorpow.chain.blocks import (
    MAX_FRUIT_PAYLOAD_BYTES,
    Fruit,
    fruit_payload_size_bytes,
    tx_merkle_root,
)
from tensorpow.chain.headers import FruitHeader
from tensorpow.crypto.signatures import SIG_TYPE_ED25519, SIG_TYPE_ED25519_BIT, sign
from tensorpow.mempool import Mempool
from tensorpow.net import (
    CODEC_GRAPHENE,
    GRAPHENE_RECEIVER_MEMPOOL_PCT,
    GRAPHENE_TARGET_COMPRESSION_PCT,
    announce_fruit,
    reconstruct_fruit,
)
from tensorpow.pow.challenge import FORMAT_EPOCH
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint, UTXOSet
from tensorpow.tx.script import pubkey_hash
from tensorpow.tx.transaction import Input, Output, Transaction

PUB1 = bytes.fromhex("343010a1aba8774dd1e6f4f0c3349bae6824908a1e64cd638dc2ed1bc625af1d")
PRIV1 = bytes.fromhex("cd4f7f79a2b8168f5cbeccb55d415492fd3504e52ed4fe7b02ea404fede9a40b")
PKH1 = pubkey_hash(PUB1)


def test_reconstructs_fruit_when_receiver_has_all_transactions() -> None:
    txs = tuple(_relay_tx(seed) for seed in range(8))
    fruit = _fruit(tuple(tx.to_bytes() for tx in txs))
    mempool = _mempool(txs)

    sketch = announce_fruit(fruit)

    assert sketch[:2] == CODEC_GRAPHENE.to_bytes(2, "little")
    assert reconstruct_fruit(sketch, mempool) == fruit


def test_reconstruction_falls_back_when_missing_tx_is_unavailable() -> None:
    txs = tuple(_relay_tx(seed) for seed in range(8))
    fruit = _fruit(tuple(tx.to_bytes() for tx in txs))
    mempool = _mempool(txs[:-1])

    assert reconstruct_fruit(announce_fruit(fruit), mempool) is None


def test_rejects_malformed_truncated_and_noncanonical_sketches() -> None:
    txs = tuple(_relay_tx(seed) for seed in range(5))
    fruit = _fruit(tuple(tx.to_bytes() for tx in txs))
    mempool = _mempool(txs)
    sketch = announce_fruit(fruit)

    bad_codec = b"\x00\x00" + sketch[2:]

    assert reconstruct_fruit(b"", mempool) is None
    assert reconstruct_fruit(sketch[:-1], mempool) is None
    assert reconstruct_fruit(sketch + b"\x00", mempool) is None
    assert reconstruct_fruit(bad_codec, mempool) is None
    assert reconstruct_fruit(_with_nonzero_bloom_tail(sketch), mempool) is None
    assert reconstruct_fruit(_with_corrupt_iblt_cell(sketch), mempool) is None
    assert reconstruct_fruit(_with_duplicate_short_id(sketch), mempool) is None


def test_rejects_overbroad_bloom_before_mempool_reconstruction() -> None:
    tx = _relay_tx(1)
    fruit = _fruit((tx.to_bytes(),))
    mempool = _mempool(tuple(_relay_tx(seed) for seed in range(1, 64)))

    assert reconstruct_fruit(_with_full_bloom(announce_fruit(fruit)), mempool) is None


def test_reconstruction_preserves_canonical_fruit_order() -> None:
    txs = tuple(_relay_tx(seed) for seed in (6, 1, 4, 2, 5, 3))
    fruit = _fruit(tuple(tx.to_bytes() for tx in txs))
    reversed_mempool = _mempool(tuple(reversed(txs)))

    reconstructed = reconstruct_fruit(announce_fruit(fruit), reversed_mempool)

    assert reconstructed == fruit
    assert reconstructed is not None
    assert reconstructed.transactions == fruit.transactions


def test_sketch_meets_95_percent_compression_for_99_percent_overlap_fixture() -> None:
    transactions = tuple(_raw_fixture_tx(seed) for seed in range(100))
    receiver_has = transactions[:99]
    fruit = _fruit(transactions)

    sketch = announce_fruit(fruit)
    compression_pct = (len(fruit.serialize()) - len(sketch)) * 100 // len(fruit.serialize())

    assert fruit_payload_size_bytes(transactions) == MAX_FRUIT_PAYLOAD_BYTES
    assert len(receiver_has) * 100 // len(transactions) == GRAPHENE_RECEIVER_MEMPOOL_PCT
    assert compression_pct >= GRAPHENE_TARGET_COMPRESSION_PCT


def test_repeated_announcements_are_deterministic() -> None:
    txs = tuple(_relay_tx(seed) for seed in range(10))
    fruit = _fruit(tuple(tx.to_bytes() for tx in txs))

    first = announce_fruit(fruit)
    second = announce_fruit(fruit)
    third = announce_fruit(fruit)

    assert first == second == third


def test_iblt_peel_rejects_non_self_clearing_pure_cell() -> None:
    cell_count = 6
    key = b"trapkey1"
    trap_index = _outside_graphene_positions(key, cell_count)
    cells = [
        graphene._IBLTCell(count=0, key_sum=bytes(8), checksum_sum=0) for _ in range(cell_count)
    ]
    cells[trap_index] = graphene._IBLTCell(
        count=1,
        key_sum=key,
        checksum_sum=graphene._iblt_checksum(key),
    )

    with _deadline():
        assert graphene._peel_iblt(tuple(cells)) is None


def _fruit(transactions: tuple[bytes, ...]) -> Fruit:
    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=bytes.fromhex("11" * 32),
        parent_bitmap=b"",
        latest_anchor=bytes.fromhex("22" * 32),
        tx_merkle_root=tx_merkle_root(transactions),
        timestamp_ms=1,
        shard_id=0,
        nonce=2,
    )
    return Fruit(header=header, transactions=transactions)


def _relay_tx(seed: int) -> Transaction:
    utxo = _utxo(seed, amount=1_000)
    output = Output(999, TEMPLATE_PKH, payload=PKH1)
    unsigned = Transaction(
        version=FORMAT_EPOCH,
        sig_type=SIG_TYPE_ED25519,
        locktime_ms=0,
        lockheight=0,
        inputs=(Input(utxo.outpoint),),
        outputs=(output,),
    )
    signature = sign(unsigned.sighash(0), PRIV1)
    return Transaction(
        version=unsigned.version,
        sig_type=unsigned.sig_type,
        locktime_ms=unsigned.locktime_ms,
        lockheight=unsigned.lockheight,
        inputs=(Input(utxo.outpoint, witness=signature + PUB1),),
        outputs=unsigned.outputs,
    )


def _mempool(txs: tuple[Transaction, ...]) -> Mempool:
    mempool = Mempool(
        utxo_view=UTXOSet(
            _utxo(input_.previous_outpoint.output_index) for tx in txs for input_ in tx.inputs
        )
    )
    for tx in txs:
        result = mempool.add_tx(tx)
        assert result.accepted
    return mempool


def _utxo(seed: int, *, amount: int = 1_000) -> UTXO:
    return UTXO(
        outpoint=Outpoint(bytes([seed]) * 32, seed),
        amount_matoms=amount,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=PKH1,
        payload=PKH1,
    )


def _raw_fixture_tx(seed: int) -> bytes:
    prefix = seed.to_bytes(2, "little")
    payload_len = 68 if seed == 99 else 78
    return prefix + bytes([seed]) * payload_len


def _with_nonzero_bloom_tail(sketch: bytes) -> bytes:
    mutated = bytearray(sketch)
    bloom_start, bloom_len, _short_id_width, _iblt_start, _short_ids_start = _sketch_offsets(sketch)
    mutated[bloom_start + bloom_len - 1] |= 0x80
    return bytes(mutated)


def _with_full_bloom(sketch: bytes) -> bytes:
    mutated = bytearray(sketch)
    bloom_start, bloom_len, _short_id_width, _iblt_start, _short_ids_start = _sketch_offsets(sketch)
    for offset in range(bloom_len):
        mutated[bloom_start + offset] = 0xFF
    return bytes(mutated)


def _with_duplicate_short_id(sketch: bytes) -> bytes:
    mutated = bytearray(sketch)
    _bloom_start, _bloom_len, short_id_width, _iblt_start, short_ids_start = _sketch_offsets(sketch)
    first_short_id = mutated[short_ids_start : short_ids_start + short_id_width]
    second_short_id_start = short_ids_start + short_id_width
    mutated[second_short_id_start : second_short_id_start + short_id_width] = first_short_id
    return bytes(mutated)


def _with_corrupt_iblt_cell(sketch: bytes) -> bytes:
    mutated = bytearray(sketch)
    _bloom_start, _bloom_len, _short_id_width, iblt_start, _short_ids_start = _sketch_offsets(
        sketch
    )
    mutated[iblt_start + 2] ^= 0x01
    return bytes(mutated)


def _sketch_offsets(sketch: bytes) -> tuple[int, int, int, int, int]:
    body_start = 10
    header_len = int.from_bytes(sketch[body_start + 4 : body_start + 6], "little")
    cursor = body_start + 4 + 2 + header_len + 2
    bloom_bits = int.from_bytes(sketch[cursor : cursor + 2], "little")
    cursor += 2
    cursor += 1
    short_id_width = sketch[cursor]
    cursor += 1
    cursor += 1
    iblt_cell_count = int.from_bytes(sketch[cursor : cursor + 2], "little")
    cursor += 2
    cursor += 1
    bloom_len = (bloom_bits + 7) // 8
    bloom_start = cursor
    iblt_start = bloom_start + bloom_len
    short_ids_start = iblt_start + iblt_cell_count * 14
    return bloom_start, bloom_len, short_id_width, iblt_start, short_ids_start


def _outside_graphene_positions(key: bytes, cell_count: int) -> int:
    positions = set(graphene._iblt_positions(key, cell_count))
    return next(index for index in range(cell_count) if index not in positions)


@contextmanager
def _deadline() -> Iterator[None]:
    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise AssertionError("IBLT peel did not terminate")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
