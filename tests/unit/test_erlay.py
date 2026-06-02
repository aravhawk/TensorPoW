"""Tests for deterministic Erlay transaction reconciliation."""

from __future__ import annotations

import pytest

from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.mempool import ROOT_SHARD_ID, Mempool, child_shard_ids
from tensorpow.net.erlay import (
    CODEC_ERLAY,
    ERLAY_FIELD_BITS,
    ERLAY_INTERVAL_MS,
    ERLAY_MAX_DECODE_CAPACITY,
    ERLAY_SKETCH_MAGIC,
    ErlayPeerState,
    ErlaySetDifference,
    ErlaySketch,
    ErlaySketchError,
    erlay_topic_for_shard,
)
from tensorpow.state.utxo import TEMPLATE_PKH
from tensorpow.tx.transaction import Output, Transaction


def test_mostly_same_mempools_converge_with_diff_sized_payload() -> None:
    common = tuple(range(96))
    only_a = tuple(range(1000, 1004))
    only_b = tuple(range(2000, 2004))
    mempool_a = _mempool((*common, *only_a))
    mempool_b = _mempool((*common, *only_b))
    tx_ids_a = set(_mempool_tx_ids(mempool_a))
    tx_ids_b = set(_mempool_tx_ids(mempool_b))
    expected_union = tx_ids_a | tx_ids_b
    diff_count = len(tx_ids_a ^ tx_ids_b)
    state_a = ErlayPeerState(b"peer-a", b"peer-b")
    state_b = ErlayPeerState(b"peer-b", b"peer-a")

    sketch_a = state_a.build_sketch(ROOT_SHARD_ID, tx_ids_a, capacity=diff_count)
    sketch_b = state_b.build_sketch(ROOT_SHARD_ID, tx_ids_b, capacity=diff_count)
    from_a = state_a.reconcile(ROOT_SHARD_ID, tx_ids_a, sketch_a, sketch_b.to_bytes())
    from_b = state_b.reconcile(ROOT_SHARD_ID, tx_ids_b, sketch_b, sketch_a.to_bytes())

    tx_ids_needed_by_a = state_b.resolve_short_ids(
        ROOT_SHARD_ID,
        tx_ids_b,
        from_a.remote_only_short_ids,
    )
    tx_ids_needed_by_b = state_a.resolve_short_ids(
        ROOT_SHARD_ID,
        tx_ids_a,
        from_b.remote_only_short_ids,
    )
    request_payload_bytes = (
        len(from_a.remote_only_short_ids) + len(from_b.remote_only_short_ids)
    ) * 8
    erlay_payload_bytes = (
        len(sketch_a.to_bytes()) + len(sketch_b.to_bytes()) + request_payload_bytes
    )
    full_inventory_bytes = (len(tx_ids_a) + len(tx_ids_b)) * HASH_LEN_BYTES

    assert set(from_a.local_only_tx_ids) == tx_ids_a - tx_ids_b
    assert set(from_b.local_only_tx_ids) == tx_ids_b - tx_ids_a
    assert tx_ids_a | set(tx_ids_needed_by_a) == expected_union
    assert tx_ids_b | set(tx_ids_needed_by_b) == expected_union
    assert erlay_payload_bytes <= diff_count * 64
    assert erlay_payload_bytes < full_inventory_bytes // 10


def test_sketch_bytes_are_deterministic_and_peer_order_independent() -> None:
    tx_ids = tuple(reversed([_tx(seed).tx_id() for seed in range(12)]))
    state_ab = ErlayPeerState(b"peer-a", b"peer-b")
    state_ba = ErlayPeerState(b"peer-b", b"peer-a")

    encoded_ab = state_ab.build_sketch(ROOT_SHARD_ID, tx_ids, capacity=4).to_bytes()
    encoded_ba = state_ba.build_sketch(ROOT_SHARD_ID, tuple(sorted(tx_ids)), capacity=4).to_bytes()

    assert encoded_ab == encoded_ba
    assert ErlaySketch.from_bytes(encoded_ab).to_bytes() == encoded_ab
    assert (
        ErlaySketch.from_bytes(encoded_ab)
        .difference(ErlaySketch.from_bytes(encoded_ba))
        .decode_short_ids()
        == ()
    )


def test_malformed_sketches_reject_lengths_and_fields() -> None:
    state = ErlayPeerState(b"peer-a", b"peer-b")
    sketch = state.build_sketch(ROOT_SHARD_ID, (_tx(1).tx_id(),), capacity=2)
    encoded = sketch.to_bytes()

    malformed = [
        b"",
        encoded[:-1],
        encoded + b"\x00",
        _replace(encoded, 0, b"X"),
        _replace(encoded, _codec_offset(), (CODEC_ERLAY ^ 1).to_bytes(2, "little")),
        _replace(encoded, _field_offset(), (ERLAY_FIELD_BITS - 1).to_bytes(2, "little")),
        _replace(encoded, _shard_offset(), (17 << 16).to_bytes(4, "little")),
        _replace(encoded, _capacity_offset(), (0).to_bytes(2, "little")),
        _replace(encoded, _count_offset(), (sketch.capacity * 2 + 1).to_bytes(2, "little")),
    ]

    for raw in malformed:
        with pytest.raises(ErlaySketchError):
            ErlaySketch.from_bytes(raw)


def test_remote_sketch_capacity_is_capped_before_decode() -> None:
    oversized = ErlaySketch.from_short_ids(
        ROOT_SHARD_ID,
        (),
        capacity=ERLAY_MAX_DECODE_CAPACITY + 1,
    )

    with pytest.raises(ErlaySketchError, match="decode limit"):
        ErlaySketch.from_bytes(oversized.to_bytes())
    with pytest.raises(ErlaySketchError, match="decode limit"):
        oversized.decode_short_ids()


def test_per_shard_separation_and_round_timing_are_independent() -> None:
    left, right = child_shard_ids(ROOT_SHARD_ID)
    state = ErlayPeerState(b"peer-a", b"peer-b")
    tx_id = _tx(7).tx_id()

    left_sketch = state.build_sketch(left, (tx_id,), capacity=1)
    right_sketch = state.build_sketch(right, (tx_id,), capacity=1)

    assert ERLAY_INTERVAL_MS == 8000
    assert erlay_topic_for_shard(left) == f"tensorpow/txs/{left:08x}/main"
    assert left_sketch.to_bytes() != right_sketch.to_bytes()
    with pytest.raises(ErlaySketchError, match="same shard"):
        left_sketch.difference(right_sketch)

    assert state.should_reconcile(left, 100)
    state.mark_reconciled(left, 100)
    assert not state.should_reconcile(left, 100 + ERLAY_INTERVAL_MS - 1)
    assert state.should_reconcile(left, 100 + ERLAY_INTERVAL_MS)
    assert state.should_reconcile(right, 101)
    assert state.next_reconcile_at_ms(left) == 100 + ERLAY_INTERVAL_MS
    assert state.next_reconcile_at_ms(right) == 0


def test_duplicate_and_noncanonical_items_are_rejected() -> None:
    state = ErlayPeerState(b"peer-a", b"peer-b")
    tx_id = _tx(1).tx_id()

    with pytest.raises(ErlaySketchError, match="duplicates"):
        state.build_sketch(ROOT_SHARD_ID, (tx_id, tx_id), capacity=1)
    with pytest.raises(ErlaySketchError, match="32 bytes"):
        state.build_sketch(ROOT_SHARD_ID, (b"short",), capacity=1)
    with pytest.raises(ErlaySketchError, match="nonzero"):
        ErlaySketch.from_short_ids(ROOT_SHARD_ID, (0,), capacity=1)
    with pytest.raises(ErlaySketchError, match="duplicates"):
        ErlaySketch.from_short_ids(ROOT_SHARD_ID, (5, 5), capacity=1)
    with pytest.raises(ErlaySketchError, match="canonical"):
        ErlaySetDifference(
            local_only_tx_ids=(_tx(2).tx_id(), _tx(1).tx_id()),
            remote_only_short_ids=(),
        )


def _tx(seed: int) -> Transaction:
    payload = bytes([seed % 251]) * HASH_LEN_BYTES
    return Transaction.coinbase(
        (
            Output(
                amount_matoms=seed + 1,
                template_id=TEMPLATE_PKH,
                lockheight=seed,
                payload=payload,
            ),
        ),
        lockheight=seed,
    )


def _mempool(seeds: tuple[int, ...]) -> Mempool:
    mempool = Mempool()
    for seed in seeds:
        assert mempool.add_tx(_tx(seed)).accepted
    return mempool


def _mempool_tx_ids(mempool: Mempool) -> tuple[bytes, ...]:
    return tuple(tx.tx_id() for tx in mempool.select_for_fruit(ROOT_SHARD_ID))


def _replace(data: bytes, offset: int, replacement: bytes) -> bytes:
    return data[:offset] + replacement + data[offset + len(replacement) :]


def _codec_offset() -> int:
    return len(ERLAY_SKETCH_MAGIC)


def _field_offset() -> int:
    return _codec_offset() + 2


def _shard_offset() -> int:
    return _field_offset() + 2


def _capacity_offset() -> int:
    return _shard_offset() + 4


def _count_offset() -> int:
    return _capacity_offset() + 2
