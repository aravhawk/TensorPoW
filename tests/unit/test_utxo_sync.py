"""Tests for deterministic UTXO reconciliation diffs."""

from __future__ import annotations

import signal
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import pytest

import tensorpow.state.sync as sync
from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.state.sync import (
    UTXO_DIFF_MAGIC,
    UTXOReconciliationError,
    apply_utxo_diff,
    build_utxo_diff,
    request_utxo_diff,
)
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint, UTXOSet


def _outpoint(seed: int) -> Outpoint:
    return Outpoint(seed.to_bytes(32, "little"), seed)


def _utxo(seed: int, *, amount: int | None = None, payload: bytes = b"") -> UTXO:
    return UTXO(
        outpoint=_outpoint(seed),
        amount_matoms=seed + 1 if amount is None else amount,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=bytes([seed % 251]) * HASH_LEN_BYTES,
        lockheight=seed,
        payload=payload,
    )


def _set(seeds: Iterable[int]) -> UTXOSet:
    return UTXOSet(_utxo(seed) for seed in seeds)


def _encode_diff(
    *,
    base_root: bytes,
    target_root: bytes,
    removals: tuple[tuple[Outpoint, bytes], ...] = (),
    additions: tuple[UTXO, ...] = (),
) -> bytes:
    remove_records = tuple(
        sync._RemoveRecord(outpoint=outpoint, value_hash=value_hash)
        for outpoint, value_hash in removals
    )
    change_count = len(removals) + len(additions)
    iblt_cell_count = sync._canonical_iblt_cell_count(change_count)
    try:
        change_keys = sync._change_keys(remove_records, additions)
    except UTXOReconciliationError:
        change_keys = ()
    if len(change_keys) == change_count:
        iblt_cells = sync._build_iblt(change_keys, cell_count=iblt_cell_count)
    else:
        iblt_cells = tuple(
            sync._IBLTCell(count=0, key_sum=bytes(sync.UTXO_IBLT_KEY_BYTES), checksum_sum=0)
            for _ in range(iblt_cell_count)
        )
    body = bytearray()
    for cell in iblt_cells:
        body += sync._encode_iblt_cell(cell)
    for outpoint, value_hash in removals:
        body += outpoint.to_bytes()
        body += value_hash
    for utxo in additions:
        utxo_bytes = utxo.to_bytes()
        body += len(utxo_bytes).to_bytes(sync.U16_BYTES, "little")
        body += utxo_bytes
    payload = b"".join(
        (
            sync.UTXO_DIFF_MAGIC,
            base_root,
            target_root,
            len(removals).to_bytes(sync.U32_BYTES, "little"),
            len(additions).to_bytes(sync.U32_BYTES, "little"),
            iblt_cell_count.to_bytes(sync.U16_BYTES, "little"),
            bytes((sync.UTXO_IBLT_HASH_COUNT,)),
            bytes(body),
        )
    )
    return payload + sync._diff_checksum(payload)


def _replace_target_root(diff: bytes, target_root: bytes) -> bytes:
    payload = bytearray(diff[:-HASH_LEN_BYTES])
    offset = len(sync.UTXO_DIFF_MAGIC) + HASH_LEN_BYTES
    payload[offset : offset + HASH_LEN_BYTES] = target_root
    return bytes(payload) + sync._diff_checksum(bytes(payload))


class _Peer:
    def __init__(self, local_set: UTXOSet, target_set: UTXOSet) -> None:
        self._local_set = local_set
        self._target_set = target_set

    def build_utxo_diff(self, local_root: bytes) -> bytes:
        assert local_root == self._local_set.merkle_root()
        return build_utxo_diff(self._local_set, self._target_set)


def test_build_request_and_apply_utxo_diff_round_trip() -> None:
    local_set = _set((1, 2, 3, 4))
    target_set = _set((2, 3, 4, 5, 6))
    diff = request_utxo_diff(local_set.merkle_root(), _Peer(local_set, target_set))

    reconciled = apply_utxo_diff(local_set, diff)

    assert diff.startswith(UTXO_DIFF_MAGIC)
    assert reconciled.merkle_root() == target_set.merkle_root()
    assert [utxo.outpoint for utxo in reconciled.utxos()] == [
        _outpoint(2),
        _outpoint(3),
        _outpoint(4),
        _outpoint(5),
        _outpoint(6),
    ]
    assert local_set.contains(_outpoint(1))
    assert not local_set.contains(_outpoint(5))


def test_apply_utxo_diff_rejects_base_and_target_root_mismatches() -> None:
    local_set = _set((10, 11, 12))
    target_set = _set((10, 12, 13))
    diff = build_utxo_diff(local_set, target_set)

    with pytest.raises(UTXOReconciliationError, match="base root"):
        apply_utxo_diff(_set((10, 11)), diff)

    forged_target = _replace_target_root(diff, bytes([7]) * HASH_LEN_BYTES)
    with pytest.raises(UTXOReconciliationError, match="target root"):
        apply_utxo_diff(local_set, forged_target)


def test_malformed_truncated_and_checksum_failed_diffs_are_rejected() -> None:
    local_set = _set((20, 21))
    diff = build_utxo_diff(local_set, _set((20, 21, 22)))

    with pytest.raises(UTXOReconciliationError, match="truncated"):
        apply_utxo_diff(local_set, diff[: sync.UTXO_DIFF_FIXED_BYTES - 1])

    bad_checksum = bytearray(diff)
    bad_checksum[-1] ^= 0x01
    with pytest.raises(UTXOReconciliationError, match="checksum"):
        apply_utxo_diff(local_set, bytes(bad_checksum))

    payload = bytearray(diff[:-HASH_LEN_BYTES])
    payload[0] ^= 0x01
    bad_magic = bytes(payload) + sync._diff_checksum(bytes(payload))
    with pytest.raises(UTXOReconciliationError, match="magic"):
        apply_utxo_diff(local_set, bad_magic)

    payload = diff[:-HASH_LEN_BYTES] + b"\x00"
    trailing = payload + sync._diff_checksum(payload)
    with pytest.raises(UTXOReconciliationError, match="trailing"):
        apply_utxo_diff(local_set, trailing)

    bad_iblt = bytearray(diff)
    iblt_offset = sync.UTXO_DIFF_HEADER_BYTES
    bad_iblt[iblt_offset + sync.U16_BYTES] ^= 0x01
    bad_iblt_payload = bytes(bad_iblt[:-HASH_LEN_BYTES])
    bad_iblt = bytearray(bad_iblt_payload + sync._diff_checksum(bad_iblt_payload))
    with pytest.raises(UTXOReconciliationError, match="IBLT"):
        apply_utxo_diff(local_set, bytes(bad_iblt))


def test_malicious_diffs_are_detected_before_state_is_accepted() -> None:
    local_set = _set((30, 31))
    existing = local_set.get(_outpoint(30))
    assert existing is not None

    remove_missing = _encode_diff(
        base_root=local_set.merkle_root(),
        target_root=local_set.merkle_root(),
        removals=((_outpoint(99), bytes(HASH_LEN_BYTES)),),
    )
    with pytest.raises(UTXOReconciliationError, match="absent"):
        apply_utxo_diff(local_set, remove_missing)

    wrong_remove_hash = _encode_diff(
        base_root=local_set.merkle_root(),
        target_root=_set((31,)).merkle_root(),
        removals=((existing.outpoint, bytes([1]) * HASH_LEN_BYTES),),
    )
    with pytest.raises(UTXOReconciliationError, match="value hash"):
        apply_utxo_diff(local_set, wrong_remove_hash)

    add_existing = _encode_diff(
        base_root=local_set.merkle_root(),
        target_root=local_set.merkle_root(),
        additions=(existing,),
    )
    with pytest.raises(UTXOReconciliationError, match="already present"):
        apply_utxo_diff(local_set, add_existing)

    remove_and_add_same_outpoint = _encode_diff(
        base_root=local_set.merkle_root(),
        target_root=local_set.merkle_root(),
        removals=((existing.outpoint, existing.value_hash()),),
        additions=(existing,),
    )
    with pytest.raises(UTXOReconciliationError, match="same outpoint"):
        apply_utxo_diff(local_set, remove_and_add_same_outpoint)


def test_duplicate_or_unsorted_diff_records_are_non_canonical() -> None:
    local_set = _set((40,))
    added = _utxo(41)

    duplicate_add = _encode_diff(
        base_root=local_set.merkle_root(),
        target_root=UTXOSet([_utxo(40), added]).merkle_root(),
        additions=(added, added),
    )
    with pytest.raises(UTXOReconciliationError, match="additions"):
        apply_utxo_diff(local_set, duplicate_add)

    unsorted_adds = _encode_diff(
        base_root=local_set.merkle_root(),
        target_root=UTXOSet([_utxo(40), _utxo(41), _utxo(42)]).merkle_root(),
        additions=(_utxo(42), _utxo(41)),
    )
    with pytest.raises(UTXOReconciliationError, match="additions"):
        apply_utxo_diff(local_set, unsorted_adds)


def test_request_utxo_diff_checks_returned_base_root() -> None:
    local_set = _set((50, 51))
    target_set = _set((50, 51, 52))

    class BadPeer:
        def build_utxo_diff(self, local_root: bytes) -> bytes:
            assert local_root == local_set.merkle_root()
            return build_utxo_diff(UTXOSet(), target_set)

    with pytest.raises(UTXOReconciliationError, match="base root"):
        request_utxo_diff(local_set.merkle_root(), BadPeer())


def test_utxo_diff_rejects_conflicting_same_outpoint_and_limits() -> None:
    local_set = UTXOSet([_utxo(1, amount=1)])
    target_set = UTXOSet([_utxo(1, amount=2)])

    with pytest.raises(UTXOReconciliationError, match="conflicting"):
        build_utxo_diff(local_set, target_set)
    with pytest.raises(ValueError, match="positive"):
        build_utxo_diff(UTXOSet(), UTXOSet(), max_entries=0)
    with pytest.raises(UTXOReconciliationError, match="entry count"):
        build_utxo_diff(UTXOSet(), UTXOSet([_utxo(1), _utxo(2)]), max_entries=1)
    with pytest.raises(UTXOReconciliationError, match="maximum encoded size"):
        build_utxo_diff(UTXOSet(), UTXOSet([_utxo(1)]), max_bytes=1)


@pytest.mark.parametrize(
    ("base_count", "removed", "added"),
    (
        (8, 2, 2),
        (64, 6, 6),
    ),
)
def test_diff_size_stays_proportional_to_small_and_medium_change_sets(
    base_count: int,
    removed: int,
    added: int,
) -> None:
    local_utxos = [_utxo(seed + 1, payload=b"x" * (seed % 7)) for seed in range(base_count)]
    removed_outpoints = {utxo.outpoint for utxo in local_utxos[:removed]}
    added_utxos = [_utxo(base_count + seed + 1, payload=b"new") for seed in range(added)]
    target_utxos = [
        utxo for utxo in local_utxos if utxo.outpoint not in removed_outpoints
    ] + added_utxos
    local_set = UTXOSet(local_utxos)
    target_set = UTXOSet(target_utxos)

    diff = build_utxo_diff(local_set, target_set)
    encoded_change_size = removed * sync.UTXO_DIFF_REMOVE_ENTRY_BYTES + sum(
        sync.UTXO_DIFF_ADD_ENTRY_OVERHEAD_BYTES + len(utxo.to_bytes()) for utxo in added_utxos
    )
    iblt_size = _iblt_size(removed + added)
    full_target_size = sum(len(utxo.to_bytes()) for utxo in target_utxos)

    assert len(diff) == sync.UTXO_DIFF_FIXED_BYTES + iblt_size + encoded_change_size
    assert len(diff) <= (2 * encoded_change_size)
    assert len(diff) < full_target_size


def test_empty_node_diff_is_less_than_twice_full_download_for_medium_fixture() -> None:
    target_utxos = [_utxo(seed + 1, payload=b"medium") for seed in range(32)]
    target_set = UTXOSet(target_utxos)

    diff = build_utxo_diff(UTXOSet(), target_set)
    full_download_size = sum(len(utxo.to_bytes()) for utxo in target_utxos)

    assert apply_utxo_diff(UTXOSet(), diff).merkle_root() == target_set.merkle_root()
    assert len(diff) == sync.UTXO_DIFF_FIXED_BYTES + sum(
        sync.UTXO_DIFF_ADD_ENTRY_OVERHEAD_BYTES + len(utxo.to_bytes()) for utxo in target_utxos
    ) + _iblt_size(len(target_utxos))
    assert len(diff) <= (2 * full_download_size)


def test_change_key_collision_falls_back_to_full_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_set = _set((1, 2))
    target_set = _set((3, 4))

    def colliding_change_key(_domain: bytes, _payload: bytes) -> bytes:
        return bytes(sync.UTXO_IBLT_KEY_BYTES)

    monkeypatch.setattr(sync, "_change_key", colliding_change_key)

    diff = build_utxo_diff(local_set, target_set)
    decoded = sync._decode_utxo_diff(diff)
    reconciled = apply_utxo_diff(local_set, diff)

    assert decoded.full_snapshot is True
    assert decoded.removals == ()
    assert decoded.additions == target_set.utxos()
    assert reconciled.utxos() == target_set.utxos()


def test_iblt_peel_rejects_non_self_clearing_pure_cell() -> None:
    cell_count = 5
    key = b"trapkey2"
    trap_index = _outside_utxo_positions(key, cell_count)
    cells = [
        sync._IBLTCell(count=0, key_sum=bytes(sync.UTXO_IBLT_KEY_BYTES), checksum_sum=0)
        for _ in range(cell_count)
    ]
    cells[trap_index] = sync._IBLTCell(
        count=1,
        key_sum=key,
        checksum_sum=sync._iblt_checksum(key),
    )

    with _deadline():
        assert sync._peel_iblt(tuple(cells)) is None


def _iblt_size(change_count: int) -> int:
    return sync._canonical_iblt_cell_count(change_count) * sync.UTXO_IBLT_CELL_BYTES


def _outside_utxo_positions(key: bytes, cell_count: int) -> int:
    positions = set(sync._iblt_positions(key, cell_count))
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
