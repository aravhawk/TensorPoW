"""Deterministic UTXO set reconciliation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from tensorpow._iblt import peel_iblt_cells
from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes
from tensorpow.state.utxo import (
    OUTPOINT_BYTES,
    TX_OUTPUT_PAYLOAD_MAX_BYTES,
    UTXO,
    UTXO_FIXED_BYTES,
    Outpoint,
    UTXOSet,
)

U32_BYTES: Final[int] = 4
U16_BYTES: Final[int] = 2
U8_BYTES: Final[int] = 1
U16_MAX: Final[int] = 0xFFFF

UTXO_DIFF_MAGIC: Final[bytes] = b"TPUIBLT1"
UTXO_DIFF_CHECKSUM_PREFIX: Final[bytes] = b"TensorPoW:utxo-diff:"
UTXO_IBLT_HASH_COUNT: Final[int] = 3
UTXO_IBLT_KEY_BYTES: Final[int] = 8
UTXO_IBLT_MIN_CELLS: Final[int] = UTXO_IBLT_HASH_COUNT
UTXO_IBLT_CELL_BYTES: Final[int] = U16_BYTES + UTXO_IBLT_KEY_BYTES + U32_BYTES
UTXO_DIFF_HEADER_BYTES: Final[int] = (
    len(UTXO_DIFF_MAGIC) + (2 * HASH_LEN_BYTES) + (2 * U32_BYTES) + U16_BYTES + U8_BYTES
)
UTXO_DIFF_FIXED_BYTES: Final[int] = UTXO_DIFF_HEADER_BYTES + HASH_LEN_BYTES
UTXO_DIFF_REMOVE_ENTRY_BYTES: Final[int] = OUTPOINT_BYTES + HASH_LEN_BYTES
UTXO_DIFF_ADD_ENTRY_OVERHEAD_BYTES: Final[int] = U16_BYTES
MAX_UTXO_DIFF_ENTRIES: Final[int] = 16_384
MAX_UTXO_DIFF_BYTES: Final[int] = 64 * 1024 * 1024
MAX_UTXO_DIFF_ADD_BYTES: Final[int] = UTXO_FIXED_BYTES + TX_OUTPUT_PAYLOAD_MAX_BYTES
_UTXO_IBLT_POSITION_DOMAIN: Final[bytes] = b"TensorPoW UTXO IBLT position"
_UTXO_IBLT_CHECKSUM_DOMAIN: Final[bytes] = b"TensorPoW UTXO IBLT checksum"
_UTXO_IBLT_REMOVE_DOMAIN: Final[bytes] = b"TensorPoW UTXO IBLT remove"
_UTXO_IBLT_ADD_DOMAIN: Final[bytes] = b"TensorPoW UTXO IBLT add"


class UTXOReconciliationError(ValueError):
    """Raised when a UTXO reconciliation diff is malformed or inconsistent."""


@runtime_checkable
class UTXODiffPeer(Protocol):
    """Minimal peer surface used by the state sync wrapper."""

    def build_utxo_diff(self, local_root: bytes) -> bytes:
        """Return a canonical diff from `local_root` to the peer's UTXO root."""


@dataclass(frozen=True, slots=True)
class _RemoveRecord:
    outpoint: Outpoint
    value_hash: bytes


@dataclass(frozen=True, slots=True)
class _DecodedUTXODiff:
    base_root: bytes
    target_root: bytes
    removals: tuple[_RemoveRecord, ...]
    additions: tuple[UTXO, ...]


@dataclass(frozen=True, slots=True)
class _IBLTCell:
    count: int
    key_sum: bytes
    checksum_sum: int


def build_utxo_diff(
    local_set: UTXOSet,
    target_set: UTXOSet,
    *,
    max_entries: int = MAX_UTXO_DIFF_ENTRIES,
    max_bytes: int = MAX_UTXO_DIFF_BYTES,
) -> bytes:
    """Build a canonical bounded diff that transforms `local_set` into `target_set`."""

    _require_utxo_set("local_set", local_set)
    _require_utxo_set("target_set", target_set)
    _require_positive_int("max_entries", max_entries)
    _require_positive_int("max_bytes", max_bytes)

    local_by_outpoint = _utxos_by_outpoint(local_set)
    target_by_outpoint = _utxos_by_outpoint(target_set)

    removals: list[_RemoveRecord] = []
    additions: list[UTXO] = []
    for outpoint, local_utxo in sorted(
        local_by_outpoint.items(), key=lambda item: item[0].to_bytes()
    ):
        target_utxo = target_by_outpoint.get(outpoint)
        if target_utxo is None:
            removals.append(_RemoveRecord(outpoint=outpoint, value_hash=local_utxo.value_hash()))
        elif target_utxo.to_bytes() != local_utxo.to_bytes():
            raise UTXOReconciliationError("same outpoint has conflicting UTXO bytes")

    for outpoint, target_utxo in sorted(
        target_by_outpoint.items(), key=lambda item: item[0].to_bytes()
    ):
        if outpoint not in local_by_outpoint:
            additions.append(target_utxo)

    return _encode_utxo_diff(
        _DecodedUTXODiff(
            base_root=local_set.merkle_root(),
            target_root=target_set.merkle_root(),
            removals=tuple(removals),
            additions=tuple(additions),
        ),
        max_entries=max_entries,
        max_bytes=max_bytes,
    )


def request_utxo_diff(local_root: bytes, peer: UTXODiffPeer) -> bytes:
    """Request and preflight-check a peer diff for the caller's current UTXO root."""

    _require_hash("local_root", local_root)
    if not isinstance(peer, UTXODiffPeer):
        raise TypeError("peer must implement build_utxo_diff(local_root)")

    diff = peer.build_utxo_diff(local_root)
    decoded = _decode_utxo_diff(diff)
    if decoded.base_root != local_root:
        raise UTXOReconciliationError("peer diff base root does not match requested root")
    return diff


def apply_utxo_diff(
    local_set: UTXOSet,
    diff: bytes,
    *,
    max_entries: int = MAX_UTXO_DIFF_ENTRIES,
    max_bytes: int = MAX_UTXO_DIFF_BYTES,
) -> UTXOSet:
    """Apply a canonical UTXO reconciliation diff and validate the resulting root."""

    _require_utxo_set("local_set", local_set)
    _require_positive_int("max_entries", max_entries)
    _require_positive_int("max_bytes", max_bytes)
    decoded = _decode_utxo_diff(diff, max_entries=max_entries, max_bytes=max_bytes)

    if local_set.merkle_root() != decoded.base_root:
        raise UTXOReconciliationError("diff base root does not match local UTXO root")

    reconciled = UTXOSet(_utxos_by_outpoint(local_set).values())
    for removal in decoded.removals:
        existing = reconciled.get(removal.outpoint)
        if existing is None:
            raise UTXOReconciliationError("diff removes an outpoint absent from local set")
        if existing.value_hash() != removal.value_hash:
            raise UTXOReconciliationError("diff removal value hash does not match local UTXO")
        reconciled.remove(removal.outpoint)

    for utxo in decoded.additions:
        if reconciled.contains(utxo.outpoint):
            raise UTXOReconciliationError("diff adds an outpoint already present after removals")
        reconciled.add(utxo)

    if reconciled.merkle_root() != decoded.target_root:
        raise UTXOReconciliationError("reconciled UTXO root does not match diff target root")
    return reconciled


def _encode_utxo_diff(
    decoded: _DecodedUTXODiff,
    *,
    max_entries: int,
    max_bytes: int,
) -> bytes:
    _require_hash("base_root", decoded.base_root)
    _require_hash("target_root", decoded.target_root)
    _require_canonical_records(decoded.removals, decoded.additions, max_entries=max_entries)

    body = bytearray()
    change_keys = _change_keys(decoded.removals, decoded.additions)
    iblt_cell_count = _canonical_iblt_cell_count(len(change_keys))
    iblt_cells = _build_iblt(change_keys, cell_count=iblt_cell_count)
    for cell in iblt_cells:
        body += _encode_iblt_cell(cell)
    for removal in decoded.removals:
        body += removal.outpoint.to_bytes()
        body += removal.value_hash
    for utxo in decoded.additions:
        utxo_bytes = utxo.to_bytes()
        if len(utxo_bytes) > MAX_UTXO_DIFF_ADD_BYTES:
            raise UTXOReconciliationError("added UTXO exceeds maximum encoded size")
        body += len(utxo_bytes).to_bytes(U16_BYTES, "little")
        body += utxo_bytes

    payload = b"".join(
        (
            UTXO_DIFF_MAGIC,
            decoded.base_root,
            decoded.target_root,
            len(decoded.removals).to_bytes(U32_BYTES, "little"),
            len(decoded.additions).to_bytes(U32_BYTES, "little"),
            iblt_cell_count.to_bytes(U16_BYTES, "little"),
            bytes((UTXO_IBLT_HASH_COUNT,)),
            bytes(body),
        )
    )
    diff = payload + _diff_checksum(payload)
    if len(diff) > max_bytes:
        raise UTXOReconciliationError("UTXO diff exceeds maximum encoded size")
    return diff


def _decode_utxo_diff(
    diff: bytes,
    *,
    max_entries: int = MAX_UTXO_DIFF_ENTRIES,
    max_bytes: int = MAX_UTXO_DIFF_BYTES,
) -> _DecodedUTXODiff:
    _require_bytes("diff", diff)
    _require_positive_int("max_entries", max_entries)
    _require_positive_int("max_bytes", max_bytes)
    if len(diff) > max_bytes:
        raise UTXOReconciliationError("UTXO diff exceeds maximum encoded size")
    if len(diff) < UTXO_DIFF_FIXED_BYTES:
        raise UTXOReconciliationError("UTXO diff is truncated")

    payload = diff[:-HASH_LEN_BYTES]
    checksum = diff[-HASH_LEN_BYTES:]
    if not payload.startswith(UTXO_DIFF_MAGIC):
        raise UTXOReconciliationError("UTXO diff magic is invalid")
    if checksum != _diff_checksum(payload):
        raise UTXOReconciliationError("UTXO diff checksum mismatch")

    offset = len(UTXO_DIFF_MAGIC)
    base_root = payload[offset : offset + HASH_LEN_BYTES]
    offset += HASH_LEN_BYTES
    target_root = payload[offset : offset + HASH_LEN_BYTES]
    offset += HASH_LEN_BYTES
    remove_count = int.from_bytes(payload[offset : offset + U32_BYTES], "little")
    offset += U32_BYTES
    add_count = int.from_bytes(payload[offset : offset + U32_BYTES], "little")
    offset += U32_BYTES
    iblt_cell_count = int.from_bytes(payload[offset : offset + U16_BYTES], "little")
    offset += U16_BYTES
    iblt_hash_count = payload[offset]
    offset += U8_BYTES

    if remove_count + add_count > max_entries:
        raise UTXOReconciliationError("UTXO diff entry count exceeds maximum")
    if iblt_hash_count != UTXO_IBLT_HASH_COUNT:
        raise UTXOReconciliationError("UTXO diff IBLT hash count is non-canonical")
    if iblt_cell_count != _canonical_iblt_cell_count(remove_count + add_count):
        raise UTXOReconciliationError("UTXO diff IBLT cell count is non-canonical")

    iblt_cells: list[_IBLTCell] = []
    for _ in range(iblt_cell_count):
        end = offset + UTXO_IBLT_CELL_BYTES
        if end > len(payload):
            raise UTXOReconciliationError("UTXO diff is truncated in IBLT cells")
        iblt_cells.append(_decode_iblt_cell(payload[offset:end]))
        offset = end

    removals: list[_RemoveRecord] = []
    for _ in range(remove_count):
        end = offset + UTXO_DIFF_REMOVE_ENTRY_BYTES
        if end > len(payload):
            raise UTXOReconciliationError("UTXO diff is truncated in removals")
        removals.append(
            _RemoveRecord(
                outpoint=Outpoint.from_bytes(payload[offset : offset + OUTPOINT_BYTES]),
                value_hash=payload[offset + OUTPOINT_BYTES : end],
            )
        )
        offset = end

    additions: list[UTXO] = []
    for _ in range(add_count):
        length_end = offset + U16_BYTES
        if length_end > len(payload):
            raise UTXOReconciliationError("UTXO diff is truncated in addition length")
        utxo_len = int.from_bytes(payload[offset:length_end], "little")
        offset = length_end
        if not UTXO_FIXED_BYTES <= utxo_len <= MAX_UTXO_DIFF_ADD_BYTES:
            raise UTXOReconciliationError("UTXO diff addition length is invalid")
        end = offset + utxo_len
        if end > len(payload):
            raise UTXOReconciliationError("UTXO diff is truncated in addition bytes")
        additions.append(UTXO.from_bytes(payload[offset:end]))
        offset = end

    if offset != len(payload):
        raise UTXOReconciliationError("UTXO diff has trailing bytes")

    decoded = _DecodedUTXODiff(
        base_root=base_root,
        target_root=target_root,
        removals=tuple(removals),
        additions=tuple(additions),
    )
    _require_canonical_records(decoded.removals, decoded.additions, max_entries=max_entries)
    _validate_iblt_sketch(tuple(iblt_cells), decoded)
    return decoded


def _require_canonical_records(
    removals: tuple[_RemoveRecord, ...],
    additions: tuple[UTXO, ...],
    *,
    max_entries: int,
) -> None:
    if len(removals) + len(additions) > max_entries:
        raise UTXOReconciliationError("UTXO diff entry count exceeds maximum")

    removal_keys = tuple(removal.outpoint.to_bytes() for removal in removals)
    addition_keys = tuple(utxo.outpoint.to_bytes() for utxo in additions)
    if tuple(sorted(removal_keys)) != removal_keys or len(set(removal_keys)) != len(removal_keys):
        raise UTXOReconciliationError("UTXO diff removals are not canonical")
    if tuple(sorted(addition_keys)) != addition_keys or len(set(addition_keys)) != len(
        addition_keys
    ):
        raise UTXOReconciliationError("UTXO diff additions are not canonical")
    if set(removal_keys) & set(addition_keys):
        raise UTXOReconciliationError("UTXO diff cannot remove and add the same outpoint")
    for removal in removals:
        _require_hash("removal.value_hash", removal.value_hash)


def _change_keys(
    removals: tuple[_RemoveRecord, ...],
    additions: tuple[UTXO, ...],
) -> tuple[bytes, ...]:
    keys = tuple(
        _change_key(_UTXO_IBLT_REMOVE_DOMAIN, removal.outpoint.to_bytes() + removal.value_hash)
        for removal in removals
    ) + tuple(_change_key(_UTXO_IBLT_ADD_DOMAIN, utxo.to_bytes()) for utxo in additions)
    if len(set(keys)) != len(keys):
        raise UTXOReconciliationError("UTXO diff has duplicate IBLT change keys")
    return keys


def _change_key(domain: bytes, payload: bytes) -> bytes:
    return hash_bytes(domain + payload)[:UTXO_IBLT_KEY_BYTES]


def _canonical_iblt_cell_count(change_count: int) -> int:
    if not isinstance(change_count, int) or isinstance(change_count, bool):
        raise TypeError("change_count must be int")
    if change_count < 0:
        raise ValueError("change_count must be nonnegative")
    return max(UTXO_IBLT_MIN_CELLS, (change_count * 2) + UTXO_IBLT_HASH_COUNT)


def _build_iblt(change_keys: tuple[bytes, ...], *, cell_count: int) -> tuple[_IBLTCell, ...]:
    _require_iblt_cell_count(cell_count)
    if len(set(change_keys)) != len(change_keys):
        raise UTXOReconciliationError("UTXO diff has duplicate IBLT change keys")
    cells = [
        _IBLTCell(count=0, key_sum=bytes(UTXO_IBLT_KEY_BYTES), checksum_sum=0)
        for _ in range(cell_count)
    ]
    for key in change_keys:
        _apply_iblt_key(cells, key, delta=1)
    return tuple(cells)


def _validate_iblt_sketch(cells: tuple[_IBLTCell, ...], decoded: _DecodedUTXODiff) -> None:
    expected_keys = tuple(sorted(_change_keys(decoded.removals, decoded.additions)))
    peeled = _peel_iblt(cells)
    if peeled is None or peeled != expected_keys:
        raise UTXOReconciliationError("UTXO diff IBLT sketch does not match records")


def _peel_iblt(cells: tuple[_IBLTCell, ...]) -> tuple[bytes, ...] | None:
    peeled = peel_iblt_cells(
        cells,
        is_pure=_is_pure_iblt_cell,
        is_empty=_is_empty_iblt_cell,
        cell_key=lambda cell: cell.key_sum,
        peel_delta=lambda _cell: -1,
        apply_key=lambda working, key, delta: _apply_iblt_key(working, key, delta=delta),
    )
    if peeled is None:
        return None
    return tuple(sorted(key for key, _delta in peeled))


def _apply_iblt_key(cells: list[_IBLTCell], key: bytes, *, delta: int) -> None:
    _require_iblt_key(key)
    if delta not in (-1, 1):
        raise ValueError("IBLT delta must be -1 or 1")
    checksum = _iblt_checksum(key)
    for cell_index in _iblt_positions(key, len(cells)):
        cell = cells[cell_index]
        cells[cell_index] = _IBLTCell(
            count=cell.count + delta,
            key_sum=_xor_bytes(cell.key_sum, key),
            checksum_sum=cell.checksum_sum ^ checksum,
        )


def _is_pure_iblt_cell(cell: _IBLTCell) -> bool:
    return (
        cell.count == 1
        and _require_iblt_key(cell.key_sum) == cell.key_sum
        and cell.checksum_sum == _iblt_checksum(cell.key_sum)
    )


def _is_empty_iblt_cell(cell: _IBLTCell) -> bool:
    return cell.count == 0 and cell.key_sum == bytes(UTXO_IBLT_KEY_BYTES) and cell.checksum_sum == 0


def _iblt_positions(key: bytes, cell_count: int) -> tuple[int, ...]:
    _require_iblt_key(key)
    _require_iblt_cell_count(cell_count)
    positions: list[int] = []
    attempt = 0
    while len(positions) < UTXO_IBLT_HASH_COUNT:
        digest = hash_bytes(
            _UTXO_IBLT_POSITION_DOMAIN + key + attempt.to_bytes(U16_BYTES, "little")
        )
        position = int.from_bytes(digest[:U32_BYTES], "little") % cell_count
        if position not in positions:
            positions.append(position)
        attempt += 1
        if attempt > U16_MAX:
            raise RuntimeError("unable to derive unique UTXO IBLT positions")
    return tuple(positions)


def _iblt_checksum(key: bytes) -> int:
    _require_iblt_key(key)
    return int.from_bytes(hash_bytes(_UTXO_IBLT_CHECKSUM_DOMAIN + key)[:U32_BYTES], "little")


def _encode_iblt_cell(cell: _IBLTCell) -> bytes:
    if cell.count < 0 or cell.count > U16_MAX:
        raise UTXOReconciliationError("UTXO diff IBLT cell count outside uint16 range")
    _require_iblt_key(cell.key_sum)
    return (
        cell.count.to_bytes(U16_BYTES, "little")
        + cell.key_sum
        + cell.checksum_sum.to_bytes(U32_BYTES, "little")
    )


def _decode_iblt_cell(data: bytes) -> _IBLTCell:
    if len(data) != UTXO_IBLT_CELL_BYTES:
        raise UTXOReconciliationError("UTXO diff IBLT cell has invalid length")
    count = int.from_bytes(data[:U16_BYTES], "little")
    key_sum = data[U16_BYTES : U16_BYTES + UTXO_IBLT_KEY_BYTES]
    checksum_sum = int.from_bytes(data[U16_BYTES + UTXO_IBLT_KEY_BYTES :], "little")
    return _IBLTCell(count=count, key_sum=key_sum, checksum_sum=checksum_sum)


def _require_iblt_cell_count(cell_count: int) -> None:
    if not isinstance(cell_count, int) or isinstance(cell_count, bool):
        raise TypeError("cell_count must be int")
    if not UTXO_IBLT_MIN_CELLS <= cell_count <= U16_MAX:
        raise ValueError("UTXO IBLT cell count is outside supported range")


def _require_iblt_key(key: bytes) -> bytes:
    _require_bytes("IBLT key", key)
    if len(key) != UTXO_IBLT_KEY_BYTES:
        raise ValueError("IBLT key has invalid length")
    return key


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("byte strings must have matching length")
    return bytes(left_byte ^ right_byte for left_byte, right_byte in zip(left, right, strict=True))


def _utxos_by_outpoint(utxo_set: UTXOSet) -> dict[Outpoint, UTXO]:
    return {utxo.outpoint: utxo for utxo in utxo_set.utxos()}


def _diff_checksum(payload: bytes) -> bytes:
    _require_bytes("payload", payload)
    return hash_bytes(UTXO_DIFF_CHECKSUM_PREFIX + payload)


def _require_utxo_set(name: str, value: UTXOSet) -> None:
    if not isinstance(value, UTXOSet):
        raise TypeError(f"{name} must be UTXOSet")


def _require_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")


def _require_hash(name: str, value: bytes) -> None:
    _require_bytes(name, value)
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "MAX_UTXO_DIFF_BYTES",
    "MAX_UTXO_DIFF_ENTRIES",
    "UTXO_DIFF_FIXED_BYTES",
    "UTXO_DIFF_REMOVE_ENTRY_BYTES",
    "UTXO_IBLT_CELL_BYTES",
    "UTXODiffPeer",
    "UTXOReconciliationError",
    "apply_utxo_diff",
    "build_utxo_diff",
    "request_utxo_diff",
]
