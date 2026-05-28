"""Anchor target adjustment helpers using WTEMA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.pow.kernel import (
    ANCHOR_INITIAL_TARGET_LE,
    ANCHOR_MAX_TARGET_LE,
    ANCHOR_MIN_TARGET_LE,
    FRUIT_TARGET_LE,
)

AnchorHash = bytes

TARGET_BITS: Final[int] = 256
TARGET_BYTES: Final[int] = 32
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF
GENESIS_PARENT_HASH: Final[bytes] = bytes(HASH_LEN_BYTES)

ANCHOR_INTERVAL_MS: Final[int] = 60000
WTEMA_WINDOW_ANCHORS: Final[int] = 100
WTEMA_ALPHA_NUM: Final[int] = 1
WTEMA_ALPHA_DEN: Final[int] = 8
WTEMA_MAX_ADJUSTMENT_FACTOR: Final[int] = 4

FIXED_POINT_SCALE: Final[int] = 1 << 64

_ANCHOR_MIN_TARGET_INT: Final[int] = int.from_bytes(ANCHOR_MIN_TARGET_LE, "little")
_ANCHOR_MAX_TARGET_INT: Final[int] = int.from_bytes(ANCHOR_MAX_TARGET_LE, "little")
_FRUIT_TARGET_INT: Final[int] = int.from_bytes(FRUIT_TARGET_LE, "little")


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    """Minimal parent-chain anchor record for target adjustment."""

    anchor_hash: AnchorHash
    parent_anchor: AnchorHash
    timestamp_ms: int
    target: bytes

    def __post_init__(self) -> None:
        _require_hash("anchor_hash", self.anchor_hash)
        _require_hash("parent_anchor", self.parent_anchor)
        _require_u64("timestamp_ms", self.timestamp_ms)
        _require_target("target", self.target)


def next_anchor_target(history: Sequence[AnchorRecord]) -> bytes:
    """Return the target for the next anchor on a parent-chain history."""

    records = _normalize_history(history)
    if not records:
        return ANCHOR_INITIAL_TARGET_LE
    _validate_parent_chain(records)
    if len(records) == 1:
        return records[-1].target

    previous_target = target_to_int(records[-1].target)
    ratio_fp = _wtema_ratio_fp(_recent_intervals(records))
    adjusted = max(1, previous_target * ratio_fp // FIXED_POINT_SCALE)

    movement_floor = _ceil_div(previous_target, WTEMA_MAX_ADJUSTMENT_FACTOR)
    movement_ceiling = previous_target * WTEMA_MAX_ADJUSTMENT_FACTOR
    adjusted = min(max(adjusted, movement_floor), movement_ceiling)
    adjusted = min(max(adjusted, _ANCHOR_MIN_TARGET_INT), _ANCHOR_MAX_TARGET_INT)
    return int_to_target(adjusted)


def target_to_int(target: bytes) -> int:
    """Interpret a 32-byte little-endian target as an integer."""

    _require_target("target", target)
    return int.from_bytes(target, "little")


def int_to_target(value: int) -> bytes:
    """Encode a target integer as 32 little-endian bytes."""

    if not isinstance(value, int):
        raise TypeError("value must be int")
    if not _ANCHOR_MIN_TARGET_INT <= value <= _ANCHOR_MAX_TARGET_INT:
        raise ValueError("value outside anchor target bounds")
    return value.to_bytes(TARGET_BYTES, "little")


def anchor_work_weight(active_target: bytes) -> int:
    """Return anchor reward weight from the active anchor target."""

    target_int = target_to_int(active_target)
    return min(_FRUIT_TARGET_INT // target_int, U64_MAX)


def _normalize_history(history: object) -> tuple[AnchorRecord, ...]:
    if isinstance(history, bytes | bytearray | str):
        raise TypeError("history must be a sequence of AnchorRecord")
    if not isinstance(history, Sequence):
        raise TypeError("history must be a sequence of AnchorRecord")
    records = tuple(history)
    for record in records:
        if not isinstance(record, AnchorRecord):
            raise TypeError("history must contain only AnchorRecord values")
    return records


def _validate_parent_chain(records: tuple[AnchorRecord, ...]) -> None:
    seen: set[AnchorHash] = set()
    previous: AnchorRecord | None = None
    for record in records:
        if record.anchor_hash in seen:
            raise ValueError("anchor history must not contain duplicate hashes")
        seen.add(record.anchor_hash)
        if previous is not None:
            if record.parent_anchor != previous.anchor_hash:
                raise ValueError("anchor history must follow parent_anchor links")
            if record.timestamp_ms <= previous.timestamp_ms:
                raise ValueError("anchor timestamps must be strictly increasing")
        previous = record


def _recent_intervals(records: tuple[AnchorRecord, ...]) -> tuple[int, ...]:
    start = max(1, len(records) - WTEMA_WINDOW_ANCHORS)
    return tuple(
        _clamp_interval(records[index].timestamp_ms - records[index - 1].timestamp_ms)
        for index in range(start, len(records))
    )


def _clamp_interval(interval_ms: int) -> int:
    min_interval = ANCHOR_INTERVAL_MS // WTEMA_MAX_ADJUSTMENT_FACTOR
    max_interval = ANCHOR_INTERVAL_MS * WTEMA_MAX_ADJUSTMENT_FACTOR
    return min(max(interval_ms, min_interval), max_interval)


def _wtema_ratio_fp(intervals: tuple[int, ...]) -> int:
    ratio_fp = FIXED_POINT_SCALE
    decay = WTEMA_ALPHA_DEN - WTEMA_ALPHA_NUM
    for interval_ms in intervals:
        sample_fp = interval_ms * FIXED_POINT_SCALE // ANCHOR_INTERVAL_MS
        ratio_fp = (decay * ratio_fp + WTEMA_ALPHA_NUM * sample_fp) // WTEMA_ALPHA_DEN
    return ratio_fp


def _require_hash(name: str, value: AnchorHash) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _require_target(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != TARGET_BYTES:
        raise ValueError(f"{name} must be {TARGET_BYTES} bytes")
    target_int = int.from_bytes(value, "little")
    if not _ANCHOR_MIN_TARGET_INT <= target_int <= _ANCHOR_MAX_TARGET_INT:
        raise ValueError(f"{name} outside anchor target bounds")


def _require_u64(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= U64_MAX:
        raise ValueError(f"{name} outside uint64 range")


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


__all__ = [
    "ANCHOR_INITIAL_TARGET_LE",
    "ANCHOR_INTERVAL_MS",
    "ANCHOR_MAX_TARGET_LE",
    "ANCHOR_MIN_TARGET_LE",
    "FRUIT_TARGET_LE",
    "GENESIS_PARENT_HASH",
    "TARGET_BITS",
    "TARGET_BYTES",
    "WTEMA_ALPHA_DEN",
    "WTEMA_ALPHA_NUM",
    "WTEMA_MAX_ADJUSTMENT_FACTOR",
    "WTEMA_WINDOW_ANCHORS",
    "AnchorRecord",
    "anchor_work_weight",
    "int_to_target",
    "next_anchor_target",
    "target_to_int",
]
