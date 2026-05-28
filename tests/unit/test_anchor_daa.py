"""Tests for anchor WTEMA target adjustment."""

from __future__ import annotations

from itertools import pairwise

import pytest

from tensorpow.consensus.anchor_daa import (
    ANCHOR_INITIAL_TARGET_LE,
    ANCHOR_INTERVAL_MS,
    ANCHOR_MAX_TARGET_LE,
    ANCHOR_MIN_TARGET_LE,
    GENESIS_PARENT_HASH,
    AnchorRecord,
    anchor_work_weight,
    int_to_target,
    next_anchor_target,
    target_to_int,
)


def _h(index: int) -> bytes:
    return index.to_bytes(32, "big")


def _record(index: int, parent: bytes, timestamp_ms: int, target: bytes) -> AnchorRecord:
    return AnchorRecord(
        anchor_hash=_h(index),
        parent_anchor=parent,
        timestamp_ms=timestamp_ms,
        target=target,
    )


def test_empty_and_single_anchor_history_use_initial_target() -> None:
    genesis = _record(1, GENESIS_PARENT_HASH, 0, ANCHOR_INITIAL_TARGET_LE)

    assert next_anchor_target(()) == ANCHOR_INITIAL_TARGET_LE
    assert next_anchor_target((genesis,)) == ANCHOR_INITIAL_TARGET_LE


def test_steady_anchor_intervals_keep_target_stable() -> None:
    history = [
        _record(1, GENESIS_PARENT_HASH, 0, ANCHOR_INITIAL_TARGET_LE),
        _record(2, _h(1), ANCHOR_INTERVAL_MS, ANCHOR_INITIAL_TARGET_LE),
        _record(3, _h(2), ANCHOR_INTERVAL_MS * 2, ANCHOR_INITIAL_TARGET_LE),
    ]

    assert next_anchor_target(history) == ANCHOR_INITIAL_TARGET_LE


def test_fast_intervals_lower_target_and_slow_intervals_raise_target() -> None:
    fast_history = [
        _record(1, GENESIS_PARENT_HASH, 0, ANCHOR_INITIAL_TARGET_LE),
        _record(2, _h(1), ANCHOR_INTERVAL_MS // 2, ANCHOR_INITIAL_TARGET_LE),
        _record(3, _h(2), ANCHOR_INTERVAL_MS, ANCHOR_INITIAL_TARGET_LE),
    ]
    slow_history = [
        _record(1, GENESIS_PARENT_HASH, 0, ANCHOR_INITIAL_TARGET_LE),
        _record(2, _h(1), ANCHOR_INTERVAL_MS * 2, ANCHOR_INITIAL_TARGET_LE),
        _record(3, _h(2), ANCHOR_INTERVAL_MS * 4, ANCHOR_INITIAL_TARGET_LE),
    ]

    assert target_to_int(next_anchor_target(fast_history)) < target_to_int(ANCHOR_INITIAL_TARGET_LE)
    assert target_to_int(next_anchor_target(slow_history)) > target_to_int(ANCHOR_INITIAL_TARGET_LE)


def test_wtema_moves_monotonically_under_sustained_hashrate_change() -> None:
    history = [_record(1, GENESIS_PARENT_HASH, 0, ANCHOR_INITIAL_TARGET_LE)]
    targets = []

    for index in range(2, 35):
        target = next_anchor_target(history)
        timestamp_ms = history[-1].timestamp_ms + ANCHOR_INTERVAL_MS // 2
        history.append(_record(index, history[-1].anchor_hash, timestamp_ms, target))
        targets.append(target_to_int(next_anchor_target(history)))

    assert all(later < earlier for earlier, later in pairwise(targets))


def test_target_bounds_and_reward_weight_are_enforced() -> None:
    min_history = [
        _record(1, GENESIS_PARENT_HASH, 0, ANCHOR_MIN_TARGET_LE),
        _record(2, _h(1), 1, ANCHOR_MIN_TARGET_LE),
    ]
    max_history = [
        _record(1, GENESIS_PARENT_HASH, 0, ANCHOR_MAX_TARGET_LE),
        _record(2, _h(1), ANCHOR_INTERVAL_MS * 10, ANCHOR_MAX_TARGET_LE),
    ]

    assert next_anchor_target(min_history) == ANCHOR_MIN_TARGET_LE
    assert next_anchor_target(max_history) == ANCHOR_MAX_TARGET_LE
    assert anchor_work_weight(ANCHOR_INITIAL_TARGET_LE) == 1000
    assert int_to_target(target_to_int(ANCHOR_INITIAL_TARGET_LE)) == ANCHOR_INITIAL_TARGET_LE


def test_anchor_daa_rejects_malformed_history_and_targets() -> None:
    genesis = _record(1, GENESIS_PARENT_HASH, 0, ANCHOR_INITIAL_TARGET_LE)

    with pytest.raises(ValueError, match="bounds"):
        AnchorRecord(_h(2), _h(1), 1, bytes(32))
    with pytest.raises(ValueError, match="bytes"):
        AnchorRecord(b"short", _h(1), 1, ANCHOR_INITIAL_TARGET_LE)
    with pytest.raises(ValueError, match="parent_anchor"):
        next_anchor_target(
            (
                genesis,
                _record(2, _h(99), ANCHOR_INTERVAL_MS, ANCHOR_INITIAL_TARGET_LE),
            )
        )
    with pytest.raises(ValueError, match="timestamps"):
        next_anchor_target(
            (
                genesis,
                _record(2, _h(1), 0, ANCHOR_INITIAL_TARGET_LE),
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        next_anchor_target(
            (
                genesis,
                AnchorRecord(_h(1), _h(1), ANCHOR_INTERVAL_MS, ANCHOR_INITIAL_TARGET_LE),
            )
        )
    with pytest.raises(TypeError):
        next_anchor_target((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bounds"):
        int_to_target(0)
