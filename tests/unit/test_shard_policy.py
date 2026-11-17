"""M-2: deterministic shard-tree and fee-floor transition."""

from __future__ import annotations

import pytest

import tensorpow.consensus.shard_policy as shard_policy_module
from tensorpow.consensus.shard_policy import (
    CoveredFruitStats,
    FeeSample,
    ShardPolicyState,
    next_fee_floor,
    next_shard_policy_state,
)
from tensorpow.mempool import (
    MAX_FRUIT_PAYLOAD_BYTES,
    ROOT_SHARD_ID,
    SHARD_MERGE_WINDOW_ANCHORS,
    SHARD_SPLIT_WINDOW_ANCHORS,
    ShardTree,
    child_shard_ids,
)

LEFT, RIGHT = child_shard_ids(ROOT_SHARD_ID)


def _root_state(floor: int = 0) -> ShardPolicyState:
    return ShardPolicyState.genesis(ShardTree(), {ROOT_SHARD_ID: floor})


def _full(shard_id: int, fees: int = 0) -> CoveredFruitStats:
    return CoveredFruitStats(shard_id, MAX_FRUIT_PAYLOAD_BYTES, fees)


def test_fee_floor_is_ewma_of_window_rate() -> None:
    state = _root_state(floor=800)
    stats = (CoveredFruitStats(ROOT_SHARD_ID, payload_bytes=2_000, fees_matoms=10_000),)

    following = next_shard_policy_state(state, stats)

    recent_rate = 10_000 * 1000 // 2_000
    assert following.fee_floors == {ROOT_SHARD_ID: (7 * 800 + recent_rate) // 8}
    assert next_fee_floor(800, ()) == 7 * 800 // 8
    assert next_fee_floor(0, (FeeSample(0, 0, 0),)) == 0


def test_transition_is_pure_and_deterministic() -> None:
    state = _root_state(floor=5)
    stats = (_full(ROOT_SHARD_ID, fees=123), _full(ROOT_SHARD_ID, fees=7))

    first = next_shard_policy_state(state, stats)
    second = next_shard_policy_state(state, stats)

    assert state.fee_floors == {ROOT_SHARD_ID: 5}
    assert len(state.fee_windows[ROOT_SHARD_ID]) == 0
    assert first.tree == second.tree
    assert first.fee_floors == second.fee_floors
    assert list(first.fee_windows[ROOT_SHARD_ID]) == list(second.fee_windows[ROOT_SHARD_ID])
    assert first.next_sequence == 2


def test_fruits_on_non_leaf_shards_do_not_count() -> None:
    state = _root_state()
    following = next_shard_policy_state(state, (_full(LEFT, fees=10**9),))

    assert following.fee_floors == {ROOT_SHARD_ID: 0}
    assert following.next_sequence == 0


def test_split_requires_sustained_utilization_and_children_inherit() -> None:
    state = _root_state(floor=64)
    for _ in range(SHARD_SPLIT_WINDOW_ANCHORS - 1):
        state = next_shard_policy_state(state, (_full(ROOT_SHARD_ID, fees=8_192),))
        assert state.tree == ShardTree()

    split = next_shard_policy_state(state, (_full(ROOT_SHARD_ID, fees=8_192),))

    assert split.tree.leaf_shard_ids == (LEFT, RIGHT)
    assert split.fee_floors[LEFT] == split.fee_floors[RIGHT]
    assert split.fee_floors[LEFT] == next_fee_floor(
        state.fee_floors[ROOT_SHARD_ID], tuple(split.fee_windows[LEFT])
    )
    assert len(split.utilization[LEFT]) == 0
    assert list(split.fee_windows[LEFT]) == list(split.fee_windows[RIGHT])


def test_one_quiet_anchor_resets_split_window() -> None:
    state = _root_state()
    for index in range(SHARD_SPLIT_WINDOW_ANCHORS + 3):
        covered = () if index == 5 else (_full(ROOT_SHARD_ID),)
        state = next_shard_policy_state(state, covered)
        if index < 5 + SHARD_SPLIT_WINDOW_ANCHORS:
            assert state.tree == ShardTree()


def test_siblings_merge_after_long_quiet_window_with_max_floor() -> None:
    state = ShardPolicyState.genesis(ShardTree((LEFT, RIGHT)), {LEFT: 40, RIGHT: 900})
    for _ in range(SHARD_MERGE_WINDOW_ANCHORS - 1):
        state = next_shard_policy_state(state, ())
        assert state.tree.leaf_shard_ids == (LEFT, RIGHT)
    left_floor, right_floor = state.fee_floors[LEFT], state.fee_floors[RIGHT]

    merged = next_shard_policy_state(state, ())

    assert merged.tree == ShardTree()
    assert merged.fee_floors[ROOT_SHARD_ID] == next_fee_floor(max(left_floor, right_floor), ())


def test_split_budget_keeps_tree_within_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shard_policy_module, "MAX_SHARD_LEAVES", 2)
    state = ShardPolicyState.genesis(ShardTree((LEFT, RIGHT)), {LEFT: 0, RIGHT: 0})
    for _ in range(SHARD_SPLIT_WINDOW_ANCHORS + 2):
        state = next_shard_policy_state(state, (_full(LEFT), _full(RIGHT)))

    assert state.tree.leaf_shard_ids == (LEFT, RIGHT)
