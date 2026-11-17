"""Deterministic per-anchor shard-tree and fee-floor transition.

Anchors commit the shard tree and per-leaf fee floors that apply to the next anchor
interval. Validators MUST recompute both from committed history and reject an anchor
whose commitments differ; otherwise a single anchor miner could set any floor (halting
or freeing all transactions) or reshape shards at will.

Everything here is integer arithmetic over data committed by the anchor chain (the
covered fruits in canonical order and the parent state), so every node derives the
same result.

Rules (see PROTOCOL_SPEC.md sections 11 and 12):

* Every covered fruit whose ``shard_id`` is a leaf of the parent tree contributes one
  fee-floor sample ``(fees, payload_bytes)`` to that leaf's window of the last
  ``FEE_FLOOR_WINDOW_FRUITS`` confirmed fruits, in canonical covered order.
* Every parent leaf records one utilization sample per anchor:
  ``payload_bytes_confirmed`` over ``max(1, covered_fruits_in_leaf)`` slots.
* A leaf splits when ``should_split`` holds over its last
  ``SHARD_SPLIT_WINDOW_ANCHORS`` samples; sibling leaves merge when ``should_merge``
  holds over both their last ``SHARD_MERGE_WINDOW_ANCHORS`` samples. Splits that
  would push the serialized tree past ``SHARD_TREE_MAX_BYTES`` are dropped from the
  highest shard id down.
* Next floor per leaf: ``max(FEE_FLOOR_MIN, (7 * previous + 1 * recent_rate) // 8)``
  with ``recent_rate = sum(fees) * BYTES_PER_KB // max(1, sum(payload_bytes))`` over the
  leaf window (``FEE_FLOOR_MIN`` for an empty window). Split children inherit the parent
  floor and window; a merged parent takes the larger child floor and the union of both
  child windows in confirmation order. New leaves start with empty utilization history.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from tensorpow.mempool.mempool import (
    BYTES_PER_KB,
    FEE_FLOOR_EWMA_DEN,
    FEE_FLOOR_EWMA_NEW_WEIGHT,
    FEE_FLOOR_EWMA_PREV_WEIGHT,
    FEE_FLOOR_MIN_MATOMS_PER_KB,
    FEE_FLOOR_WINDOW_FRUITS,
)
from tensorpow.mempool.shard_tree import (
    SHARD_MERGE_WINDOW_ANCHORS,
    SHARD_SPLIT_WINDOW_ANCHORS,
    SHARD_TREE_MAX_BYTES,
    ShardId,
    ShardMerge,
    ShardSplit,
    ShardTree,
    ShardTreeOperation,
    ShardUtilizationSample,
    apply_operations,
    child_shard_ids,
    decode_shard_id,
    parent_shard_id,
    should_merge,
    should_split,
)

U32_BYTES: Final[int] = 4
UTILIZATION_HISTORY_ANCHORS: Final[int] = max(
    SHARD_SPLIT_WINDOW_ANCHORS,
    SHARD_MERGE_WINDOW_ANCHORS,
)
MAX_SHARD_LEAVES: Final[int] = (SHARD_TREE_MAX_BYTES - U32_BYTES) // U32_BYTES


@dataclass(frozen=True, slots=True)
class CoveredFruitStats:
    """Fee and payload contribution of one covered fruit, in canonical covered order."""

    shard_id: ShardId
    payload_bytes: int
    fees_matoms: int


@dataclass(frozen=True, slots=True)
class FeeSample:
    """One confirmed fruit's floor-eligible fees and payload, tagged with its sequence."""

    sequence: int
    fees_matoms: int
    payload_bytes: int


@dataclass(slots=True)
class ShardPolicyState:
    """Branch-local shard/fee history needed to derive the next anchor commitments."""

    tree: ShardTree
    fee_floors: dict[ShardId, int]
    fee_windows: dict[ShardId, deque[FeeSample]]
    utilization: dict[ShardId, deque[ShardUtilizationSample]]
    next_sequence: int = 0

    @classmethod
    def genesis(cls, tree: ShardTree, fee_floors: Mapping[ShardId, int]) -> ShardPolicyState:
        return cls(
            tree=tree,
            fee_floors=dict(fee_floors),
            fee_windows={leaf: _fee_window() for leaf in tree.leaf_shard_ids},
            utilization={leaf: _utilization_window() for leaf in tree.leaf_shard_ids},
        )

    def copy(self) -> ShardPolicyState:
        return ShardPolicyState(
            tree=self.tree,
            fee_floors=dict(self.fee_floors),
            fee_windows={leaf: _fee_window(window) for leaf, window in self.fee_windows.items()},
            utilization={
                leaf: _utilization_window(window) for leaf, window in self.utilization.items()
            },
            next_sequence=self.next_sequence,
        )

    def floor_entries(self) -> tuple[tuple[ShardId, int], ...]:
        return tuple(
            (leaf, self.fee_floors.get(leaf, FEE_FLOOR_MIN_MATOMS_PER_KB))
            for leaf in self.tree.leaf_shard_ids
        )


def next_shard_policy_state(
    state: ShardPolicyState,
    covered: Iterable[CoveredFruitStats],
) -> ShardPolicyState:
    """Return the shard/fee state an anchor covering ``covered`` must commit."""

    working = state.copy()
    leaves = working.tree.leaf_set
    payload_by_leaf = dict.fromkeys(working.tree.leaf_shard_ids, 0)
    fruits_by_leaf = dict.fromkeys(working.tree.leaf_shard_ids, 0)
    for stats in covered:
        if stats.shard_id not in leaves:
            continue
        working.fee_windows.setdefault(stats.shard_id, _fee_window()).append(
            FeeSample(working.next_sequence, stats.fees_matoms, stats.payload_bytes)
        )
        working.next_sequence += 1
        payload_by_leaf[stats.shard_id] += stats.payload_bytes
        fruits_by_leaf[stats.shard_id] += 1
    for leaf in working.tree.leaf_shard_ids:
        working.utilization.setdefault(leaf, _utilization_window()).append(
            ShardUtilizationSample(
                shard_id=leaf,
                payload_bytes_confirmed=payload_by_leaf[leaf],
                fruit_slots_observed=max(1, fruits_by_leaf[leaf]),
            )
        )

    operations = _tree_operations(working)
    next_tree = apply_operations(working.tree, operations).tree

    next_floors: dict[ShardId, int] = {}
    next_windows: dict[ShardId, deque[FeeSample]] = {}
    next_utilization: dict[ShardId, deque[ShardUtilizationSample]] = {}
    for leaf in next_tree.leaf_shard_ids:
        if leaf in leaves:
            previous = working.fee_floors.get(leaf, FEE_FLOOR_MIN_MATOMS_PER_KB)
            window = working.fee_windows.get(leaf, _fee_window())
            history = working.utilization.get(leaf, _utilization_window())
        elif _parent_or_none(leaf) in leaves:
            parent = parent_shard_id(leaf)
            previous = working.fee_floors.get(parent, FEE_FLOOR_MIN_MATOMS_PER_KB)
            window = _fee_window(working.fee_windows.get(parent, ()))
            history = _utilization_window()
        else:
            left, right = child_shard_ids(leaf)
            previous = max(
                working.fee_floors.get(left, FEE_FLOOR_MIN_MATOMS_PER_KB),
                working.fee_floors.get(right, FEE_FLOOR_MIN_MATOMS_PER_KB),
            )
            merged = sorted(
                (*working.fee_windows.get(left, ()), *working.fee_windows.get(right, ())),
                key=lambda sample: sample.sequence,
            )
            window = _fee_window(merged)
            history = _utilization_window()
        next_floors[leaf] = next_fee_floor(previous, window)
        next_windows[leaf] = _fee_window(window)
        next_utilization[leaf] = _utilization_window(history)

    return ShardPolicyState(
        tree=next_tree,
        fee_floors=next_floors,
        fee_windows=next_windows,
        utilization=next_utilization,
        next_sequence=working.next_sequence,
    )


def next_fee_floor(previous_floor: int, window: Iterable[FeeSample]) -> int:
    """Return the EWMA fee floor for one leaf from its previous floor and window."""

    samples = tuple(window)
    if samples:
        total_fees = sum(sample.fees_matoms for sample in samples)
        total_payload = sum(sample.payload_bytes for sample in samples)
        recent_rate = total_fees * BYTES_PER_KB // max(1, total_payload)
    else:
        recent_rate = FEE_FLOOR_MIN_MATOMS_PER_KB
    return max(
        FEE_FLOOR_MIN_MATOMS_PER_KB,
        (FEE_FLOOR_EWMA_PREV_WEIGHT * previous_floor + FEE_FLOOR_EWMA_NEW_WEIGHT * recent_rate)
        // FEE_FLOOR_EWMA_DEN,
    )


def _tree_operations(state: ShardPolicyState) -> tuple[ShardTreeOperation, ...]:
    leaves = state.tree.leaf_set
    splits = [
        leaf
        for leaf in state.tree.leaf_shard_ids
        if should_split(leaf, tuple(state.utilization.get(leaf, ())))
    ]
    merges: list[tuple[ShardId, ShardId]] = []
    for leaf in state.tree.leaf_shard_ids:
        depth, _path = decode_shard_id(leaf)
        if depth == 0:
            continue
        left, right = child_shard_ids(parent_shard_id(leaf))
        if leaf != left or right not in leaves:
            continue
        history = (*state.utilization.get(left, ()), *state.utilization.get(right, ()))
        if should_merge((left, right), history):
            merges.append((left, right))
    leaf_budget = MAX_SHARD_LEAVES - (len(leaves) - len(merges))
    splits = splits[: max(0, leaf_budget)]
    return (
        *(ShardSplit(leaf) for leaf in splits),
        *(ShardMerge(pair) for pair in merges),
    )


def _parent_or_none(leaf: ShardId) -> ShardId | None:
    depth, _path = decode_shard_id(leaf)
    return None if depth == 0 else parent_shard_id(leaf)


def _fee_window(samples: Iterable[FeeSample] = ()) -> deque[FeeSample]:
    return deque(samples, maxlen=FEE_FLOOR_WINDOW_FRUITS)


def _utilization_window(
    samples: Iterable[ShardUtilizationSample] = (),
) -> deque[ShardUtilizationSample]:
    return deque(samples, maxlen=UTILIZATION_HISTORY_ANCHORS)


__all__ = [
    "MAX_SHARD_LEAVES",
    "UTILIZATION_HISTORY_ANCHORS",
    "CoveredFruitStats",
    "FeeSample",
    "ShardPolicyState",
    "next_fee_floor",
    "next_shard_policy_state",
]
