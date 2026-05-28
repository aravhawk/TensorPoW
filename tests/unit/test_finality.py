"""Tests for finality tier helpers."""

from __future__ import annotations

import pytest

from tensorpow.consensus.finality import (
    FinalityTier,
    anchor_depth,
    blue_depth,
    estimate_finality_seconds,
    finality_tier,
    finality_tier_from_depths,
    satisfied_finality_tiers,
)
from tensorpow.consensus.ghostdag import BlockDAG


def _h(index: int) -> bytes:
    return index.to_bytes(32, "big")


def _linear_dag(length: int) -> tuple[BlockDAG, bytes, bytes]:
    dag = BlockDAG()
    genesis = _h(1)
    dag.add_fruit(genesis, timestamp_ms=1)
    tip = genesis
    for index in range(2, length + 1):
        dag.add_fruit(_h(index), (tip,), timestamp_ms=index)
        tip = _h(index)
    return dag, genesis, tip


def test_finality_tier_from_depths_matches_protocol_thresholds() -> None:
    assert finality_tier_from_depths(0, 0, seen=False) is FinalityTier.NONE
    assert finality_tier_from_depths(0, 0, seen=True) is FinalityTier.SEEN
    assert finality_tier_from_depths(5, 0, seen=True) is FinalityTier.FAST
    assert finality_tier_from_depths(20, 0, seen=True) is FinalityTier.ECONOMIC
    assert finality_tier_from_depths(0, 1, seen=True) is FinalityTier.ANCHOR_SECURED
    assert finality_tier_from_depths(100, 6, seen=True) is FinalityTier.SETTLEMENT


def test_satisfied_finality_tiers_returns_all_matching_tiers() -> None:
    assert satisfied_finality_tiers(100, 6, seen=True) == {
        FinalityTier.SEEN,
        FinalityTier.FAST,
        FinalityTier.ECONOMIC,
        FinalityTier.ANCHOR_SECURED,
        FinalityTier.SETTLEMENT,
    }
    assert satisfied_finality_tiers(0, 0, seen=False) == {FinalityTier.NONE}


def test_blue_depth_and_finality_over_linear_chain() -> None:
    dag, genesis, tip = _linear_dag(21)

    assert blue_depth(dag, genesis, tip, 15) == 20
    assert finality_tier(dag, genesis, tip, 15) is FinalityTier.ECONOMIC
    assert finality_tier(dag, _h(18), tip, 15) is FinalityTier.SEEN
    assert finality_tier(dag, _h(18), tip, 15, anchor_depth_value=1) is FinalityTier.ANCHOR_SECURED


def test_anchor_depth_is_inclusive_and_validates_heights() -> None:
    assert anchor_depth(None, 10) == 0
    assert anchor_depth(7, 7) == 1
    assert anchor_depth(7, 12) == 6

    with pytest.raises(ValueError, match="at least"):
        anchor_depth(7, 6)
    with pytest.raises(ValueError, match="non-negative"):
        anchor_depth(-1, 6)
    with pytest.raises(TypeError):
        anchor_depth(1.5, 6)  # type: ignore[arg-type]


def test_red_or_unreachable_fruits_do_not_gain_blue_depth() -> None:
    dag = BlockDAG()
    dag.add_fruit(_h(1), timestamp_ms=1)
    dag.add_fruit(_h(2), (_h(1),), timestamp_ms=2)
    dag.add_fruit(_h(100), (_h(1),), timestamp_ms=100)
    dag.add_fruit(_h(200), (_h(2), _h(100)), timestamp_ms=200)
    dag.add_fruit(_h(300), timestamp_ms=300)

    assert blue_depth(dag, _h(100), _h(200), 0) == 0
    assert finality_tier(dag, _h(100), _h(200), 0) is FinalityTier.SEEN
    assert blue_depth(dag, _h(300), _h(200), 0) == 0
    assert finality_tier(dag, _h(999), _h(200), 0) is FinalityTier.NONE


def test_estimate_finality_seconds_and_malformed_inputs() -> None:
    assert estimate_finality_seconds(FinalityTier.SEEN, 35.0) == 0
    assert estimate_finality_seconds(FinalityTier.FAST, 5.0) == 1
    assert estimate_finality_seconds(FinalityTier.ECONOMIC, 10.0) == 2
    assert estimate_finality_seconds(FinalityTier.ANCHOR_SECURED, 35.0) == 60
    assert estimate_finality_seconds(FinalityTier.SETTLEMENT, 1.0) == 360

    with pytest.raises(ValueError, match="positive"):
        estimate_finality_seconds(FinalityTier.FAST, 0)
    with pytest.raises(TypeError):
        estimate_finality_seconds("Fast", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        finality_tier_from_depths(-1, 0, seen=True)
    with pytest.raises(TypeError):
        finality_tier_from_depths(0, 0, seen=1)  # type: ignore[arg-type]
