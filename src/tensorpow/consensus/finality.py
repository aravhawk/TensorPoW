"""Finality tier helpers for TensorPoW fruits."""

from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Final

from tensorpow.consensus.anchor_daa import ANCHOR_INTERVAL_MS
from tensorpow.consensus.ghostdag import BlockDAG, FruitHash

FINALITY_FAST_BLUE_DEPTH: Final[int] = 5
FINALITY_ECONOMIC_BLUE_DEPTH: Final[int] = 20
FINALITY_SETTLEMENT_BLUE_DEPTH: Final[int] = 100
FINALITY_ANCHOR_SECURED_DEPTH: Final[int] = 1
FINALITY_SETTLEMENT_ANCHOR_DEPTH: Final[int] = 6


class FinalityTier(Enum):
    """Named finality states from the protocol spec."""

    NONE = "None"
    SEEN = "Seen"
    FAST = "Fast"
    ECONOMIC = "Economic"
    ANCHOR_SECURED = "AnchorSecured"
    SETTLEMENT = "Settlement"


def blue_depth(dag: BlockDAG, fruit_hash: FruitHash, tip: FruitHash, k: int) -> int:
    """Return blue depth for a fruit from a selected tip."""

    if not isinstance(dag, BlockDAG):
        raise TypeError("dag must be BlockDAG")
    if not dag.has_block(fruit_hash):
        raise KeyError("fruit_hash is unknown")

    reachable = set(dag.ancestors(tip))
    reachable.add(tip)
    if fruit_hash not in reachable:
        return 0

    metadata = dag.ghostdag_metadata(k)
    tip_data = metadata[tip]
    if fruit_hash not in tip_data.blues:
        return 0
    return max(0, tip_data.blue_score - metadata[fruit_hash].blue_score)


def anchor_depth(covering_anchor_height: int | None, tip_anchor_height: int) -> int:
    """Return inclusive depth from the anchor that first covered a fruit."""

    _require_optional_height("covering_anchor_height", covering_anchor_height)
    _require_height("tip_anchor_height", tip_anchor_height)
    if covering_anchor_height is None:
        return 0
    if tip_anchor_height < covering_anchor_height:
        raise ValueError("tip_anchor_height must be at least covering_anchor_height")
    return tip_anchor_height - covering_anchor_height + 1


def finality_tier(
    dag: BlockDAG,
    fruit_hash: FruitHash,
    tip: FruitHash,
    k: int,
    *,
    anchor_depth_value: int = 0,
) -> FinalityTier:
    """Return the strongest satisfied finality tier for a fruit."""

    if not isinstance(dag, BlockDAG):
        raise TypeError("dag must be BlockDAG")
    if not dag.has_block(fruit_hash):
        return FinalityTier.NONE
    return finality_tier_from_depths(
        blue_depth(dag, fruit_hash, tip, k),
        anchor_depth_value,
        seen=True,
    )


def finality_tier_from_depths(
    blue_depth_value: int,
    anchor_depth_value: int,
    *,
    seen: bool,
) -> FinalityTier:
    """Return the strongest finality tier from precomputed depths."""

    _require_depth("blue_depth_value", blue_depth_value)
    _require_depth("anchor_depth_value", anchor_depth_value)
    if not isinstance(seen, bool):
        raise TypeError("seen must be bool")
    if not seen:
        return FinalityTier.NONE
    if (
        blue_depth_value >= FINALITY_SETTLEMENT_BLUE_DEPTH
        and anchor_depth_value >= FINALITY_SETTLEMENT_ANCHOR_DEPTH
    ):
        return FinalityTier.SETTLEMENT
    if anchor_depth_value >= FINALITY_ANCHOR_SECURED_DEPTH:
        return FinalityTier.ANCHOR_SECURED
    if blue_depth_value >= FINALITY_ECONOMIC_BLUE_DEPTH:
        return FinalityTier.ECONOMIC
    if blue_depth_value >= FINALITY_FAST_BLUE_DEPTH:
        return FinalityTier.FAST
    return FinalityTier.SEEN


def satisfied_finality_tiers(
    blue_depth_value: int,
    anchor_depth_value: int,
    *,
    seen: bool,
) -> frozenset[FinalityTier]:
    """Return all finality tiers satisfied by the supplied depths."""

    _require_depth("blue_depth_value", blue_depth_value)
    _require_depth("anchor_depth_value", anchor_depth_value)
    if not isinstance(seen, bool):
        raise TypeError("seen must be bool")
    if not seen:
        return frozenset((FinalityTier.NONE,))

    tiers = {FinalityTier.SEEN}
    if blue_depth_value >= FINALITY_FAST_BLUE_DEPTH:
        tiers.add(FinalityTier.FAST)
    if blue_depth_value >= FINALITY_ECONOMIC_BLUE_DEPTH:
        tiers.add(FinalityTier.ECONOMIC)
    if anchor_depth_value >= FINALITY_ANCHOR_SECURED_DEPTH:
        tiers.add(FinalityTier.ANCHOR_SECURED)
    if (
        blue_depth_value >= FINALITY_SETTLEMENT_BLUE_DEPTH
        and anchor_depth_value >= FINALITY_SETTLEMENT_ANCHOR_DEPTH
    ):
        tiers.add(FinalityTier.SETTLEMENT)
    return frozenset(tiers)


def estimate_finality_seconds(tier: FinalityTier, observed_block_rate: float) -> float:
    """Estimate advisory wall-clock time for a finality tier."""

    if not isinstance(tier, FinalityTier):
        raise TypeError("tier must be FinalityTier")
    if not isinstance(observed_block_rate, int | float) or isinstance(observed_block_rate, bool):
        raise TypeError("observed_block_rate must be numeric")
    if not isfinite(float(observed_block_rate)) or observed_block_rate <= 0:
        raise ValueError("observed_block_rate must be positive and finite")

    if tier in (FinalityTier.NONE, FinalityTier.SEEN):
        return 0.0
    if tier is FinalityTier.FAST:
        return FINALITY_FAST_BLUE_DEPTH / observed_block_rate
    if tier is FinalityTier.ECONOMIC:
        return FINALITY_ECONOMIC_BLUE_DEPTH / observed_block_rate
    if tier is FinalityTier.ANCHOR_SECURED:
        return FINALITY_ANCHOR_SECURED_DEPTH * ANCHOR_INTERVAL_MS / 1000
    return max(
        FINALITY_SETTLEMENT_BLUE_DEPTH / observed_block_rate,
        FINALITY_SETTLEMENT_ANCHOR_DEPTH * ANCHOR_INTERVAL_MS / 1000,
    )


def _require_depth(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_height(name: str, value: int) -> None:
    _require_depth(name, value)


def _require_optional_height(name: str, value: int | None) -> None:
    if value is None:
        return
    _require_height(name, value)
