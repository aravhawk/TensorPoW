"""Tests for deterministic issuance and coinbase reward accounting."""

from __future__ import annotations

import pytest

from tensorpow.consensus.anchor_daa import ANCHOR_INITIAL_TARGET_LE, anchor_work_weight
from tensorpow.consensus.rewards import (
    COINBASE_MATURITY_ANCHORS,
    HALVING_INTERVAL_ANCHORS,
    INITIAL_EPOCH_SUBSIDY_MATOMS,
    coinbase_maturity_height,
    fruit_subsidy_assignments,
    interval_subsidy_matoms,
    reward_pools,
)
from tensorpow.state.utxo import MAX_SUPPLY_MATOMS


def test_interval_subsidy_halves_and_caps_at_remaining_supply() -> None:
    first = interval_subsidy_matoms(1)
    second_epoch = interval_subsidy_matoms(HALVING_INTERVAL_ANCHORS + 1)

    assert first == INITIAL_EPOCH_SUBSIDY_MATOMS // HALVING_INTERVAL_ANCHORS + 1
    assert second_epoch < first
    assert interval_subsidy_matoms(1, MAX_SUPPLY_MATOMS - 7) == 7
    assert interval_subsidy_matoms(1, MAX_SUPPLY_MATOMS) == 0


def test_fruit_subsidy_assignments_are_deterministic_and_bounded() -> None:
    fruit_a = bytes.fromhex("aa" * 32)
    fruit_b = bytes.fromhex("bb" * 32)
    assignments = fruit_subsidy_assignments(
        {
            fruit_b: bytes.fromhex("01" * 32),
            fruit_a: bytes.fromhex("00" * 32),
        },
        interval_subsidy=10_000,
        anchor_target=ANCHOR_INITIAL_TARGET_LE,
    )

    assert set(assignments) == {fruit_a, fruit_b}
    assert sum(assignments.values()) < 10_000
    assert assignments[fruit_a] >= assignments[fruit_b]


def test_reward_pools_expose_anchor_claim_path() -> None:
    fruit_pool, anchor_pool = reward_pools(
        fruit_count=2,
        interval_subsidy=10_000,
        anchor_target=ANCHOR_INITIAL_TARGET_LE,
    )

    assert fruit_pool > 0
    assert anchor_pool > 0
    assert fruit_pool + anchor_pool == 10_000
    assert reward_pools(
        fruit_count=0,
        interval_subsidy=10_000,
        anchor_target=ANCHOR_INITIAL_TARGET_LE,
    ) == (0, 10_000)


def test_reward_pools_route_top_level_remainder_to_anchor_pool() -> None:
    fruit_a = bytes.fromhex("aa" * 32)
    fruit_b = bytes.fromhex("bb" * 32)
    anchor_weight = anchor_work_weight(ANCHOR_INITIAL_TARGET_LE)
    interval_subsidy = anchor_weight + 3

    fruit_pool, anchor_pool = reward_pools(
        fruit_count=2,
        interval_subsidy=interval_subsidy,
        anchor_target=ANCHOR_INITIAL_TARGET_LE,
    )
    assignments = fruit_subsidy_assignments(
        {
            fruit_b: bytes.fromhex("01" * 32),
            fruit_a: bytes.fromhex("00" * 32),
        },
        interval_subsidy=interval_subsidy,
        anchor_target=ANCHOR_INITIAL_TARGET_LE,
    )

    assert (fruit_pool, anchor_pool) == (2, anchor_weight + 1)
    assert assignments == {fruit_a: 1, fruit_b: 1}


def test_coinbase_maturity_height_and_malformed_inputs() -> None:
    assert coinbase_maturity_height(1) == 1 + COINBASE_MATURITY_ANCHORS
    with pytest.raises(ValueError):
        interval_subsidy_matoms(0)
    with pytest.raises(ValueError):
        interval_subsidy_matoms(1, MAX_SUPPLY_MATOMS + 1)
    with pytest.raises(ValueError):
        fruit_subsidy_assignments(
            {bytes(31): bytes(32)},
            interval_subsidy=1,
            anchor_target=ANCHOR_INITIAL_TARGET_LE,
        )
