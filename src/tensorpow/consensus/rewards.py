"""Deterministic issuance and coinbase reward accounting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from tensorpow.consensus.anchor_daa import anchor_work_weight
from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.state.utxo import MAX_SUPPLY_MATOMS

FRUIT_REWARD_WEIGHT: Final[int] = 1
MATOMS_PER_TSC: Final[int] = 100_000_000
MAX_SUPPLY_TSC: Final[int] = 21_000_000
HALVING_YEARS: Final[int] = 4
DAYS_PER_YEAR: Final[int] = 365
HOURS_PER_DAY: Final[int] = 24
MINUTES_PER_HOUR: Final[int] = 60
HALVING_INTERVAL_ANCHORS: Final[int] = (
    HALVING_YEARS * DAYS_PER_YEAR * HOURS_PER_DAY * MINUTES_PER_HOUR
)
INITIAL_ISSUANCE_TSC_PER_YEAR: Final[int] = 2_625_000
INITIAL_EPOCH_SUBSIDY_TSC: Final[int] = 10_500_000
INITIAL_EPOCH_SUBSIDY_MATOMS: Final[int] = INITIAL_EPOCH_SUBSIDY_TSC * MATOMS_PER_TSC
COINBASE_MATURITY_ANCHORS: Final[int] = 100
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF


def interval_subsidy_matoms(anchor_height: int, minted_supply_matoms: int = 0) -> int:
    """Return the new issuance available to a non-genesis anchor interval."""

    _require_positive_int("anchor_height", anchor_height)
    _require_nonnegative_int("minted_supply_matoms", minted_supply_matoms)
    if minted_supply_matoms > MAX_SUPPLY_MATOMS:
        raise ValueError("minted_supply_matoms exceeds MAX_SUPPLY_MATOMS")

    interval_index = anchor_height - 1
    epoch_index = interval_index // HALVING_INTERVAL_ANCHORS
    anchor_index_in_epoch = interval_index % HALVING_INTERVAL_ANCHORS
    epoch_pool = INITIAL_EPOCH_SUBSIDY_MATOMS >> epoch_index
    if epoch_pool == 0:
        return 0

    base = epoch_pool // HALVING_INTERVAL_ANCHORS
    remainder = epoch_pool % HALVING_INTERVAL_ANCHORS
    subsidy = base + (1 if anchor_index_in_epoch < remainder else 0)
    remaining_supply = MAX_SUPPLY_MATOMS - minted_supply_matoms
    return min(subsidy, remaining_supply)


def fruit_subsidy_assignments(
    fruit_reward_keys: Mapping[bytes, bytes],
    *,
    interval_subsidy: int,
    anchor_target: bytes,
) -> dict[bytes, int]:
    """Split the fruit side of an interval subsidy over covered fruit hashes."""

    if not isinstance(fruit_reward_keys, Mapping):
        raise TypeError("fruit_reward_keys must be a mapping")
    _require_nonnegative_int("interval_subsidy", interval_subsidy)
    _require_hash("anchor_target", anchor_target)
    if not fruit_reward_keys:
        return {}

    checked: dict[bytes, bytes] = {}
    for fruit_hash, reward_key in fruit_reward_keys.items():
        _require_hash("fruit_hash", fruit_hash)
        _require_hash("reward_key", reward_key)
        checked[fruit_hash] = reward_key

    fruit_pool, _anchor_pool = reward_pools(
        fruit_count=len(checked),
        interval_subsidy=interval_subsidy,
        anchor_target=anchor_target,
    )
    base, remainder = divmod(fruit_pool, len(checked))
    ordered_hashes = sorted(checked, key=lambda fruit_hash: (checked[fruit_hash], fruit_hash))
    return {
        fruit_hash: base + (1 if index < remainder else 0)
        for index, fruit_hash in enumerate(ordered_hashes)
    }


def reward_pools(
    *,
    fruit_count: int,
    interval_subsidy: int,
    anchor_target: bytes,
) -> tuple[int, int]:
    """Return ``(fruit_pool, anchor_pool)`` for one anchor interval."""

    _require_nonnegative_int("fruit_count", fruit_count)
    _require_nonnegative_int("interval_subsidy", interval_subsidy)
    _require_hash("anchor_target", anchor_target)
    if fruit_count == 0 or interval_subsidy == 0:
        return 0, interval_subsidy
    fruit_weight = fruit_count * FRUIT_REWARD_WEIGHT
    total_weight = fruit_weight + anchor_work_weight(anchor_target)
    fruit_pool = interval_subsidy * fruit_weight // total_weight
    return fruit_pool, interval_subsidy - fruit_pool


def coinbase_maturity_height(anchor_height: int) -> int:
    """Return the first anchor height at which coinbase outputs are spendable."""

    _require_positive_int("anchor_height", anchor_height)
    return anchor_height + COINBASE_MATURITY_ANCHORS


def _require_hash(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _require_positive_int(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if value > U64_MAX:
        raise ValueError(f"{name} outside uint64 range")
    return value


__all__ = [
    "COINBASE_MATURITY_ANCHORS",
    "DAYS_PER_YEAR",
    "FRUIT_REWARD_WEIGHT",
    "HALVING_INTERVAL_ANCHORS",
    "HALVING_YEARS",
    "HOURS_PER_DAY",
    "INITIAL_EPOCH_SUBSIDY_MATOMS",
    "INITIAL_EPOCH_SUBSIDY_TSC",
    "INITIAL_ISSUANCE_TSC_PER_YEAR",
    "MATOMS_PER_TSC",
    "MAX_SUPPLY_TSC",
    "MINUTES_PER_HOUR",
    "coinbase_maturity_height",
    "fruit_subsidy_assignments",
    "interval_subsidy_matoms",
    "reward_pools",
]
