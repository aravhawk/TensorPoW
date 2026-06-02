"""GHOSTDAG ordering helpers for the TensorPoW fruit DAG."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import (
    ROUND_CEILING,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Final

from tensorpow.crypto.hash import HASH_LEN_BYTES

FruitHash = bytes
Numeric = int | Decimal

MS_PER_SECOND: Final[int] = 1000
DYNAMIC_K_FACTOR: Final[int] = 2
DYNAMIC_K_MIN: Final[int] = 15
DYNAMIC_K_MAX: Final[int] = 10000
DYNAMIC_K_DELTA_NUM: Final[int] = 1
DYNAMIC_K_DELTA_DEN: Final[int] = 1000000
DYNAMIC_K_OBSERVATION_ANCHORS: Final[int] = 100
DYNAMIC_K_D_MAX_MIN_MS: Final[int] = 100
DYNAMIC_K_D_MAX_MAX_MS: Final[int] = 5000
_DYNAMIC_K_DECIMAL_CONTEXT: Final[Context] = Context(
    prec=80,
    rounding=ROUND_HALF_EVEN,
    traps=[InvalidOperation, DivisionByZero, Overflow],
)

U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True, slots=True)
class FruitBlock:
    """Minimal fruit DAG record used by consensus ordering helpers."""

    fruit_hash: FruitHash
    parents: tuple[FruitHash, ...]
    timestamp_ms: int
    work: int = 1

    def __post_init__(self) -> None:
        _require_hash("fruit_hash", self.fruit_hash)
        _require_hash_tuple("parents", self.parents)
        _require_u64("timestamp_ms", self.timestamp_ms)
        if not isinstance(self.work, int):
            raise TypeError("work must be int")
        if self.work <= 0:
            raise ValueError("work must be positive")


@dataclass(frozen=True, slots=True)
class GhostdagData:
    """Cached GHOSTDAG metadata for one fruit at one K value."""

    selected_parent: FruitHash | None
    blues: frozenset[FruitHash]
    reds: frozenset[FruitHash]
    blue_score: int
    accumulated_blue_work: int


class BlockDAG:
    """Small typed BlockDAG model with deterministic GHOSTDAG metadata."""

    def __init__(self, blocks: Iterable[FruitBlock] | None = None) -> None:
        self._blocks: dict[FruitHash, FruitBlock] = {}
        self._ancestor_cache: dict[FruitHash, frozenset[FruitHash]] = {}
        self._metadata_cache: dict[int, dict[FruitHash, GhostdagData]] = {}
        for block in blocks or ():
            self.add_block(block)

    def __contains__(self, fruit_hash: object) -> bool:
        return fruit_hash in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def add_fruit(
        self,
        fruit_hash: FruitHash,
        parents: Iterable[FruitHash] = (),
        *,
        timestamp_ms: int,
        work: int = 1,
    ) -> FruitBlock:
        """Add a fruit after validating parent reachability and timestamp order."""

        return self.add_block(
            FruitBlock(
                fruit_hash=fruit_hash,
                parents=tuple(parents),
                timestamp_ms=timestamp_ms,
                work=work,
            )
        )

    def add_block(self, block: FruitBlock) -> FruitBlock:
        """Add a pre-built fruit block to the DAG."""

        if not isinstance(block, FruitBlock):
            raise TypeError("block must be FruitBlock")
        if block.fruit_hash in self._blocks:
            raise ValueError("fruit_hash already exists")
        for parent in block.parents:
            parent_block = self._blocks.get(parent)
            if parent_block is None:
                raise KeyError("parent is unknown")
            if block.timestamp_ms <= parent_block.timestamp_ms:
                raise ValueError("fruit timestamp must be greater than parent timestamps")

        ancestors: set[FruitHash] = set()
        for parent in block.parents:
            ancestors.add(parent)
            ancestors.update(self._ancestor_cache[parent])
        self._blocks[block.fruit_hash] = block
        self._ancestor_cache[block.fruit_hash] = frozenset(ancestors)
        self._metadata_cache.clear()
        return block

    def has_block(self, fruit_hash: FruitHash) -> bool:
        """Return whether a fruit hash is known to the DAG."""

        _require_hash("fruit_hash", fruit_hash)
        return fruit_hash in self._blocks

    def get_block(self, fruit_hash: FruitHash) -> FruitBlock:
        """Return a fruit block by hash."""

        _require_hash("fruit_hash", fruit_hash)
        try:
            return self._blocks[fruit_hash]
        except KeyError as error:
            raise KeyError("fruit_hash is unknown") from error

    def ancestors(self, fruit_hash: FruitHash) -> frozenset[FruitHash]:
        """Return all strict ancestors of a fruit."""

        self.get_block(fruit_hash)
        return self._ancestor_cache[fruit_hash]

    def ghostdag_data(self, fruit_hash: FruitHash, k: int) -> GhostdagData:
        """Return cached GHOSTDAG metadata for a fruit and K."""

        self.get_block(fruit_hash)
        return self.ghostdag_metadata(k)[fruit_hash]

    def ghostdag_metadata(self, k: int) -> dict[FruitHash, GhostdagData]:
        """Return GHOSTDAG metadata for every known fruit at K."""

        _require_k(k)
        cached = self._metadata_cache.get(k)
        if cached is not None and len(cached) == len(self._blocks):
            return dict(cached)

        metadata: dict[FruitHash, GhostdagData] = {}
        for fruit_hash in self._topological_hashes():
            block = self._blocks[fruit_hash]
            if not block.parents:
                metadata[fruit_hash] = GhostdagData(
                    selected_parent=None,
                    blues=frozenset((fruit_hash,)),
                    reds=frozenset(),
                    blue_score=1,
                    accumulated_blue_work=block.work,
                )
                continue

            selected_parent = self._select_parent(block.parents, metadata)
            selected_data = metadata[selected_parent]
            selected_past = set(self.ancestors(selected_parent))
            selected_past.add(selected_parent)

            merge_past: set[FruitHash] = set()
            for parent in block.parents:
                if parent == selected_parent:
                    continue
                merge_past.add(parent)
                merge_past.update(self.ancestors(parent))

            current_blues = set(selected_data.blues)
            for candidate in sorted(
                merge_past - selected_past,
                key=lambda item: (
                    metadata[item].blue_score,
                    self._blocks[item].timestamp_ms,
                    item,
                ),
            ):
                if self._anticone_size(candidate, current_blues) <= k:
                    current_blues.add(candidate)

            blues = frozenset((*current_blues, fruit_hash))
            past = self._past_of_parents(block.parents)
            reds = frozenset(past - set(blues))
            metadata[fruit_hash] = GhostdagData(
                selected_parent=selected_parent,
                blues=blues,
                reds=reds,
                blue_score=len(blues),
                accumulated_blue_work=sum(self._blocks[item].work for item in blues),
            )

        self._metadata_cache[k] = metadata
        return dict(metadata)

    def _select_parent(
        self,
        parents: tuple[FruitHash, ...],
        metadata: dict[FruitHash, GhostdagData],
    ) -> FruitHash:
        return sorted(parents, key=lambda item: (-metadata[item].accumulated_blue_work, item))[0]

    def _past_of_parents(self, parents: tuple[FruitHash, ...]) -> set[FruitHash]:
        past: set[FruitHash] = set()
        for parent in parents:
            past.add(parent)
            past.update(self.ancestors(parent))
        return past

    def _topological_hashes(self) -> tuple[FruitHash, ...]:
        visited: set[FruitHash] = set()
        visiting: set[FruitHash] = set()
        ordered: list[FruitHash] = []

        def visit(fruit_hash: FruitHash) -> None:
            if fruit_hash in visited:
                return
            if fruit_hash in visiting:
                raise ValueError("fruit DAG contains a cycle")
            visiting.add(fruit_hash)
            for parent in self._blocks[fruit_hash].parents:
                visit(parent)
            visiting.remove(fruit_hash)
            visited.add(fruit_hash)
            ordered.append(fruit_hash)

        for fruit_hash in sorted(
            self._blocks,
            key=lambda item: (self._blocks[item].timestamp_ms, item),
        ):
            visit(fruit_hash)
        return tuple(ordered)

    def _anticone_size(self, fruit_hash: FruitHash, candidate_blues: set[FruitHash]) -> int:
        return sum(1 for blue in candidate_blues if not self._is_related(fruit_hash, blue))

    def _is_related(self, left: FruitHash, right: FruitHash) -> bool:
        if left == right:
            return True
        return left in self.ancestors(right) or right in self.ancestors(left)


def compute_k(
    observed_lambda: Numeric,
    observed_d_max_ms: Numeric,
    delta: Numeric | None = None,
    *,
    observation_anchors: int = DYNAMIC_K_OBSERVATION_ANCHORS,
) -> int:
    """Compute reserved dynamic K with exact, window-pinned decimal arithmetic."""

    _require_dynamic_k_observation_anchors(observation_anchors)
    lambda_dec = _require_nonnegative_decimal("observed_lambda", observed_lambda)
    d_max_dec = _require_positive_decimal("observed_d_max_ms", observed_d_max_ms)
    with localcontext(_DYNAMIC_K_DECIMAL_CONTEXT) as context:
        context.clear_flags()
        delta_dec = (
            Decimal(DYNAMIC_K_DELTA_NUM) / Decimal(DYNAMIC_K_DELTA_DEN)
            if delta is None
            else _require_positive_decimal("delta", delta)
        )
        if delta_dec >= 1:
            raise ValueError("delta must be less than one")

        min_delay = Decimal(DYNAMIC_K_D_MAX_MIN_MS)
        max_delay = Decimal(DYNAMIC_K_D_MAX_MAX_MS)
        bounded_delay = min(max(d_max_dec, min_delay), max_delay)
        raw = (
            Decimal(DYNAMIC_K_FACTOR)
            * lambda_dec
            * bounded_delay
            / Decimal(MS_PER_SECOND)
            * (Decimal(1) / delta_dec).ln()
        )
        rounded = int(raw.to_integral_value(rounding=ROUND_CEILING))
    return min(max(rounded, DYNAMIC_K_MIN), DYNAMIC_K_MAX)


def blue_set(dag: BlockDAG, tip: FruitHash, k: int) -> set[FruitHash]:
    """Return the blue set selected for a tip."""

    _require_dag(dag)
    return set(dag.ghostdag_data(tip, k).blues)


def red_set(dag: BlockDAG, tip: FruitHash, k: int) -> set[FruitHash]:
    """Return the red set selected for a tip."""

    _require_dag(dag)
    return set(dag.ghostdag_data(tip, k).reds)


def topological_order(dag: BlockDAG, tip: FruitHash, k: int) -> list[FruitHash]:
    """Return reachable fruits sorted by blue score, timestamp, then hash."""

    _require_dag(dag)
    metadata = dag.ghostdag_metadata(k)
    reachable = set(dag.ancestors(tip))
    reachable.add(tip)
    return sorted(
        reachable,
        key=lambda item: (metadata[item].blue_score, dag.get_block(item).timestamp_ms, item),
    )


def _require_dag(dag: BlockDAG) -> None:
    if not isinstance(dag, BlockDAG):
        raise TypeError("dag must be BlockDAG")


def _require_hash(name: str, value: FruitHash) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _require_hash_tuple(name: str, value: tuple[FruitHash, ...]) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    for item in value:
        _require_hash(name, item)


def _require_u64(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= U64_MAX:
        raise ValueError(f"{name} outside uint64 range")


def _require_k(k: int) -> None:
    if not isinstance(k, int):
        raise TypeError("k must be int")
    if k < 0:
        raise ValueError("k must be non-negative")


def _require_dynamic_k_observation_anchors(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("observation_anchors must be int")
    if value != DYNAMIC_K_OBSERVATION_ANCHORS:
        raise ValueError("observation_anchors must equal DYNAMIC_K_OBSERVATION_ANCHORS")


def _require_nonnegative_decimal(name: str, value: Numeric) -> Decimal:
    number = _decimal_from_numeric(name, value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _require_positive_decimal(name: str, value: Numeric) -> Decimal:
    number = _decimal_from_numeric(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _decimal_from_numeric(name: str, value: Numeric) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        raise TypeError(f"{name} must be int or Decimal")
    try:
        number = value if isinstance(value, Decimal) else Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be finite") from error
    if not number.is_finite():
        raise ValueError(f"{name} must be finite")
    return number
