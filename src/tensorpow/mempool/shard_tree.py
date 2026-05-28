"""Canonical hierarchical shard tree state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tensorpow.chain.merkle import require_hash
from tensorpow.crypto.hash import DOMAIN_SHARD_TREE, domain_hash

ShardId = int

U32_BYTES: Final[int] = 4
U32_MAX: Final[int] = 0xFFFFFFFF
SHARD_ID_DEPTH_SHIFT: Final[int] = 16
SHARD_MAX_DEPTH: Final[int] = 16
SHARD_ID_PATH_MASK: Final[int] = (1 << SHARD_ID_DEPTH_SHIFT) - 1
ROOT_SHARD_ID: Final[ShardId] = 0

MAX_FRUIT_PAYLOAD_BYTES: Final[int] = 8192
SHARD_SPLIT_THRESHOLD_PCT: Final[int] = 80
SHARD_SPLIT_WINDOW_ANCHORS: Final[int] = 10
SHARD_MERGE_THRESHOLD_PCT: Final[int] = 20
SHARD_MERGE_WINDOW_ANCHORS: Final[int] = 1440
SHARD_TREE_MAX_BYTES: Final[int] = 262_144


class ShardTreeDecodeError(ValueError):
    """Raised when serialized shard tree bytes are malformed or non-canonical."""


@dataclass(frozen=True, slots=True)
class ShardUtilizationSample:
    """Per-shard utilization aggregate for one anchor interval."""

    shard_id: ShardId
    payload_bytes_confirmed: int
    fruit_slots_observed: int = 1

    def __post_init__(self) -> None:
        require_shard_id(self.shard_id)
        _require_uint("payload_bytes_confirmed", self.payload_bytes_confirmed)
        _require_uint("fruit_slots_observed", self.fruit_slots_observed)
        if self.fruit_slots_observed == 0:
            raise ValueError("fruit_slots_observed must be positive")
        max_payload = MAX_FRUIT_PAYLOAD_BYTES * self.fruit_slots_observed
        if self.payload_bytes_confirmed > max_payload:
            raise ValueError("payload_bytes_confirmed exceeds observed fruit capacity")


@dataclass(frozen=True, slots=True)
class ShardTree:
    """Canonical binary shard tree represented by its complete leaf set."""

    leaf_shard_ids: tuple[ShardId, ...] = (ROOT_SHARD_ID,)

    def __post_init__(self) -> None:
        if not isinstance(self.leaf_shard_ids, tuple):
            raise TypeError("leaf_shard_ids must be a tuple")
        _validate_leaf_partition(self.leaf_shard_ids)

    @property
    def leaf_set(self) -> frozenset[ShardId]:
        return frozenset(self.leaf_shard_ids)

    def route_tx(self, tx_hash: bytes) -> ShardId:
        """Return the unique leaf shard selected by the low bits of a tx hash."""

        require_hash("tx_hash", tx_hash)
        route_int = int.from_bytes(tx_hash, "little")
        for shard_id in self.leaf_shard_ids:
            depth, path = decode_shard_id(shard_id)
            if path == route_int & _path_mask(depth):
                return shard_id
        raise AssertionError("validated shard tree has no route for tx_hash")

    def serialize(self) -> bytes:
        payload = bytearray()
        payload.extend(_u32(len(self.leaf_shard_ids)))
        for shard_id in self.leaf_shard_ids:
            payload.extend(_u32(shard_id))
        return bytes(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> ShardTree:
        reader = _ShardTreeReader(data)
        leaf_count = reader.u32()
        leaf_shard_ids = tuple(reader.u32() for _ in range(leaf_count))
        reader.finish()
        try:
            return cls(leaf_shard_ids=leaf_shard_ids)
        except (TypeError, ValueError) as exc:
            raise ShardTreeDecodeError(str(exc)) from exc

    def state_root(self) -> bytes:
        """Return BLAKE3(DOMAIN_SHARD_TREE || canonical_shard_tree_bytes)."""

        return shard_tree_root(self.serialize())

    def commitment_root(self) -> bytes:
        """Return the canonical shard tree commitment root."""

        return self.state_root()


@dataclass(frozen=True, slots=True)
class ShardSplit:
    """Shard tree operation that splits one leaf into two children."""

    shard_id: ShardId

    def __post_init__(self) -> None:
        require_shard_id(self.shard_id)


@dataclass(frozen=True, slots=True)
class ShardMerge:
    """Shard tree operation that merges sibling leaves into their parent."""

    siblings: tuple[ShardId, ShardId]

    def __post_init__(self) -> None:
        object.__setattr__(self, "siblings", require_sibling_shards(self.siblings))


ShardTreeOperation = ShardSplit | ShardMerge


@dataclass(frozen=True, slots=True)
class ShardTreeUpdate:
    """Result of applying a same-anchor batch of shard tree operations."""

    tree: ShardTree
    applied: tuple[ShardTreeOperation, ...]
    queued: tuple[ShardTreeOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tree, ShardTree):
            raise TypeError("tree must be ShardTree")
        _require_operation_tuple("applied", self.applied)
        _require_operation_tuple("queued", self.queued)


def encode_shard_id(depth: int, path: int) -> ShardId:
    """Encode a shard depth and low-bit path into the canonical uint32 id."""

    _require_uint("depth", depth)
    _require_uint("path", path)
    if depth > SHARD_MAX_DEPTH:
        raise ValueError("shard_id depth exceeds SHARD_MAX_DEPTH")
    if path >= (1 << depth):
        raise ValueError("shard_id path is outside depth")
    return (depth << SHARD_ID_DEPTH_SHIFT) | path


def decode_shard_id(shard_id: ShardId) -> tuple[int, int]:
    """Return ``(depth, path)`` for a canonical shard id."""

    require_shard_id(shard_id)
    return shard_id >> SHARD_ID_DEPTH_SHIFT, shard_id & SHARD_ID_PATH_MASK


def require_shard_id(shard_id: ShardId) -> ShardId:
    """Validate and return a canonical shard id."""

    _require_uint("shard_id", shard_id)
    if shard_id > U32_MAX:
        raise ValueError("shard_id outside uint range")
    depth = shard_id >> SHARD_ID_DEPTH_SHIFT
    path = shard_id & SHARD_ID_PATH_MASK
    if depth > SHARD_MAX_DEPTH:
        raise ValueError("shard_id depth exceeds SHARD_MAX_DEPTH")
    if path >= (1 << depth):
        raise ValueError("shard_id path is outside depth")
    return shard_id


def parent_shard_id(shard_id: ShardId) -> ShardId:
    depth, path = decode_shard_id(shard_id)
    if depth == 0:
        raise ValueError("root shard has no parent")
    return encode_shard_id(depth - 1, path & _path_mask(depth - 1))


def child_shard_ids(shard_id: ShardId) -> tuple[ShardId, ShardId]:
    depth, path = decode_shard_id(shard_id)
    if depth >= SHARD_MAX_DEPTH:
        raise ValueError("split would exceed SHARD_MAX_DEPTH")
    child_depth = depth + 1
    return (
        encode_shard_id(child_depth, path),
        encode_shard_id(child_depth, path | (1 << depth)),
    )


def require_sibling_shards(siblings: tuple[ShardId, ShardId]) -> tuple[ShardId, ShardId]:
    if not isinstance(siblings, tuple):
        raise TypeError("siblings must be a tuple")
    if len(siblings) != 2:
        raise ValueError("siblings must contain exactly two shard ids")
    left, right = siblings
    left_depth, left_path = decode_shard_id(left)
    right_depth, right_path = decode_shard_id(right)
    if left == right:
        raise ValueError("siblings must be distinct")
    if left_depth != right_depth:
        raise ValueError("siblings must have the same depth")
    if left_depth == 0:
        raise ValueError("root shard cannot be merged")
    parent_mask = _path_mask(left_depth - 1)
    if (left_path & parent_mask) != (right_path & parent_mask):
        raise ValueError("siblings must share a parent")
    if {left_path >> (left_depth - 1), right_path >> (right_depth - 1)} != {0, 1}:
        raise ValueError("siblings must be opposite children")
    sorted_siblings = sorted((left, right))
    return sorted_siblings[0], sorted_siblings[1]


def should_split(shard_id: ShardId, history: Sequence[ShardUtilizationSample]) -> bool:
    """Return whether a leaf has met the sustained 80% split threshold."""

    depth, _path = decode_shard_id(shard_id)
    if depth >= SHARD_MAX_DEPTH:
        return False
    samples = _recent_samples(shard_id, history, SHARD_SPLIT_WINDOW_ANCHORS)
    if len(samples) < SHARD_SPLIT_WINDOW_ANCHORS:
        return False
    return all(
        _sample_meets_threshold(sample, SHARD_SPLIT_THRESHOLD_PCT, at_least=True)
        for sample in samples
    )


def should_merge(
    siblings: tuple[ShardId, ShardId],
    history: Sequence[ShardUtilizationSample],
) -> bool:
    """Return whether sibling leaves have met the sustained 20% merge threshold."""

    left, right = require_sibling_shards(siblings)
    left_samples = _recent_samples(left, history, SHARD_MERGE_WINDOW_ANCHORS)
    right_samples = _recent_samples(right, history, SHARD_MERGE_WINDOW_ANCHORS)
    if (
        len(left_samples) < SHARD_MERGE_WINDOW_ANCHORS
        or len(right_samples) < SHARD_MERGE_WINDOW_ANCHORS
    ):
        return False
    return all(
        _sample_meets_threshold(sample, SHARD_MERGE_THRESHOLD_PCT, at_least=False)
        for sample in (*left_samples, *right_samples)
    )


def apply_split(tree: ShardTree, shard_id: ShardId) -> ShardTree:
    """Return a new tree with ``shard_id`` replaced by its two child leaves."""

    _require_tree(tree)
    require_shard_id(shard_id)
    if shard_id not in tree.leaf_set:
        raise ValueError("shard_id is not a leaf")
    children = child_shard_ids(shard_id)
    leaves = tuple(sorted((tree.leaf_set - {shard_id}) | set(children)))
    return ShardTree(leaves)


def apply_merge(tree: ShardTree, siblings: tuple[ShardId, ShardId]) -> ShardTree:
    """Return a new tree with sibling leaves replaced by their parent."""

    _require_tree(tree)
    left, right = require_sibling_shards(siblings)
    if left not in tree.leaf_set or right not in tree.leaf_set:
        raise ValueError("siblings must both be leaves")
    parent = parent_shard_id(left)
    leaves = tuple(sorted((tree.leaf_set - {left, right}) | {parent}))
    return ShardTree(leaves)


def apply_operations(
    tree: ShardTree,
    operations: Sequence[ShardTreeOperation],
) -> ShardTreeUpdate:
    """Apply independent same-anchor operations and queue later same-parent conflicts."""

    _require_tree(tree)
    if not isinstance(operations, Sequence):
        raise TypeError("operations must be a sequence")

    next_tree = tree
    used_parents: set[ShardId] = set()
    applied: list[ShardTreeOperation] = []
    queued: list[ShardTreeOperation] = []
    for operation in operations:
        parent = _operation_parent(operation)
        if parent in used_parents:
            queued.append(operation)
            continue
        used_parents.add(parent)
        next_tree = _apply_operation(next_tree, operation)
        applied.append(operation)

    return ShardTreeUpdate(tree=next_tree, applied=tuple(applied), queued=tuple(queued))


def shard_tree_root(canonical_shard_tree_bytes: bytes) -> bytes:
    if not isinstance(canonical_shard_tree_bytes, bytes):
        raise TypeError("canonical_shard_tree_bytes must be bytes")
    if len(canonical_shard_tree_bytes) > SHARD_TREE_MAX_BYTES:
        raise ValueError("canonical_shard_tree_bytes exceeds SHARD_TREE_MAX_BYTES")
    return domain_hash(DOMAIN_SHARD_TREE, canonical_shard_tree_bytes)


class _ShardTreeReader:
    def __init__(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) > SHARD_TREE_MAX_BYTES:
            raise ShardTreeDecodeError("shard tree exceeds SHARD_TREE_MAX_BYTES")
        self._data = data
        self._offset = 0

    def u32(self) -> int:
        end = self._offset + U32_BYTES
        if end > len(self._data):
            raise ShardTreeDecodeError("truncated shard tree")
        value = int.from_bytes(self._data[self._offset : end], "little")
        self._offset = end
        return value

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise ShardTreeDecodeError("trailing shard tree bytes")


def _validate_leaf_partition(leaf_shard_ids: tuple[ShardId, ...]) -> None:
    if not leaf_shard_ids:
        raise ValueError("leaf_shard_ids must be nonzero")
    if tuple(sorted(leaf_shard_ids)) != leaf_shard_ids:
        raise ValueError("leaf_shard_ids must be strictly ascending")
    if len(set(leaf_shard_ids)) != len(leaf_shard_ids):
        raise ValueError("duplicate leaf shard_id")

    seen: set[ShardId] = set()
    coverage = 0
    for shard_id in leaf_shard_ids:
        depth, path = decode_shard_id(shard_id)
        for ancestor_depth in range(depth):
            ancestor = encode_shard_id(ancestor_depth, path & _path_mask(ancestor_depth))
            if ancestor in seen:
                raise ValueError("leaf shard_ids overlap")
        seen.add(shard_id)
        coverage += 1 << (SHARD_MAX_DEPTH - depth)
    if coverage != 1 << SHARD_MAX_DEPTH:
        raise ValueError("leaf shard_ids leave a gap in the root partition")


def _recent_samples(
    shard_id: ShardId,
    history: Sequence[ShardUtilizationSample],
    window: int,
) -> tuple[ShardUtilizationSample, ...]:
    if not isinstance(history, Sequence):
        raise TypeError("history must be a sequence")
    samples = []
    for sample in history:
        if not isinstance(sample, ShardUtilizationSample):
            raise TypeError("history entries must be ShardUtilizationSample")
        if sample.shard_id == shard_id:
            samples.append(sample)
    return tuple(samples[-window:])


def _sample_meets_threshold(
    sample: ShardUtilizationSample,
    threshold_pct: int,
    *,
    at_least: bool,
) -> bool:
    numerator = sample.payload_bytes_confirmed * 100
    denominator = MAX_FRUIT_PAYLOAD_BYTES * sample.fruit_slots_observed
    threshold = denominator * threshold_pct
    if at_least:
        return numerator >= threshold
    return numerator <= threshold


def _operation_parent(operation: ShardTreeOperation) -> ShardId:
    if isinstance(operation, ShardSplit):
        return operation.shard_id
    if isinstance(operation, ShardMerge):
        left, _right = operation.siblings
        return parent_shard_id(left)
    raise TypeError("operation must be ShardSplit or ShardMerge")


def _apply_operation(tree: ShardTree, operation: ShardTreeOperation) -> ShardTree:
    if isinstance(operation, ShardSplit):
        return apply_split(tree, operation.shard_id)
    if isinstance(operation, ShardMerge):
        return apply_merge(tree, operation.siblings)
    raise TypeError("operation must be ShardSplit or ShardMerge")


def _require_operation_tuple(name: str, operations: tuple[ShardTreeOperation, ...]) -> None:
    if not isinstance(operations, tuple):
        raise TypeError(f"{name} must be a tuple")
    for operation in operations:
        _operation_parent(operation)


def _require_tree(tree: ShardTree) -> None:
    if not isinstance(tree, ShardTree):
        raise TypeError("tree must be ShardTree")


def _require_uint(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} outside uint range")


def _path_mask(depth: int) -> int:
    return (1 << depth) - 1


def _u32(value: int) -> bytes:
    return value.to_bytes(U32_BYTES, "little")


__all__ = [
    "MAX_FRUIT_PAYLOAD_BYTES",
    "ROOT_SHARD_ID",
    "SHARD_ID_DEPTH_SHIFT",
    "SHARD_MAX_DEPTH",
    "SHARD_MERGE_THRESHOLD_PCT",
    "SHARD_MERGE_WINDOW_ANCHORS",
    "SHARD_SPLIT_THRESHOLD_PCT",
    "SHARD_SPLIT_WINDOW_ANCHORS",
    "SHARD_TREE_MAX_BYTES",
    "ShardId",
    "ShardMerge",
    "ShardSplit",
    "ShardTree",
    "ShardTreeDecodeError",
    "ShardTreeOperation",
    "ShardTreeUpdate",
    "ShardUtilizationSample",
    "apply_merge",
    "apply_operations",
    "apply_split",
    "child_shard_ids",
    "decode_shard_id",
    "encode_shard_id",
    "parent_shard_id",
    "require_shard_id",
    "require_sibling_shards",
    "shard_tree_root",
    "should_merge",
    "should_split",
]
