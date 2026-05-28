"""Ordered Merkle roots used by block bodies."""

from __future__ import annotations

from collections.abc import Sequence

from tensorpow.crypto.hash import (
    HASH_LEN_BYTES,
    MERKLE_EMPTY_TAG,
    MERKLE_LEAF_TAG,
    MERKLE_NODE_TAG,
    domain_hash,
)


def ordered_merkle_root(
    domain: int,
    items: Sequence[bytes],
    *,
    reject_duplicates: bool = False,
) -> bytes:
    """Return the ordered Merkle root defined for block-body commitments."""

    if not isinstance(items, Sequence):
        raise TypeError("items must be a sequence")
    checked_items = tuple(_require_bytes("item", item) for item in items)
    if reject_duplicates and len(set(checked_items)) != len(checked_items):
        raise ValueError("duplicate items are not allowed")
    if not checked_items:
        return domain_hash(domain, bytes((MERKLE_EMPTY_TAG,)) + (0).to_bytes(4, "little"))

    level = tuple(
        domain_hash(
            domain,
            bytes((MERKLE_LEAF_TAG,))
            + index.to_bytes(4, "little")
            + len(item).to_bytes(4, "little")
            + item,
        )
        for index, item in enumerate(checked_items)
    )
    level_number = 0
    while len(level) > 1:
        next_level = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = (
                level[index + 1]
                if index + 1 < len(level)
                else domain_hash(
                    domain,
                    bytes((MERKLE_EMPTY_TAG,))
                    + level_number.to_bytes(2, "little")
                    + (index // 2).to_bytes(4, "little"),
                )
            )
            next_level.append(
                domain_hash(
                    domain,
                    bytes((MERKLE_NODE_TAG,))
                    + level_number.to_bytes(2, "little")
                    + (index // 2).to_bytes(4, "little")
                    + left
                    + right,
                )
            )
        level = tuple(next_level)
        level_number += 1
    return level[0]


def _require_bytes(name: str, value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) > 0xFFFFFFFF:
        raise ValueError(f"{name} is too large")
    return value


def require_hash(name: str, value: bytes) -> bytes:
    """Validate and return a consensus hash."""

    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")
    return value
