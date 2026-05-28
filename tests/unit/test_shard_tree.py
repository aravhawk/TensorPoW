"""Tests for canonical hierarchical shard tree state."""

from __future__ import annotations

import pytest

from tensorpow.mempool import (
    MAX_FRUIT_PAYLOAD_BYTES,
    ROOT_SHARD_ID,
    SHARD_MAX_DEPTH,
    SHARD_MERGE_THRESHOLD_PCT,
    SHARD_MERGE_WINDOW_ANCHORS,
    SHARD_SPLIT_THRESHOLD_PCT,
    SHARD_SPLIT_WINDOW_ANCHORS,
    ShardMerge,
    ShardSplit,
    ShardTree,
    ShardTreeDecodeError,
    ShardUtilizationSample,
    apply_merge,
    apply_operations,
    apply_split,
    encode_shard_id,
    require_shard_id,
    require_sibling_shards,
    shard_tree_root,
    should_merge,
    should_split,
)

REFERENCE_SHARD_TREE_BYTES = "03000000010001000000020002000200"
REFERENCE_SHARD_TREE_ROOT = "38584ad27921618c8ed12144f94e919c73e3e48804ce693d3d6cbb41474df95a"


def test_route_tx_is_deterministic_over_low_hash_bits() -> None:
    tree = apply_split(apply_split(ShardTree(), ROOT_SHARD_ID), encode_shard_id(1, 0))

    assert tree.route_tx(_hash_with_low_bits(0)) == encode_shard_id(2, 0)
    assert tree.route_tx(_hash_with_low_bits(2)) == encode_shard_id(2, 2)
    assert tree.route_tx(_hash_with_low_bits(1)) == encode_shard_id(1, 1)
    assert tree.route_tx(_hash_with_low_bits(3)) == encode_shard_id(1, 1)
    assert tree.route_tx(_hash_with_low_bits(2)) == tree.route_tx(_hash_with_low_bits(2))


def test_shard_tree_serialization_hashes_and_round_trips_reference_vector() -> None:
    tree = apply_split(apply_split(ShardTree(), ROOT_SHARD_ID), encode_shard_id(1, 0))

    assert tree.leaf_shard_ids == (
        encode_shard_id(1, 1),
        encode_shard_id(2, 0),
        encode_shard_id(2, 2),
    )
    assert tree.serialize().hex() == REFERENCE_SHARD_TREE_BYTES
    assert tree.state_root().hex() == REFERENCE_SHARD_TREE_ROOT
    assert tree.commitment_root() == tree.state_root()
    assert shard_tree_root(tree.serialize()) == tree.state_root()
    assert ShardTree.deserialize(tree.serialize()) == tree


def test_split_threshold_uses_last_sustained_window() -> None:
    shard_id = ROOT_SHARD_ID
    high_payload = _ceil_percent(MAX_FRUIT_PAYLOAD_BYTES, SHARD_SPLIT_THRESHOLD_PCT)
    low_payload = high_payload - 1

    assert not should_split(
        shard_id,
        tuple(
            ShardUtilizationSample(shard_id, high_payload)
            for _ in range(SHARD_SPLIT_WINDOW_ANCHORS - 1)
        ),
    )
    assert should_split(
        shard_id,
        tuple(
            ShardUtilizationSample(shard_id, high_payload)
            for _ in range(SHARD_SPLIT_WINDOW_ANCHORS)
        ),
    )
    assert not should_split(
        shard_id,
        (
            *(
                ShardUtilizationSample(shard_id, high_payload)
                for _ in range(SHARD_SPLIT_WINDOW_ANCHORS - 1)
            ),
            ShardUtilizationSample(shard_id, low_payload),
        ),
    )
    assert should_split(
        shard_id,
        (
            ShardUtilizationSample(shard_id, low_payload),
            *(
                ShardUtilizationSample(shard_id, high_payload)
                for _ in range(SHARD_SPLIT_WINDOW_ANCHORS)
            ),
        ),
    )


def test_merge_threshold_requires_both_siblings_for_24h_window() -> None:
    left = encode_shard_id(1, 0)
    right = encode_shard_id(1, 1)
    low_payload = (MAX_FRUIT_PAYLOAD_BYTES * SHARD_MERGE_THRESHOLD_PCT) // 100
    high_payload = low_payload + 1
    low_history = (
        *(ShardUtilizationSample(left, low_payload) for _ in range(SHARD_MERGE_WINDOW_ANCHORS)),
        *(ShardUtilizationSample(right, low_payload) for _ in range(SHARD_MERGE_WINDOW_ANCHORS)),
    )

    assert should_merge((right, left), low_history)
    assert not should_merge((left, right), low_history[:-1])
    assert not should_merge(
        (left, right),
        (
            *low_history[:-1],
            ShardUtilizationSample(right, high_payload),
        ),
    )


def test_apply_merge_and_concurrent_independent_updates() -> None:
    depth_one = apply_split(ShardTree(), ROOT_SHARD_ID)
    depth_two = apply_operations(
        depth_one,
        (
            ShardSplit(encode_shard_id(1, 0)),
            ShardSplit(encode_shard_id(1, 1)),
        ),
    ).tree

    assert depth_two.leaf_shard_ids == tuple(encode_shard_id(2, path) for path in range(4))
    assert apply_merge(depth_one, (encode_shard_id(1, 1), encode_shard_id(1, 0))) == ShardTree()

    update = apply_operations(
        depth_two,
        (
            ShardMerge((encode_shard_id(2, 0), encode_shard_id(2, 2))),
            ShardSplit(encode_shard_id(2, 1)),
        ),
    )

    assert update.queued == ()
    assert update.tree.leaf_shard_ids == (
        encode_shard_id(1, 0),
        encode_shard_id(2, 3),
        encode_shard_id(3, 1),
        encode_shard_id(3, 5),
    )


def test_same_parent_conflicts_are_queued() -> None:
    tree = apply_operations(
        apply_split(ShardTree(), ROOT_SHARD_ID),
        (
            ShardSplit(encode_shard_id(1, 0)),
            ShardSplit(encode_shard_id(1, 1)),
        ),
    ).tree
    merge = ShardMerge((encode_shard_id(2, 0), encode_shard_id(2, 2)))
    duplicate_merge = ShardMerge((encode_shard_id(2, 2), encode_shard_id(2, 0)))

    update = apply_operations(tree, (merge, duplicate_merge))

    assert update.applied == (merge,)
    assert update.queued == (duplicate_merge,)
    assert update.tree.leaf_shard_ids == (
        encode_shard_id(1, 0),
        encode_shard_id(2, 1),
        encode_shard_id(2, 3),
    )


def test_max_depth_split_is_rejected() -> None:
    tree = _max_depth_zero_branch_tree()

    with pytest.raises(ValueError, match="exceed"):
        apply_split(tree, encode_shard_id(SHARD_MAX_DEPTH, 0))


def test_malformed_shard_ids_hashes_and_tree_bytes_are_rejected() -> None:
    with pytest.raises(ValueError, match="path"):
        require_shard_id(1)
    with pytest.raises(ValueError, match="depth"):
        encode_shard_id(SHARD_MAX_DEPTH + 1, 0)
    with pytest.raises(ValueError, match="same depth"):
        require_sibling_shards((encode_shard_id(1, 0), encode_shard_id(2, 0)))
    with pytest.raises(ValueError, match="32 bytes"):
        ShardTree().route_tx(b"short")
    with pytest.raises(TypeError, match="bytes"):
        shard_tree_root("not-bytes")

    with pytest.raises(ValueError, match="ascending"):
        ShardTree((encode_shard_id(1, 1), encode_shard_id(1, 0)))
    with pytest.raises(ValueError, match="duplicate"):
        ShardTree((encode_shard_id(1, 0), encode_shard_id(1, 0)))
    with pytest.raises(ValueError, match="overlap"):
        ShardTree((encode_shard_id(1, 0), encode_shard_id(1, 1), encode_shard_id(2, 0)))
    with pytest.raises(ValueError, match="gap"):
        ShardTree((encode_shard_id(1, 0),))

    with pytest.raises(ShardTreeDecodeError, match="nonzero"):
        ShardTree.deserialize((0).to_bytes(4, "little"))
    with pytest.raises(ShardTreeDecodeError, match="truncated"):
        ShardTree.deserialize(b"\x01\x00\x00")
    with pytest.raises(ShardTreeDecodeError, match="trailing"):
        ShardTree.deserialize(ShardTree().serialize() + b"\x00")


def test_malformed_history_is_rejected() -> None:
    with pytest.raises(TypeError, match="history"):
        should_split(ROOT_SHARD_ID, [object()])
    with pytest.raises(TypeError, match="sequence"):
        should_split(ROOT_SHARD_ID, iter(()))
    with pytest.raises(ValueError, match="positive"):
        ShardUtilizationSample(ROOT_SHARD_ID, 0, 0)
    with pytest.raises(ValueError, match="capacity"):
        ShardUtilizationSample(ROOT_SHARD_ID, MAX_FRUIT_PAYLOAD_BYTES + 1)


def _hash_with_low_bits(value: int) -> bytes:
    return value.to_bytes(32, "little")


def _ceil_percent(value: int, pct: int) -> int:
    return (value * pct + 99) // 100


def _max_depth_zero_branch_tree() -> ShardTree:
    leaves = [encode_shard_id(depth, 1 << (depth - 1)) for depth in range(1, SHARD_MAX_DEPTH + 1)]
    leaves.append(encode_shard_id(SHARD_MAX_DEPTH, 0))
    return ShardTree(tuple(sorted(leaves)))
