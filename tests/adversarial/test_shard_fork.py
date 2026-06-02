"""Adversarial regression for conflicting shard split and merge operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorpow.mempool import (
    ROOT_SHARD_ID,
    ShardMerge,
    ShardSplit,
    ShardTree,
    apply_operations,
    apply_split,
    encode_shard_id,
)
from tensorpow.node import TensorPowConfig, TensorPowNode
from tests.adversarial._helpers import anchor, coinbase_tx, fruit, genesis_anchor


def test_same_parent_shard_fork_conflicts_are_queued_deterministically() -> None:
    duplicate_split = ShardSplit(ROOT_SHARD_ID)
    root_update = apply_operations(ShardTree(), (duplicate_split, duplicate_split))

    assert root_update.applied == (duplicate_split,)
    assert root_update.queued == (duplicate_split,)
    assert root_update.tree.leaf_shard_ids == (encode_shard_id(1, 0), encode_shard_id(1, 1))

    depth_one = apply_split(ShardTree(), ROOT_SHARD_ID)
    depth_two = apply_operations(
        depth_one,
        (
            ShardSplit(encode_shard_id(1, 0)),
            ShardSplit(encode_shard_id(1, 1)),
        ),
    ).tree
    merge_left = ShardMerge((encode_shard_id(2, 0), encode_shard_id(2, 2)))
    duplicate_merge_left = ShardMerge((encode_shard_id(2, 2), encode_shard_id(2, 0)))
    independent_split = ShardSplit(encode_shard_id(2, 1))

    fork_update = apply_operations(
        depth_two,
        (merge_left, duplicate_merge_left, independent_split),
    )

    assert fork_update.applied == (merge_left, independent_split)
    assert fork_update.queued == (duplicate_merge_left,)
    assert ShardTree.deserialize(fork_update.tree.serialize()) == fork_update.tree
    assert fork_update.tree.leaf_shard_ids == (
        encode_shard_id(1, 0),
        encode_shard_id(2, 3),
        encode_shard_id(3, 1),
        encode_shard_id(3, 5),
    )


def test_malformed_shard_fork_partition_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ShardTree(
            (
                encode_shard_id(1, 0),
                encode_shard_id(1, 1),
                encode_shard_id(2, 0),
            )
        )


def test_node_rejects_malformed_shard_tree_anchor_after_split_fork(tmp_path: Path) -> None:
    genesis = genesis_anchor()
    node = TensorPowNode(
        TensorPowConfig(
            data_dir=tmp_path / "shard-fork-node",
            expected_genesis_hash=genesis.block_hash(),
        ),
        pow_verifier=lambda _header, _target, _backend: True,
    )
    try:
        seed_fruit = fruit(
            (coinbase_tx(61).to_bytes(),),
            nonce=61,
            timestamp_ms=2,
            latest_anchor=genesis.block_hash(),
        )
        bad_anchor = anchor(
            shard_tree_bytes=b"\x02\x00\x00\x00",
            covered_fruit_hashes=(seed_fruit.block_hash(),),
            parent_anchor=genesis.block_hash(),
            timestamp_ms=3,
            nonce=62,
        )

        assert node.process_anchor(genesis)
        assert node.process_fruit(seed_fruit)
        assert node.process_anchor(bad_anchor).reason == "bad_shard_tree"
        assert node.status()["shard_leaves"] == 1
    finally:
        node.close()
