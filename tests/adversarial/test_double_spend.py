"""Adversarial regression for concurrent double-spend attempts."""

from __future__ import annotations

from pathlib import Path

from tensorpow.chain.blocks import (
    Anchor,
    FeeFloorEntry,
    Fruit,
    anchor_reward_root,
    fee_floor_set_root,
    fruit_set_root,
    parent_candidate_root,
    tx_merkle_root,
)
from tensorpow.chain.headers import AnchorHeader, FruitHeader
from tensorpow.consensus.ghostdag import BlockDAG, topological_order
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.mempool import ROOT_SHARD_ID, ShardTree
from tensorpow.node import TensorPowConfig, TensorPowNode
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH
from tensorpow.state.utxo import Outpoint
from tests.adversarial._helpers import coinbase_tx, fruit, genesis_anchor, h, signed_tx, utxo


def test_dag_order_allows_only_one_conflicting_spend(tmp_path: Path) -> None:
    funded = utxo(100, amount=1_000)
    left_spend = signed_tx(funded, fee=100, recipient_pubkey_hash=h(900))
    right_spend = signed_tx(funded, fee=101, recipient_pubkey_hash=h(901))
    left_fruit = fruit(
        (coinbase_tx(1).to_bytes(), left_spend.to_bytes()),
        nonce=11,
        timestamp_ms=2,
    )
    right_fruit = fruit(
        (coinbase_tx(2).to_bytes(), right_spend.to_bytes()),
        nonce=12,
        timestamp_ms=2,
    )

    dag = BlockDAG()
    dag.add_fruit(h(1), timestamp_ms=1)
    fruits_by_hash = {
        left_fruit.block_hash(): left_fruit,
        right_fruit.block_hash(): right_fruit,
    }
    for fruit_hash in fruits_by_hash:
        dag.add_fruit(fruit_hash, (h(1),), timestamp_ms=2)
    merge_tip = h(999)
    dag.add_fruit(merge_tip, tuple(fruits_by_hash), timestamp_ms=3)
    canonical_fruits = [
        fruits_by_hash[fruit_hash]
        for fruit_hash in topological_order(dag, merge_tip, 0)
        if fruit_hash in fruits_by_hash
    ]

    node = TensorPowNode(
        TensorPowConfig(data_dir=tmp_path / "node"),
        pow_verifier=lambda _header, _target, _backend: True,
    )
    try:
        genesis = genesis_anchor()
        assert node.process_anchor(genesis)
        node.utxo_set.add(funded)
        first_fruit = _with_node_parents(
            canonical_fruits[0],
            parent_selected=GENESIS_PARENT_HASH,
            latest_anchor=genesis.block_hash(),
        )
        second_fruit = _with_node_parents(
            canonical_fruits[1],
            parent_selected=first_fruit.block_hash(),
            latest_anchor=genesis.block_hash(),
            timestamp_ms=first_fruit.header.timestamp_ms + 1,
        )
        first = node.process_fruit(first_fruit)
        second = node.process_fruit(second_fruit)
        anchor = _anchor_for_fruits(
            first_fruit,
            second_fruit,
            parent_anchor=genesis.block_hash(),
        )
        anchored = node.process_anchor(anchor)

        winner_tx = left_spend if canonical_fruits[0] == left_fruit else right_spend
        loser_tx = right_spend if winner_tx == left_spend else left_spend

        assert first.accepted
        assert second.accepted
        assert anchored.accepted
        assert node.utxo_set.get(funded.outpoint) is None
        assert node.utxo_set.get(Outpoint(winner_tx.tx_id(), 0)) is not None
        assert node.utxo_set.get(Outpoint(loser_tx.tx_id(), 0)) is None
    finally:
        node.close()


def _with_node_parents(
    fruit_: Fruit,
    *,
    parent_selected: bytes,
    latest_anchor: bytes,
    timestamp_ms: int | None = None,
) -> Fruit:
    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=parent_selected,
        parent_bitmap=b"",
        latest_anchor=latest_anchor,
        tx_merkle_root=tx_merkle_root(fruit_.transactions),
        timestamp_ms=fruit_.header.timestamp_ms if timestamp_ms is None else timestamp_ms,
        shard_id=fruit_.header.shard_id,
        nonce=fruit_.header.nonce,
    )
    return Fruit(header=header, transactions=fruit_.transactions)


def _anchor_for_fruits(
    first_fruit: Fruit,
    second_fruit: Fruit,
    *,
    parent_anchor: bytes,
) -> Anchor:
    tree = ShardTree()
    fee_entries = (FeeFloorEntry(ROOT_SHARD_ID, 0),)
    covered = tuple(sorted((first_fruit.block_hash(), second_fruit.block_hash())))
    parent_candidates = (second_fruit.block_hash(),)
    header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=parent_anchor,
        fruit_set_root=fruit_set_root(covered),
        parent_candidate_root=parent_candidate_root(parent_candidates),
        shard_tree_state_root=tree.state_root(),
        fee_floor_set_root=fee_floor_set_root(fee_entries),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=second_fruit.header.timestamp_ms + 1,
        nonce=99,
    )
    return Anchor(
        header=header,
        covered_fruit_hashes=covered,
        parent_candidate_hashes=parent_candidates,
        shard_tree_bytes=tree.serialize(),
        fee_floor_entries=fee_entries,
    )
