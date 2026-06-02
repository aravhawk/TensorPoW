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
)
from tensorpow.chain.headers import AnchorHeader
from tensorpow.crypto.hash import hash_bytes
from tensorpow.mempool import ROOT_SHARD_ID, ShardTree
from tensorpow.node import TensorPowConfig, TensorPowNode
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.tx.transaction import Output
from tests.adversarial._helpers import (
    OWNER_PUBKEY_HASH,
    coinbase_tx,
    fruit,
    genesis_anchor,
    h,
    signed_tx,
    trusted_adversarial_pow_verifier,
)


def test_dag_order_allows_only_one_conflicting_spend(tmp_path: Path) -> None:
    genesis = genesis_anchor()
    funding_fruit = fruit(
        (coinbase_tx(0).to_bytes(),),
        latest_anchor=genesis.block_hash(),
        nonce=10,
        timestamp_ms=1,
        parent_selected=GENESIS_PARENT_HASH,
    )
    funding_output = Output(1_000, TEMPLATE_PKH, payload=OWNER_PUBKEY_HASH)
    funding_anchor = _anchor_for_fruits(
        funding_fruit,
        parent_anchor=genesis.block_hash(),
        anchor_reward_outputs=(funding_output,),
    )
    funded = UTXO(
        outpoint=Outpoint(hash_bytes(b"anchorreward:" + funding_anchor.block_hash()), 0),
        amount_matoms=funding_output.amount_matoms,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=OWNER_PUBKEY_HASH,
        payload=OWNER_PUBKEY_HASH,
    )
    left_spend = signed_tx(funded, fee=100, recipient_pubkey_hash=h(900))
    right_spend = signed_tx(funded, fee=101, recipient_pubkey_hash=h(901))
    left_fruit = fruit(
        (coinbase_tx(1).to_bytes(), left_spend.to_bytes()),
        nonce=11,
        timestamp_ms=3,
        parent_selected=funding_fruit.block_hash(),
        latest_anchor=funding_anchor.block_hash(),
    )
    right_fruit = fruit(
        (coinbase_tx(2).to_bytes(), right_spend.to_bytes()),
        nonce=12,
        timestamp_ms=3,
        parent_selected=funding_fruit.block_hash(),
        latest_anchor=funding_anchor.block_hash(),
    )

    node = TensorPowNode(
        TensorPowConfig(
            data_dir=tmp_path / "node",
            expected_genesis_hash=genesis.block_hash(),
        ),
        pow_verifier=trusted_adversarial_pow_verifier,
    )
    try:
        assert node.process_anchor(genesis)
        assert node.process_fruit(funding_fruit)
        assert node.process_anchor(funding_anchor)
        assert node.utxo_set.get(funded.outpoint) == funded
        first = node.process_fruit(left_fruit)
        second = node.process_fruit(right_fruit)
        anchor = _anchor_for_fruits(
            left_fruit,
            right_fruit,
            parent_anchor=funding_anchor.block_hash(),
        )
        anchored = node.process_anchor(anchor)

        assert first.accepted
        assert second.accepted
        assert anchored.accepted
        assert node.utxo_set.get(funded.outpoint) is None
        surviving_outputs = [
            outpoint
            for outpoint in (
                Outpoint(left_spend.tx_id(), 0),
                Outpoint(right_spend.tx_id(), 0),
            )
            if node.utxo_set.get(outpoint) is not None
        ]
        assert len(surviving_outputs) == 1
    finally:
        node.close()


def _anchor_for_fruits(
    *fruits: Fruit,
    parent_anchor: bytes,
    anchor_reward_outputs: tuple[Output, ...] = (),
) -> Anchor:
    if not fruits:
        raise ValueError("at least one fruit is required")
    tree = ShardTree()
    fee_entries = (FeeFloorEntry(ROOT_SHARD_ID, 0),)
    covered = tuple(sorted(fruit_.block_hash() for fruit_ in fruits))
    parent_candidates = covered
    header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=parent_anchor,
        fruit_set_root=fruit_set_root(covered),
        parent_candidate_root=parent_candidate_root(parent_candidates),
        shard_tree_state_root=tree.state_root(),
        fee_floor_set_root=fee_floor_set_root(fee_entries),
        anchor_reward_root=anchor_reward_root(anchor_reward_outputs),
        timestamp_ms=max(fruit_.header.timestamp_ms for fruit_ in fruits) + 1,
        nonce=99,
    )
    return Anchor(
        header=header,
        covered_fruit_hashes=covered,
        parent_candidate_hashes=parent_candidates,
        shard_tree_bytes=tree.serialize(),
        fee_floor_entries=fee_entries,
        anchor_reward_outputs=anchor_reward_outputs,
    )
