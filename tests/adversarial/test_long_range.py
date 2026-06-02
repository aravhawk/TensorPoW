"""Adversarial regression for long-range alternate-history attacks."""

from __future__ import annotations

from pathlib import Path

from tensorpow.consensus.finality import (
    FINALITY_ECONOMIC_BLUE_DEPTH,
    FinalityTier,
    blue_depth,
    finality_tier,
)
from tensorpow.consensus.ghostdag import DYNAMIC_K_MIN, BlockDAG, red_set
from tensorpow.node import TensorPowConfig, TensorPowNode
from tensorpow.pow.challenge import GENESIS_PARENT_HASH
from tests.adversarial._helpers import coinbase_tx, fruit, genesis_anchor, h


def test_late_alternate_history_cannot_replace_economic_fruit() -> None:
    dag = BlockDAG()
    genesis = h(1)
    dag.add_fruit(genesis, timestamp_ms=1)

    honest_tip = genesis
    honest_hashes: list[bytes] = []
    for index in range(2, 2 + FINALITY_ECONOMIC_BLUE_DEPTH + 4):
        honest_hash = h(index)
        dag.add_fruit(honest_hash, (honest_tip,), timestamp_ms=index)
        honest_hashes.append(honest_hash)
        honest_tip = honest_hash
    protected_fruit = honest_hashes[0]

    assert finality_tier(dag, protected_fruit, honest_tip, DYNAMIC_K_MIN) is FinalityTier.ECONOMIC

    alternate_tip = genesis
    alternate_hashes: set[bytes] = set()
    for index in range(500, 514):
        alternate_hash = h(index)
        dag.add_fruit(alternate_hash, (alternate_tip,), timestamp_ms=index)
        alternate_hashes.add(alternate_hash)
        alternate_tip = alternate_hash

    merge_tip = h(2_000)
    dag.add_fruit(merge_tip, (honest_tip, alternate_tip), timestamp_ms=700)

    total_post_genesis_work = len(honest_hashes) + len(alternate_hashes)
    assert len(alternate_hashes) * 100 < 40 * total_post_genesis_work
    assert dag.ghostdag_data(merge_tip, DYNAMIC_K_MIN).selected_parent == honest_tip
    assert blue_depth(dag, protected_fruit, alternate_tip, DYNAMIC_K_MIN) == 0
    assert alternate_hashes <= red_set(dag, merge_tip, DYNAMIC_K_MIN)
    assert finality_tier(dag, protected_fruit, merge_tip, DYNAMIC_K_MIN) is FinalityTier.ECONOMIC


def test_node_rejects_late_genesis_parent_alternate_history(tmp_path: Path) -> None:
    genesis = genesis_anchor()
    node = TensorPowNode(
        TensorPowConfig(
            data_dir=tmp_path / "long-range-node",
            expected_genesis_hash=genesis.block_hash(),
        ),
        pow_verifier=lambda _header, _target, _backend: True,
    )
    try:
        honest = fruit(
            (coinbase_tx(41).to_bytes(),),
            nonce=41,
            timestamp_ms=2,
            latest_anchor=genesis.block_hash(),
        )
        late_alternate = fruit(
            (coinbase_tx(42).to_bytes(),),
            nonce=42,
            timestamp_ms=1_000,
            parent_selected=GENESIS_PARENT_HASH,
            latest_anchor=genesis.block_hash(),
        )

        assert node.process_anchor(genesis)
        assert node.process_fruit(honest)
        assert node.process_fruit(late_alternate).reason == "genesis_fruit_parent_not_first"
        assert node.get_block(late_alternate.block_hash()) is None
    finally:
        node.close()
