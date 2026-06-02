"""Adversarial regression for selfish mining with withheld fruits."""

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
from tests.adversarial._helpers import (
    ADVERSARY_FRUIT_WORK,
    ADVERSARY_REORG_COMPUTE_PCT_LIMIT,
    HONEST_FRUIT_WORK,
    coinbase_tx,
    fruit,
    genesis_anchor,
    h,
    trusted_adversarial_pow_verifier,
)


def test_withheld_fruits_under_forty_percent_do_not_reorg_economic_depth() -> None:
    dag = BlockDAG()
    genesis = h(1)
    dag.add_fruit(genesis, timestamp_ms=1)

    honest_tip = genesis
    honest_hashes: list[bytes] = []
    for index in range(2, 2 + FINALITY_ECONOMIC_BLUE_DEPTH + 6):
        honest_hash = h(index)
        dag.add_fruit(honest_hash, (honest_tip,), timestamp_ms=index, work=HONEST_FRUIT_WORK)
        honest_hashes.append(honest_hash)
        honest_tip = honest_hash
    protected_fruit = honest_hashes[0]

    attacker_tip = genesis
    attacker_hashes: set[bytes] = set()
    for index in range(100, 116):
        attacker_hash = h(index)
        dag.add_fruit(
            attacker_hash,
            (attacker_tip,),
            timestamp_ms=index,
            work=ADVERSARY_FRUIT_WORK,
        )
        attacker_hashes.add(attacker_hash)
        attacker_tip = attacker_hash

    merge_tip = h(1_000)
    dag.add_fruit(merge_tip, (honest_tip, attacker_tip), timestamp_ms=200)

    honest_work = len(honest_hashes) * HONEST_FRUIT_WORK
    attacker_work = len(attacker_hashes) * ADVERSARY_FRUIT_WORK
    total_post_genesis_work = honest_work + attacker_work
    assert attacker_work * 100 < ADVERSARY_REORG_COMPUTE_PCT_LIMIT * total_post_genesis_work
    assert dag.ghostdag_data(merge_tip, DYNAMIC_K_MIN).selected_parent == honest_tip
    assert attacker_hashes <= red_set(dag, merge_tip, DYNAMIC_K_MIN)
    assert blue_depth(dag, protected_fruit, merge_tip, DYNAMIC_K_MIN) >= (
        FINALITY_ECONOMIC_BLUE_DEPTH
    )
    assert finality_tier(dag, protected_fruit, merge_tip, DYNAMIC_K_MIN) is FinalityTier.ECONOMIC


def test_node_rejects_withheld_fruit_with_unknown_parent(tmp_path: Path) -> None:
    genesis = genesis_anchor()
    node = TensorPowNode(
        TensorPowConfig(
            data_dir=tmp_path / "selfish-node",
            expected_genesis_hash=genesis.block_hash(),
        ),
        pow_verifier=trusted_adversarial_pow_verifier,
    )
    try:
        honest = fruit(
            (coinbase_tx(31).to_bytes(),),
            nonce=31,
            timestamp_ms=2,
            parent_selected=GENESIS_PARENT_HASH,
            latest_anchor=genesis.block_hash(),
        )
        withheld = fruit(
            (coinbase_tx(32).to_bytes(),),
            nonce=32,
            timestamp_ms=3,
            parent_selected=h(99_999),
            latest_anchor=genesis.block_hash(),
        )

        assert node.process_anchor(genesis)
        assert node.process_fruit(honest)
        assert node.process_fruit(withheld).reason == "missing_fruit_parent"
        assert node.get_block(honest.block_hash()) == honest.serialize()
        assert node.get_block(withheld.block_hash()) is None
    finally:
        node.close()
