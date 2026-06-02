"""Tests for GHOSTDAG ordering helpers."""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal, localcontext

import pytest

from tensorpow.consensus.ghostdag import (
    DYNAMIC_K_OBSERVATION_ANCHORS,
    BlockDAG,
    FruitBlock,
    blue_set,
    compute_k,
    red_set,
    topological_order,
)


def _h(index: int) -> bytes:
    return index.to_bytes(32, "big")


def test_compute_k_matches_reference_and_clamps_bounds() -> None:
    assert compute_k(35, 1000) == 968
    assert compute_k(Decimal("35"), Decimal("10")) == 97
    assert (
        compute_k(
            Decimal("35"),
            Decimal("1000"),
            observation_anchors=DYNAMIC_K_OBSERVATION_ANCHORS,
        )
        == 968
    )
    assert compute_k(0, 1000) == 15
    assert compute_k(1_000_000, 5000) == 10000

    with pytest.raises(ValueError, match="delta"):
        compute_k(1, 1000, 1)
    with pytest.raises(ValueError, match="positive"):
        compute_k(1, 1000, 0)
    with pytest.raises(ValueError, match="finite"):
        compute_k(Decimal("NaN"), 1000)
    with pytest.raises(TypeError, match="int or Decimal"):
        compute_k(1.5, 1000)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_k(True, 1000)
    with pytest.raises(ValueError, match="observation_anchors"):
        compute_k(1, 1000, observation_anchors=DYNAMIC_K_OBSERVATION_ANCHORS - 1)

    with localcontext() as context:
        context.prec = 5
        context.rounding = ROUND_FLOOR
        assert compute_k(Decimal("35"), Decimal("1000")) == 968


def test_linear_dag_is_all_blue_and_topologically_ordered() -> None:
    dag = BlockDAG()
    dag.add_fruit(_h(1), timestamp_ms=1)
    dag.add_fruit(_h(2), (_h(1),), timestamp_ms=2)
    dag.add_fruit(_h(3), (_h(2),), timestamp_ms=3)

    assert blue_set(dag, _h(3), 15) == {_h(1), _h(2), _h(3)}
    assert red_set(dag, _h(3), 15) == set()
    assert topological_order(dag, _h(3), 15) == [_h(1), _h(2), _h(3)]
    assert dag.ghostdag_data(_h(3), 15).selected_parent == _h(2)


def test_ancestors_are_cached_incrementally() -> None:
    dag = BlockDAG()
    dag.add_fruit(_h(1), timestamp_ms=1)
    dag.add_fruit(_h(2), (_h(1),), timestamp_ms=2)
    dag.add_fruit(_h(3), (_h(2),), timestamp_ms=3)

    ancestors = dag.ancestors(_h(3))

    assert ancestors == frozenset((_h(1), _h(2)))
    assert dag.ancestors(_h(3)) is ancestors


def test_merge_selects_highest_blue_work_then_lexicographic_parent() -> None:
    dag = BlockDAG()
    dag.add_fruit(_h(1), timestamp_ms=1, work=1)
    dag.add_fruit(_h(2), timestamp_ms=1, work=1)
    dag.add_fruit(_h(3), (_h(2), _h(1)), timestamp_ms=2)

    data = dag.ghostdag_data(_h(3), 0)

    assert data.selected_parent == _h(1)
    assert blue_set(dag, _h(3), 0) == {_h(1), _h(3)}
    assert red_set(dag, _h(3), 0) == {_h(2)}


def test_merge_prefers_more_accumulated_blue_work_before_hash_tie() -> None:
    dag = BlockDAG()
    dag.add_fruit(_h(1), timestamp_ms=1, work=1)
    dag.add_fruit(_h(2), timestamp_ms=1, work=5)
    dag.add_fruit(_h(3), (_h(1), _h(2)), timestamp_ms=2)

    assert dag.ghostdag_data(_h(3), 15).selected_parent == _h(2)


def test_adversarial_withheld_branch_is_red_when_anticone_exceeds_k() -> None:
    dag = BlockDAG()
    genesis = _h(1)
    dag.add_fruit(genesis, timestamp_ms=1)

    honest_tip = genesis
    for index in range(2, 27):
        dag.add_fruit(_h(index), (honest_tip,), timestamp_ms=index)
        honest_tip = _h(index)

    attacker_tip = genesis
    attacker_hashes = set()
    for index in range(100, 109):
        dag.add_fruit(_h(index), (attacker_tip,), timestamp_ms=index)
        attacker_tip = _h(index)
        attacker_hashes.add(attacker_tip)

    merge_tip = _h(200)
    dag.add_fruit(merge_tip, (honest_tip, attacker_tip), timestamp_ms=200)

    assert attacker_hashes <= red_set(dag, merge_tip, 0)
    assert honest_tip in blue_set(dag, merge_tip, 0)


def test_blockdag_rejects_malformed_records() -> None:
    dag = BlockDAG()
    dag.add_fruit(_h(1), timestamp_ms=1)

    with pytest.raises(ValueError, match="already"):
        dag.add_fruit(_h(1), timestamp_ms=2)
    with pytest.raises(KeyError, match="unknown"):
        dag.add_fruit(_h(2), (_h(99),), timestamp_ms=2)
    with pytest.raises(ValueError, match="timestamp"):
        dag.add_fruit(_h(2), (_h(1),), timestamp_ms=1)
    with pytest.raises(ValueError, match="duplicates"):
        FruitBlock(_h(2), (_h(1), _h(1)), timestamp_ms=2)
    with pytest.raises(ValueError):
        FruitBlock(b"short", (), timestamp_ms=0)
    with pytest.raises(ValueError, match="non-negative"):
        blue_set(dag, _h(1), -1)
