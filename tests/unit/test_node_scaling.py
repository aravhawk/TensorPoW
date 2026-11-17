"""M-4: per-message work must not grow with total chain or mempool size."""

from __future__ import annotations

from pathlib import Path

import pytest

import tensorpow.node.node as node_module
from tensorpow.consensus.ghostdag import DYNAMIC_K_MIN, BlockDAG
from tensorpow.node import TensorPowNode
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint, UTXOSet
from tensorpow.storage import COLUMN_DAG
from tensorpow.wallet import Wallet
from tests.unit.test_node import (
    _canonical_anchor,
    _coinbase_tx,
    _fruit,
    _genesis_anchor,
    _node,
)


def _build_linear_chain(node: TensorPowNode, anchors: int) -> tuple[bytes, bytes, int]:
    genesis = _genesis_anchor()
    assert node.process_anchor(genesis)
    parent_anchor = genesis.block_hash()
    parent_fruit = bytes(32)
    timestamp = 1
    for index in range(anchors):
        fruit = _fruit(
            (_coinbase_tx(100 + index).to_bytes(),),
            latest_anchor=parent_anchor,
            parent_selected=parent_fruit,
            timestamp_ms=timestamp,
            nonce=1_000 + index,
        )
        assert node.process_fruit(fruit).accepted
        anchor = _canonical_anchor(
            node, (fruit.block_hash(),), parent_anchor=parent_anchor, timestamp_ms=timestamp + 1
        )
        assert node.process_anchor(anchor).accepted
        parent_anchor = anchor.block_hash()
        parent_fruit = fruit.block_hash()
        timestamp += 2
    return parent_anchor, parent_fruit, timestamp


def test_extending_the_tip_does_not_replay_the_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(tmp_path / "node")
    try:
        parent_anchor, parent_fruit, timestamp = _build_linear_chain(node, 12)
        fruit = _fruit(
            (_coinbase_tx(250).to_bytes(),),
            latest_anchor=parent_anchor,
            parent_selected=parent_fruit,
            timestamp_ms=timestamp,
            nonce=9_999,
        )
        assert node.process_fruit(fruit).accepted
        anchor = _canonical_anchor(
            node, (fruit.block_hash(),), parent_anchor=parent_anchor, timestamp_ms=timestamp + 1
        )

        applied: list[bytes] = []
        original = TensorPowNode._apply_anchor_to_branch

        def counting(self: TensorPowNode, anchor_: object, **kwargs: object) -> object:
            applied.append(kwargs["anchor_hash"])  # type: ignore[arg-type]
            return original(self, anchor_, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(TensorPowNode, "_apply_anchor_to_branch", counting)
        assert node.process_anchor(anchor).accepted
        assert applied == [anchor.block_hash()]
    finally:
        node.close()


def test_raw_tx_admission_does_not_rebuild_the_mempool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(tmp_path / "node")
    wallet = Wallet.recover("81" * 32)
    recipient = Wallet.recover("82" * 32)
    funded = tuple(
        UTXO(
            outpoint=Outpoint(bytes([index + 1]) * 32, 0),
            amount_matoms=10_000,
            template_id=TEMPLATE_PKH,
            owner_pubkey_hash=wallet.pubkey_hash(),
            payload=wallet.pubkey_hash(),
        )
        for index in range(3)
    )
    try:
        for utxo in funded:
            node.utxo_set.add(utxo)

        def fail(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("mempool rebuilt on the per-transaction path")

        monkeypatch.setattr(node, "_rebuild_mempool", fail)
        for utxo in funded:
            spend = wallet.create_signed_transaction(
                utxos=(utxo,),
                recipient_address=recipient.address,
                amount_matoms=1_000,
                fee_matoms=10,
            )
            result, _add = node.process_raw_tx(spend.to_bytes())
            assert result.accepted
        assert len(node.mempool) == 3
    finally:
        node.close()


def test_fruit_validation_does_not_copy_the_utxo_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    try:
        assert node.process_anchor(genesis)

        def fail(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("whole UTXO set materialised for one fruit")

        monkeypatch.setattr(node.utxo_set, "utxos", fail)
        monkeypatch.setattr(node, "_rebuild_mempool", fail)
        fruit = _fruit((_coinbase_tx(90).to_bytes(),), latest_anchor=genesis.block_hash())
        assert node.process_fruit(fruit).accepted
    finally:
        node.close()


def test_block_enumeration_uses_kind_index_and_migrates_old_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(tmp_path / "node")
    try:
        _build_linear_chain(node, 3)
        fruits = node._stored_fruit_hashes()
        anchors = node._stored_anchor_hashes()
        assert len(fruits) == 3
        assert len(anchors) == 4

        decodes: list[int] = []
        original = node_module.Fruit.deserialize

        def counting(data: bytes) -> object:
            decodes.append(1)
            return original(data)

        monkeypatch.setattr(node_module.Fruit, "deserialize", counting)
        assert node._stored_fruit_hashes() == fruits
        assert node._has_stored_fruits()
        assert decodes == []
        monkeypatch.undo()

        for key, _value in list(node.store.items(COLUMN_DAG)):
            if key.startswith((b"kind:", b"meta:kind-index")):
                node.store.delete(COLUMN_DAG, key)
    finally:
        node.close()

    reopened = _node(tmp_path / "node")
    try:
        assert reopened._stored_fruit_hashes() == fruits
        assert set(reopened._stored_anchor_hashes()) == set(anchors)
    finally:
        reopened.close()


def test_ghostdag_metadata_is_incremental_and_matches_full_rebuild() -> None:
    hashes = [bytes([index + 1]) * 32 for index in range(30)]
    incremental = BlockDAG()
    incremental.add_fruit(hashes[0], timestamp_ms=1)
    snapshots = []
    for index in range(1, len(hashes)):
        parents = (hashes[index - 1],) if index % 3 else (hashes[index - 1], hashes[index - 2])
        incremental.add_fruit(hashes[index], parents, timestamp_ms=index + 1)
        snapshots.append(incremental.ghostdag_metadata(DYNAMIC_K_MIN))

    rebuilt = BlockDAG()
    rebuilt.add_fruit(hashes[0], timestamp_ms=1)
    for index in range(1, len(hashes)):
        parents = (hashes[index - 1],) if index % 3 else (hashes[index - 1], hashes[index - 2])
        rebuilt.add_fruit(hashes[index], parents, timestamp_ms=index + 1)

    assert incremental.ghostdag_metadata(DYNAMIC_K_MIN) == rebuilt.ghostdag_metadata(DYNAMIC_K_MIN)
    # Earlier entries are reused, not recomputed, after a later block is added.
    assert snapshots[0][hashes[1]] is snapshots[-1][hashes[1]]


def test_utxo_set_copy_is_independent() -> None:
    owner = bytes([7]) * 32
    original = UTXOSet(
        (
            UTXO(
                outpoint=Outpoint(bytes([1]) * 32, 0),
                amount_matoms=5,
                template_id=TEMPLATE_PKH,
                owner_pubkey_hash=owner,
                payload=owner,
            ),
        )
    )
    root = original.merkle_root()
    clone = original.copy()
    clone.remove(Outpoint(bytes([1]) * 32, 0))

    assert len(original) == 1
    assert original.merkle_root() == root
    assert len(clone) == 0
    assert original.by_owner(owner)


def test_overlay_view_add_remove_semantics() -> None:
    owner = bytes([9]) * 32
    base_utxo = UTXO(
        outpoint=Outpoint(bytes([2]) * 32, 0),
        amount_matoms=5,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=owner,
        payload=owner,
    )
    base = UTXOSet((base_utxo,))
    view = node_module._OverlayUTXOView(base)

    assert view.remove(base_utxo.outpoint) == base_utxo
    assert view.get(base_utxo.outpoint) is None
    view.add(base_utxo)
    assert view.get(base_utxo.outpoint) == base_utxo
    with pytest.raises(KeyError):
        view.add(base_utxo)
    assert base.get(base_utxo.outpoint) == base_utxo
