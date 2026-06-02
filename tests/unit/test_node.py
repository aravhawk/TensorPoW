"""Tests for full-node orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

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
from tensorpow.consensus.anchor_daa import ANCHOR_INITIAL_TARGET_LE, AnchorRecord
from tensorpow.consensus.rewards import interval_subsidy_matoms, reward_pools
from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.genesis import GENESIS_CHAIN_ID_TESTNET, GenesisInputs, build_genesis_artifact
from tensorpow.mempool import ROOT_SHARD_ID, ShardTree
from tensorpow.node import TensorPowConfig, TensorPowNode, write_default_config
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH
from tensorpow.state.utxo import MAX_SUPPLY_MATOMS, TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.tx.transaction import Output, Transaction
from tensorpow.wallet import Wallet


def test_config_round_trip_and_start_stop(tmp_path: Path) -> None:
    config_path = write_default_config(
        tmp_path,
        TensorPowConfig(
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            data_dir=tmp_path / "node",
            expected_genesis_hash=bytes.fromhex("11" * 32),
            rpc_port=19001,
            p2p_tcp_port=19002,
        ),
    )

    config = TensorPowConfig.from_toml(config_path)
    assert config.chain_id == GENESIS_CHAIN_ID_TESTNET
    assert config.expected_genesis_hash == bytes.fromhex("11" * 32)
    node = TensorPowNode(config)
    asyncio.run(node.start())
    assert node.running
    asyncio.run(node.stop())
    assert not node.running
    node.close()


def test_process_fruit_updates_utxo_and_rejects_bad_payloads(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    tx = _coinbase_tx(3)
    fruit = _fruit((tx.to_bytes(),), latest_anchor=genesis.block_hash())
    anchor = _anchor(fruit.block_hash(), parent_anchor=genesis.block_hash())

    assert node.process_anchor(genesis)
    accepted = node.process_fruit(fruit)
    anchored = node.process_anchor(anchor)

    assert accepted.accepted
    assert anchored.accepted
    assert accepted.object_hash == fruit.block_hash()
    assert node.get_block(fruit.block_hash()) == fruit.serialize()
    assert node.utxo_set.get(Outpoint(tx.tx_id(), 0)) is not None
    assert node.process_fruit_bytes(fruit.serialize()[:-1]).reason == "malformed_fruit"

    bad_fruit = bytearray(fruit.serialize())
    bad_fruit[-1] ^= 0x01
    assert node.process_fruit_bytes(bytes(bad_fruit)).reason == "malformed_fruit"
    node.close()


def test_process_anchor_updates_shard_tree_fee_floor_and_syncs_peer(tmp_path: Path) -> None:
    first = _node(tmp_path / "first")
    second = _node(tmp_path / "second")
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(4).to_bytes(),), latest_anchor=genesis.block_hash())
    anchor = _anchor(fruit.block_hash(), parent_anchor=genesis.block_hash())

    assert first.process_anchor(genesis)
    assert first.process_fruit(fruit)
    assert first.process_anchor(anchor)
    second.sync_from(first)

    assert second.get_block(genesis.block_hash()) == genesis.serialize()
    assert second.get_block(fruit.block_hash()) == fruit.serialize()
    assert second.get_block(anchor.block_hash()) == anchor.serialize()
    assert second.shard_tree == ShardTree()
    assert second.store.fee_floor(ROOT_SHARD_ID) == 5
    assert second.status()["blocks"] == 3
    first.close()
    second.close()


def test_inbound_tx_rejects_coinbase_and_malformed_bytes(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")

    coinbase = _coinbase_tx(5)
    malformed, add_result = node.process_raw_tx(b"not-a-transaction")
    rejected, rejected_add = node.process_raw_tx(coinbase.to_bytes())

    assert not malformed.accepted
    assert malformed.reason == "malformed_tx"
    assert add_result is None
    assert not rejected.accepted
    assert rejected.reason == "coinbase_not_relayable"
    assert rejected_add is None
    node.close()


def test_invalid_anchor_is_rejected(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    anchor = _anchor(bytes.fromhex("01" * 32), parent_anchor=bytes.fromhex("02" * 32))
    corrupted = bytearray(anchor.serialize())
    corrupted[2 + 2 + HASH_LEN_BYTES] ^= 0x01

    assert node.process_anchor_bytes(bytes(corrupted)).reason == "malformed_anchor"
    node.close()


def test_anchor_dependencies_are_required(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(6).to_bytes(),), latest_anchor=genesis.block_hash())
    anchor = _anchor(fruit.block_hash(), parent_anchor=genesis.block_hash())

    assert node.process_anchor(anchor).reason == "missing_anchor_parent"
    assert node.process_anchor(genesis)
    assert node.process_anchor(genesis).reason == "genesis_not_first"
    assert node.process_anchor(anchor).reason == "missing_covered_fruit"
    assert node.process_fruit(fruit)
    assert node.process_anchor(anchor)
    node.close()


def test_expected_genesis_hash_is_enforced(tmp_path: Path) -> None:
    genesis = _genesis_anchor()
    wrong_genesis = build_genesis_artifact(
        GenesisInputs.create(
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            whitepaper_hash=bytes.fromhex("11" * 32),
            bitcoin_block_hash=bytes.fromhex("22" * 32),
            ethereum_block_hash=bytes.fromhex("33" * 32),
            founder_pubkey_hash=bytes.fromhex("44" * 32),
        )
    ).anchor
    node = TensorPowNode(
        TensorPowConfig(
            data_dir=tmp_path / "node",
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            expected_genesis_hash=genesis.block_hash(),
        ),
        pow_verifier=lambda _header, _target, _backend: True,
    )

    assert node.process_anchor(wrong_genesis).reason == "wrong_genesis"
    assert node.process_anchor(genesis)
    node.close()


def test_unpinned_genesis_hash_is_rejected(tmp_path: Path) -> None:
    genesis = _genesis_anchor()
    node = TensorPowNode(
        TensorPowConfig(data_dir=tmp_path / "node"),
        pow_verifier=lambda _header, _target, _backend: True,
    )

    assert node.process_anchor(genesis).reason == "missing_expected_genesis_hash"
    node.close()


def test_fruit_dependencies_pow_and_coinbase_limits_are_required(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(7).to_bytes(),), latest_anchor=genesis.block_hash())

    assert node.process_fruit(fruit).reason == "missing_latest_anchor"
    assert node.process_anchor(genesis)
    assert (
        node.process_fruit(_fruit((_coinbase_tx(8).to_bytes(),))).reason == "missing_latest_anchor"
    )

    bad_parent = _fruit(
        (_coinbase_tx(9).to_bytes(),),
        latest_anchor=genesis.block_hash(),
        parent_selected=bytes.fromhex("ab" * 32),
    )
    assert node.process_fruit(bad_parent).reason == "missing_fruit_parent"

    too_large = Transaction.coinbase(
        (Output(MAX_SUPPLY_MATOMS, TEMPLATE_PKH, payload=bytes.fromhex("09" * 32)),)
    )
    too_large_fruit = _fruit((too_large.to_bytes(),), latest_anchor=genesis.block_hash())
    assert node.process_fruit(too_large_fruit).reason == "coinbase_too_large"
    node.close()


def test_pow_verifier_rejection_blocks_fruit_and_anchor(tmp_path: Path) -> None:
    genesis = _genesis_anchor()
    node = TensorPowNode(
        TensorPowConfig(
            data_dir=tmp_path / "node",
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            expected_genesis_hash=genesis.block_hash(),
        ),
        pow_verifier=lambda _header, _target, _backend: False,
    )
    fruit = _fruit((_coinbase_tx(10).to_bytes(),), latest_anchor=genesis.block_hash())
    anchor = _anchor(fruit.block_hash(), parent_anchor=genesis.block_hash())

    assert node.process_anchor(genesis)
    assert node.process_fruit(fruit).reason == "fruit_pow_invalid"
    node.close()

    accepting = _node(tmp_path / "accepting")
    try:
        assert accepting.process_anchor(genesis)
        assert accepting.process_fruit(fruit)
        rejecting_anchor = TensorPowNode(
            TensorPowConfig(
                data_dir=tmp_path / "rejecting-anchor",
                chain_id=GENESIS_CHAIN_ID_TESTNET,
                expected_genesis_hash=genesis.block_hash(),
            ),
            pow_verifier=lambda _header, _target, _backend: False,
        )
        try:
            rejecting_anchor.sync_from(accepting)
            assert rejecting_anchor.process_anchor(anchor).reason == "anchor_pow_invalid"
        finally:
            rejecting_anchor.close()
    finally:
        accepting.close()


def test_anchored_fee_floor_is_enforced_for_raw_transactions(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    wallet = Wallet.recover("11" * 32)
    recipient = Wallet.recover("22" * 32)
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(11).to_bytes(),), latest_anchor=genesis.block_hash())
    anchor = _anchor(
        fruit.block_hash(),
        parent_anchor=genesis.block_hash(),
        fee_floor_matoms_per_kb=100_000,
    )
    funded = UTXO(
        outpoint=Outpoint(bytes.fromhex("ee" * 32), 0),
        amount_matoms=10_000,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=wallet.pubkey_hash(),
        payload=wallet.pubkey_hash(),
    )
    low_fee_tx = wallet.create_signed_transaction(
        utxos=(funded,),
        recipient_address=recipient.address,
        amount_matoms=1_000,
        fee_matoms=1,
    )

    assert node.process_anchor(genesis)
    assert node.process_fruit(fruit)
    assert node.process_anchor(anchor)
    node.utxo_set.add(funded)
    result, mempool_result = node.process_raw_tx(low_fee_tx.to_bytes())

    assert result.reason == "below_fee_floor"
    assert mempool_result is not None
    assert mempool_result.reason == "below_fee_floor"
    node.close()


def test_anchor_reward_outputs_are_committed_and_claimed(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(14).to_bytes(),), latest_anchor=genesis.block_hash())
    reward_output = Output(1, TEMPLATE_PKH, payload=bytes.fromhex("14" * 32))
    anchor = _anchor(
        fruit.block_hash(),
        parent_anchor=genesis.block_hash(),
        anchor_reward_outputs=(reward_output,),
    )
    reward_outpoint = Outpoint(hash_bytes(b"anchorreward:" + anchor.block_hash()), 0)

    assert node.process_anchor(genesis)
    assert node.process_fruit(fruit)
    assert node.process_anchor(anchor)
    assert node.utxo_set.get(reward_outpoint) is not None
    node.close()


def test_anchor_reward_overclaim_is_rejected(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(15).to_bytes(),), latest_anchor=genesis.block_hash())
    interval_subsidy = interval_subsidy_matoms(1)
    _fruit_pool, anchor_pool = reward_pools(
        fruit_count=1,
        interval_subsidy=interval_subsidy,
        anchor_target=ANCHOR_INITIAL_TARGET_LE,
    )
    reward_output = Output(anchor_pool + 1, TEMPLATE_PKH, payload=bytes.fromhex("15" * 32))
    anchor = _anchor(
        fruit.block_hash(),
        parent_anchor=genesis.block_hash(),
        anchor_reward_outputs=(reward_output,),
    )

    assert node.process_anchor(genesis)
    assert node.process_fruit(fruit)
    assert node.process_anchor(anchor).reason == "anchor_reward_too_large"
    node.close()


def test_duplicate_coinbase_outpoints_reject_deterministically(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    coinbase = _coinbase_tx(16)
    first = _fruit(
        (coinbase.to_bytes(),),
        latest_anchor=genesis.block_hash(),
        timestamp_ms=1,
        nonce=16,
    )
    second = _fruit(
        (coinbase.to_bytes(),),
        latest_anchor=genesis.block_hash(),
        parent_selected=first.block_hash(),
        timestamp_ms=2,
        nonce=17,
    )
    anchor = _anchor_many(
        (first.block_hash(), second.block_hash()),
        parent_anchor=genesis.block_hash(),
        parent_candidate_hashes=(second.block_hash(),),
    )

    assert node.process_anchor(genesis)
    assert node.process_fruit(first)
    assert node.process_fruit(second)
    assert node.process_anchor(anchor).reason == "duplicate_coinbase_outpoint"
    node.close()


def test_fruit_inclusion_enforces_transaction_locks(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    wallet = Wallet.recover("55" * 32)
    recipient = Wallet.recover("66" * 32)
    genesis = _genesis_anchor()
    funded = UTXO(
        outpoint=Outpoint(bytes.fromhex("55" * 32), 0),
        amount_matoms=10_000,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=wallet.pubkey_hash(),
        payload=wallet.pubkey_hash(),
    )
    locked = wallet.create_signed_transaction(
        utxos=(funded,),
        recipient_address=recipient.address,
        amount_matoms=1_000,
        fee_matoms=100,
        locktime_ms=10_000,
    )
    fruit = _fruit(
        (_coinbase_tx(18).to_bytes(), locked.to_bytes()),
        latest_anchor=genesis.block_hash(),
        timestamp_ms=1,
        nonce=18,
    )

    assert node.process_anchor(genesis)
    node.store.put_utxo(funded)
    node.utxo_set.add(funded)
    assert node.process_fruit(fruit).reason == "tx_lock_unmatured"
    node.close()


def test_fruit_timestamps_reject_future_and_parent_time_regressions(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    future = _fruit(
        (_coinbase_tx(19).to_bytes(),),
        latest_anchor=genesis.block_hash(),
        timestamp_ms=10**15,
        nonce=19,
    )
    parent = _fruit(
        (_coinbase_tx(20).to_bytes(),),
        latest_anchor=genesis.block_hash(),
        timestamp_ms=10,
        nonce=20,
    )
    child = _fruit(
        (_coinbase_tx(21).to_bytes(),),
        latest_anchor=genesis.block_hash(),
        parent_selected=parent.block_hash(),
        timestamp_ms=10,
        nonce=21,
    )

    assert node.process_anchor(genesis)
    assert node.process_fruit(future).reason == "fruit_time_too_new"
    assert node.process_fruit(parent)
    assert node.process_fruit(child).reason == "fruit_time_not_after_parent"
    node.close()


def test_anchor_timestamp_rejects_median_time_past_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(tmp_path / "node")
    covered_fruit_hash = bytes.fromhex("77" * HASH_LEN_BYTES)
    history_hashes = tuple(index.to_bytes(HASH_LEN_BYTES, "little") for index in range(1, 12))
    timestamps = (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 1)
    history = tuple(
        AnchorRecord(
            anchor_hash=history_hashes[index],
            parent_anchor=GENESIS_PARENT_HASH if index == 0 else history_hashes[index - 1],
            timestamp_ms=timestamps[index],
            target=ANCHOR_INITIAL_TARGET_LE,
        )
        for index in range(len(history_hashes))
    )
    anchor = _anchor_many(
        (covered_fruit_hash,),
        parent_anchor=history[-1].anchor_hash,
        parent_candidate_hashes=(covered_fruit_hash,),
        timestamp_ms=2,
    )
    dummy_fruit = _fruit((_coinbase_tx(31).to_bytes(),), latest_anchor=history[-1].anchor_hash)

    monkeypatch.setattr(node, "_anchor_history", lambda _tip_hash: history)
    monkeypatch.setattr(node, "_load_fruit", lambda _fruit_hash: dummy_fruit)
    monkeypatch.setattr(node, "_canonical_parent_candidates", lambda: (covered_fruit_hash,))

    assert node._validate_anchor_dependencies(anchor) == "anchor_time_too_old"
    node.close()


def test_anchor_parent_candidates_must_match_canonical_frontier(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(22).to_bytes(),), latest_anchor=genesis.block_hash())
    anchor = _anchor_many(
        (fruit.block_hash(),),
        parent_anchor=genesis.block_hash(),
        parent_candidate_hashes=(),
    )

    assert node.process_anchor(genesis)
    assert node.process_fruit(fruit)
    assert node.process_anchor(anchor).reason == "bad_parent_candidate_order"
    node.close()


def test_competing_anchor_height_comes_from_parent_chain(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    first_fruit = _fruit((_coinbase_tx(24).to_bytes(),), latest_anchor=genesis.block_hash())
    first_anchor = _anchor(first_fruit.block_hash(), parent_anchor=genesis.block_hash())
    sibling_fruit = _fruit(
        (_coinbase_tx(25).to_bytes(),),
        latest_anchor=genesis.block_hash(),
        parent_selected=first_fruit.block_hash(),
        timestamp_ms=2,
        nonce=25,
    )
    sibling_anchor = _anchor(
        sibling_fruit.block_hash(),
        parent_anchor=genesis.block_hash(),
        timestamp_ms=3,
    )

    assert node.process_anchor(genesis)
    assert node.process_fruit(first_fruit)
    assert node.process_anchor(first_anchor)
    assert node.process_fruit(sibling_fruit)
    assert node.process_anchor(sibling_anchor)

    first_meta = node._anchor_meta(first_anchor.block_hash())
    sibling_meta = node._anchor_meta(sibling_anchor.block_hash())
    assert first_meta is not None
    assert sibling_meta is not None
    assert first_meta.height == 1
    assert sibling_meta.height == 1
    node.close()


def test_higher_work_side_branch_reorgs_active_utxo_state(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    old_coinbase = _coinbase_tx(26)
    old_fruit = _fruit((old_coinbase.to_bytes(),), latest_anchor=genesis.block_hash())
    old_anchor = _anchor(old_fruit.block_hash(), parent_anchor=genesis.block_hash())
    side_coinbase = _coinbase_tx(27)
    side_fruit = _fruit(
        (side_coinbase.to_bytes(),),
        latest_anchor=genesis.block_hash(),
        parent_selected=old_fruit.block_hash(),
        timestamp_ms=2,
        nonce=27,
    )
    side_anchor = _anchor(
        side_fruit.block_hash(),
        parent_anchor=genesis.block_hash(),
        timestamp_ms=3,
    )
    side_child_coinbase = _coinbase_tx(28)
    side_child_fruit = _fruit(
        (side_child_coinbase.to_bytes(),),
        latest_anchor=side_anchor.block_hash(),
        parent_selected=side_fruit.block_hash(),
        timestamp_ms=4,
        nonce=28,
    )
    side_child_anchor = _anchor(
        side_child_fruit.block_hash(),
        parent_anchor=side_anchor.block_hash(),
        timestamp_ms=5,
    )

    assert node.process_anchor(genesis)
    assert node.process_fruit(old_fruit)
    assert node.process_anchor(old_anchor)
    assert node.utxo_set.get(Outpoint(old_coinbase.tx_id(), 0)) is not None

    assert node.process_fruit(side_fruit)
    assert node.process_anchor(side_anchor)
    assert node.process_fruit(side_child_fruit)
    assert node.process_anchor(side_child_anchor)

    assert node._anchor_height() == 2
    assert node.utxo_set.get(Outpoint(old_coinbase.tx_id(), 0)) is None
    assert node.utxo_set.get(Outpoint(side_coinbase.tx_id(), 0)) is not None
    assert node.utxo_set.get(Outpoint(side_child_coinbase.tx_id(), 0)) is not None
    node.close()


def test_node_finality_comes_from_accepted_dag_and_anchors(tmp_path: Path) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(23).to_bytes(),), latest_anchor=genesis.block_hash())
    anchor = _anchor(fruit.block_hash(), parent_anchor=genesis.block_hash())

    assert node.get_finality(fruit.block_hash())["seen"] is False
    assert node.process_anchor(genesis)
    assert node.process_fruit(fruit)
    assert node.process_anchor(anchor)

    finality = node.get_finality(fruit.block_hash())
    assert finality["seen"] is True
    assert finality["tier"] == "AnchorSecured"
    assert finality["anchor_depth"] == 1
    assert "AnchorSecured" in finality["satisfied_tiers"]
    node.close()


def test_sync_deletes_stale_state(tmp_path: Path) -> None:
    source = _node(tmp_path / "source")
    target = _node(tmp_path / "target")
    extra = UTXO(
        outpoint=Outpoint(bytes.fromhex("ef" * 32), 0),
        amount_matoms=10,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=bytes.fromhex("01" * 32),
        payload=bytes.fromhex("01" * 32),
    )
    try:
        target.store.put_utxo(extra)
        target.utxo_set.add(extra)

        target.sync_from(source)

        assert target.store.get_utxo(extra.outpoint) is None
        assert target.utxo_set.get(extra.outpoint) is None
    finally:
        source.close()
        target.close()


def test_fruit_persistence_failure_does_not_mutate_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    fruit = _fruit((_coinbase_tx(12).to_bytes(),), latest_anchor=genesis.block_hash())

    assert node.process_anchor(genesis)
    before_status = node.status()

    def fail_write(_batch: object) -> None:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(node.store, "write_batch", fail_write)

    with pytest.raises(RuntimeError, match="simulated"):
        node.process_fruit(fruit)
    assert node.status() == before_status
    assert node.get_block(fruit.block_hash()) is None
    node.close()


def test_anchor_persistence_failure_does_not_mutate_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(tmp_path / "node")
    genesis = _genesis_anchor()
    tx = _coinbase_tx(13)
    fruit = _fruit((tx.to_bytes(),), latest_anchor=genesis.block_hash())
    anchor = _anchor(fruit.block_hash(), parent_anchor=genesis.block_hash())

    assert node.process_anchor(genesis)
    assert node.process_fruit(fruit)
    before_status = node.status()

    def fail_write(_batch: object) -> None:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(node.store, "write_batch", fail_write)

    with pytest.raises(RuntimeError, match="simulated"):
        node.process_anchor(anchor)
    assert node.status() == before_status
    assert node.get_block(anchor.block_hash()) is None
    assert node.utxo_set.get(Outpoint(tx.tx_id(), 0)) is None
    node.close()


def test_raw_tx_persistence_failure_does_not_mutate_mempool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(tmp_path / "node")
    wallet = Wallet.recover("33" * 32)
    recipient = Wallet.recover("44" * 32)
    funded = UTXO(
        outpoint=Outpoint(bytes.fromhex("ed" * 32), 0),
        amount_matoms=10_000,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=wallet.pubkey_hash(),
        payload=wallet.pubkey_hash(),
    )
    spend = wallet.create_signed_transaction(
        utxos=(funded,),
        recipient_address=recipient.address,
        amount_matoms=1_000,
        fee_matoms=1,
    )
    node.utxo_set.add(funded)
    before_status = node.status()

    def fail_put(_tx: object) -> None:
        raise RuntimeError("simulated put failure")

    monkeypatch.setattr(node.store, "put_mempool_tx", fail_put)

    with pytest.raises(RuntimeError, match="simulated"):
        node.process_raw_tx(spend.to_bytes())
    assert node.status() == before_status
    assert node.get_tx(spend.tx_id()) is None
    node.close()


def _coinbase_tx(seed: int) -> Transaction:
    return Transaction.coinbase(
        (
            Output(
                amount_matoms=seed,
                template_id=TEMPLATE_PKH,
                payload=bytes([seed]) * HASH_LEN_BYTES,
            ),
        )
    )


def _node(data_dir: Path) -> TensorPowNode:
    genesis = _genesis_anchor()
    return TensorPowNode(
        TensorPowConfig(
            data_dir=data_dir,
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            expected_genesis_hash=genesis.block_hash(),
        ),
        pow_verifier=lambda _header, _target, _backend: True,
    )


def _fruit(
    transactions: tuple[bytes, ...],
    *,
    latest_anchor: bytes = bytes(HASH_LEN_BYTES),
    parent_selected: bytes = bytes(HASH_LEN_BYTES),
    timestamp_ms: int = 1,
    nonce: int = 2,
) -> Fruit:
    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=parent_selected,
        parent_bitmap=b"",
        latest_anchor=latest_anchor,
        tx_merkle_root=tx_merkle_root(transactions),
        timestamp_ms=timestamp_ms,
        shard_id=ROOT_SHARD_ID,
        nonce=nonce,
    )
    return Fruit(header=header, transactions=transactions)


def _anchor(
    covered_fruit_hash: bytes,
    *,
    parent_anchor: bytes,
    fee_floor_matoms_per_kb: int = 5,
    anchor_reward_outputs: tuple[Output, ...] = (),
    timestamp_ms: int = 2,
) -> Anchor:
    return _anchor_many(
        (covered_fruit_hash,),
        parent_anchor=parent_anchor,
        parent_candidate_hashes=(covered_fruit_hash,),
        fee_floor_matoms_per_kb=fee_floor_matoms_per_kb,
        anchor_reward_outputs=anchor_reward_outputs,
        timestamp_ms=timestamp_ms,
    )


def _anchor_many(
    covered_fruit_hashes: tuple[bytes, ...],
    *,
    parent_anchor: bytes,
    parent_candidate_hashes: tuple[bytes, ...],
    fee_floor_matoms_per_kb: int = 5,
    anchor_reward_outputs: tuple[Output, ...] = (),
    timestamp_ms: int = 2,
) -> Anchor:
    tree = ShardTree()
    fee_entries = (FeeFloorEntry(ROOT_SHARD_ID, fee_floor_matoms_per_kb),)
    covered = tuple(sorted(covered_fruit_hashes))
    header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=parent_anchor,
        fruit_set_root=fruit_set_root(covered),
        parent_candidate_root=parent_candidate_root(parent_candidate_hashes),
        shard_tree_state_root=tree.state_root(),
        fee_floor_set_root=fee_floor_set_root(fee_entries),
        anchor_reward_root=anchor_reward_root(anchor_reward_outputs),
        timestamp_ms=timestamp_ms,
        nonce=3,
    )
    return Anchor(
        header=header,
        covered_fruit_hashes=covered,
        parent_candidate_hashes=parent_candidate_hashes,
        shard_tree_bytes=tree.serialize(),
        fee_floor_entries=fee_entries,
        anchor_reward_outputs=anchor_reward_outputs,
    )


def _genesis_anchor() -> Anchor:
    return build_genesis_artifact(
        GenesisInputs.create(
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            whitepaper_hash=bytes.fromhex("aa" * 32),
            bitcoin_block_hash=bytes.fromhex("bb" * 32),
            ethereum_block_hash=bytes.fromhex("cc" * 32),
            founder_pubkey_hash=bytes.fromhex("dd" * 32),
        )
    ).anchor
