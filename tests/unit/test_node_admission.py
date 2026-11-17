"""M-1: PoW verification is rate limited per peer, banned on failure and off the state lock."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tensorpow.chain.blocks import Fruit, tx_merkle_root
from tensorpow.chain.headers import FruitHeader
from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.genesis import GENESIS_CHAIN_ID_TESTNET, GenesisInputs, build_genesis_artifact
from tensorpow.mempool import ROOT_SHARD_ID
from tensorpow.node import TensorPowConfig, TensorPowNode
from tensorpow.node.admission import (
    REASON_PEER_BANNED,
    REASON_PEER_RATE_LIMITED,
    REASON_POW_VERIFIER_BUSY,
    AdmissionConfig,
    PeerAdmission,
)
from tensorpow.pow.challenge import FORMAT_EPOCH, PowHeader
from tensorpow.pow.kernel import Backend
from tensorpow.state.utxo import TEMPLATE_PKH
from tensorpow.tx.transaction import Output, Transaction


class _CountingVerifier:
    def __init__(self, *, result: bool) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, _header: PowHeader, _target: bytes, _backend: Backend) -> bool:
        self.calls += 1
        return self.result


def test_invalid_pow_bans_peer_before_any_further_verification(tmp_path: Path) -> None:
    verifier = _CountingVerifier(result=True)
    genesis = _genesis()
    node = _node(tmp_path, verifier)
    try:
        assert node.process_anchor(genesis)
        verifier.result = False
        junk = _fruit(1, genesis.block_hash())
        assert node.process_fruit(junk, peer_id="attacker").reason == "fruit_pow_invalid"
        calls_after_first = verifier.calls

        for nonce in range(2, 50):
            result = node.process_fruit_bytes(
                _fruit(nonce, genesis.block_hash()).serialize(), peer_id="attacker"
            )
            assert result.reason == REASON_PEER_BANNED
        assert verifier.calls == calls_after_first

        verifier.result = True
        assert node.process_fruit(_fruit(99, genesis.block_hash()), peer_id="honest").accepted
    finally:
        node.close()


def test_per_peer_pow_budget_refills_over_time() -> None:
    clock = _Clock()
    admission = PeerAdmission(
        AdmissionConfig(pow_verify_rate_per_second=1.0, pow_verify_burst=3),
        clock=clock,
    )
    reasons = [admission.consume_pow_budget("flooder") for _ in range(10)]

    assert reasons.count(None) == 3
    assert reasons.count(REASON_PEER_RATE_LIMITED) == 7
    assert admission.consume_pow_budget("someone-else") is None
    assert admission.consume_pow_budget(None) is None
    clock.now += 2.0
    assert admission.consume_pow_budget("flooder") is None
    assert admission.consume_pow_budget("flooder") is None
    assert admission.consume_pow_budget("flooder") == REASON_PEER_RATE_LIMITED


def test_rate_limited_peer_does_not_reach_the_verifier(tmp_path: Path) -> None:
    verifier = _CountingVerifier(result=True)
    genesis = _genesis()
    node = _node(
        tmp_path,
        verifier,
        admission=AdmissionConfig(pow_verify_rate_per_second=0.001, pow_verify_burst=1),
    )
    try:
        assert node.process_anchor(genesis)
        first = node.process_fruit(_fruit(1, genesis.block_hash()), peer_id="flooder")
        assert first.accepted
        calls = verifier.calls
        # A second genesis-parent fruit fails a cheap check; build siblings of the first.
        second = _fruit(2, genesis.block_hash(), parent=first.object_hash, timestamp_ms=2)
        assert node.process_fruit(second, peer_id="flooder").reason == REASON_PEER_RATE_LIMITED
        assert verifier.calls == calls
        assert node.process_fruit(second, peer_id="other").accepted
    finally:
        node.close()


def test_pow_verification_runs_without_holding_the_state_lock(tmp_path: Path) -> None:
    genesis = _genesis()
    observed: list[bool] = []
    holder: dict[str, TensorPowNode] = {}

    def verifier(_header: PowHeader, _target: bytes, _backend: Backend) -> bool:
        lock_free = threading.Event()

        def probe() -> None:
            lock = holder["node"]._state_lock
            if lock.acquire(blocking=False):
                lock.release()
                lock_free.set()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        observed.append(lock_free.is_set())
        return True

    node = _node(tmp_path, verifier)
    holder["node"] = node
    try:
        assert node.process_anchor(genesis)
        assert node.process_fruit(_fruit(1, genesis.block_hash()), peer_id="peer").accepted
        assert observed == [True]
    finally:
        node.close()


def test_remote_pow_requests_are_shed_when_all_slots_are_busy(tmp_path: Path) -> None:
    genesis = _genesis()
    release = threading.Event()
    entered = threading.Event()

    def slow_verifier(_header: PowHeader, _target: bytes, _backend: Backend) -> bool:
        entered.set()
        release.wait(timeout=10)
        return True

    node = _node(
        tmp_path,
        slow_verifier,
        admission=AdmissionConfig(max_concurrent_pow_verifications=1),
    )
    try:
        assert node.process_anchor(genesis)
        first_fruit = _fruit(1, genesis.block_hash())
        results: list[object] = []
        worker = threading.Thread(
            target=lambda: results.append(node.process_fruit(first_fruit, peer_id="a"))
        )
        worker.start()
        assert entered.wait(timeout=10)
        busy = node.process_fruit(_fruit(2, genesis.block_hash()), peer_id="b")
        assert busy.reason == REASON_POW_VERIFIER_BUSY
        release.set()
        worker.join(timeout=10)
        assert results and getattr(results[0], "accepted", False)
    finally:
        release.set()
        node.close()


def test_malformed_payload_strikes_eventually_ban() -> None:
    admission = PeerAdmission(AdmissionConfig(malformed_strike_limit=3))
    admission.strike("p")
    admission.strike("p")
    assert not admission.is_banned("p")
    admission.strike("p")
    assert admission.is_banned("p")


def test_admission_peer_table_is_bounded_and_keeps_bans() -> None:
    admission = PeerAdmission(AdmissionConfig(max_tracked_peers=4))
    admission.ban("banned")
    for index in range(50):
        admission.consume_pow_budget(f"peer-{index}")
    assert admission.is_banned("banned")
    assert len(admission._peers) <= 4


def test_admission_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        AdmissionConfig(pow_verify_burst=0)
    with pytest.raises(ValueError):
        AdmissionConfig(pow_verify_rate_per_second=0)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _node(
    tmp_path: Path,
    verifier: object,
    *,
    admission: AdmissionConfig | None = None,
) -> TensorPowNode:
    return TensorPowNode(
        TensorPowConfig(
            data_dir=tmp_path / "node",
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            expected_genesis_hash=_genesis().block_hash(),
        ),
        pow_verifier=verifier,  # type: ignore[arg-type]
        admission=admission,
    )


def _fruit(
    seed: int,
    latest_anchor: bytes,
    *,
    parent: bytes | None = None,
    timestamp_ms: int = 1,
) -> Fruit:
    coinbase = Transaction.coinbase(
        (Output(seed, TEMPLATE_PKH, payload=bytes([seed % 256]) * HASH_LEN_BYTES),)
    )
    transactions = (coinbase.to_bytes(),)
    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=bytes(HASH_LEN_BYTES) if parent is None else parent,
        parent_bitmap=b"",
        latest_anchor=latest_anchor,
        tx_merkle_root=tx_merkle_root(transactions),
        timestamp_ms=timestamp_ms,
        shard_id=ROOT_SHARD_ID,
        nonce=seed,
    )
    return Fruit(header=header, transactions=transactions)


def _genesis():  # type: ignore[no-untyped-def]
    return build_genesis_artifact(
        GenesisInputs.create(
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            whitepaper_hash=bytes.fromhex("aa" * 32),
            bitcoin_block_hash=bytes.fromhex("bb" * 32),
            ethereum_block_hash=bytes.fromhex("cc" * 32),
            founder_pubkey_hash=bytes.fromhex("dd" * 32),
        )
    ).anchor
