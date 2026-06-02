"""Genesis ceremony and public testnet helper integration tests."""

from __future__ import annotations

import http.client
import json
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from threading import Event, Thread

import pytest
import scripts.testnet_ops as testnet_ops
from scripts.genesis_ceremony import main as genesis_main
from scripts.launch_gates import (
    DAY_MS,
    FIRST_FRUIT_MAX_DELAY_MS,
    PUBLIC_TESTNET_MIN_MONITORS,
    PUBLICATION_MIN_ATTESTERS,
    REQUIRED_ATTACK_SCENARIOS,
    SIGNATURE_DOMAIN_ATTACK_REPORT,
    SIGNATURE_DOMAIN_PUBLICATION,
    SIGNATURE_DOMAIN_SOURCE_SELECTION,
    SIGNATURE_DOMAIN_TESTNET_LOG,
    SIGNATURE_DOMAIN_TESTNET_OBSERVATION,
    LaunchGateError,
    validate_genesis_publication_evidence,
    validate_public_testnet_evidence,
)
from scripts.launch_gates import (
    main as run_launch_gates,
)
from scripts.testnet_ops import (
    PUBLIC_TESTNET_MIN_DAYS,
    PUBLIC_TESTNET_MIN_NODES,
    FaucetState,
    build_faucet_transaction,
    create_testnet_http_server,
)
from scripts.testnet_ops import (
    main as run_testnet_ops,
)

from tensorpow.chain.blocks import Fruit, tx_merkle_root
from tensorpow.chain.headers import FruitHeader
from tensorpow.crypto.hash import hash_bytes
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT, sign
from tensorpow.genesis import (
    GENESIS_CHAIN_ID_MAINNET,
    GENESIS_CHAIN_ID_TESTNET,
    GenesisError,
    GenesisInputs,
    artifact_from_json,
    build_genesis_artifact,
    founder_address,
    founder_pubkey_hash,
)
from tensorpow.launch_policy import (
    GENESIS_BITCOIN_SELECTION_RULE,
    GENESIS_BTC_CONFIRMATIONS,
    GENESIS_BTC_MIN_CONFIRMATIONS,
    GENESIS_BTC_SELECTION_RULE,
    GENESIS_ETH_SELECTION_RULE,
    GENESIS_ETHEREUM_SELECTION_RULE,
)
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH
from tensorpow.pow.kernel import FRUIT_TARGET_LE
from tensorpow.pow.miner import mine
from tensorpow.pow.verify import verify_pow
from tensorpow.rpc.server import InMemoryRpcBackend, JsonRpcServer, create_http_server
from tensorpow.state import TEMPLATE_PKH, UTXO, Outpoint, UTXOSet
from tensorpow.tx import Output, Transaction
from tensorpow.wallet import Wallet, load_utxos_json, utxos_to_json

WHITEPAPER_HASH = (
    Path("docs/whitepaper/tensorpow.pdf.blake3").read_text(encoding="utf-8").strip().split()[-1]
)
TESTNET_START_MS = 1_700_000_000_000
CEREMONY_START_MS = 1_709_999_999_000
BITCOIN_OBSERVED_AT_MS = CEREMONY_START_MS - 2_000
BITCOIN_CONFIRMATION_TIP_OBSERVED_AT_MS = CEREMONY_START_MS - 500
ETHEREUM_FINALIZED_AT_MS = CEREMONY_START_MS - 1_000
PUBLICATION_PUBLISHED_AT_MS = 1_710_000_000_000


def test_genesis_artifact_is_deterministic_and_chain_separated() -> None:
    common = {
        "whitepaper_hash": bytes.fromhex("11" * 32),
        "bitcoin_block_hash": bytes.fromhex("22" * 32),
        "ethereum_block_hash": bytes.fromhex("33" * 32),
        "founder_pubkey_hash": bytes.fromhex("44" * 32),
    }
    mainnet = build_genesis_artifact(
        GenesisInputs.create(chain_id=GENESIS_CHAIN_ID_MAINNET, **common)
    )
    testnet = build_genesis_artifact(
        GenesisInputs.create(chain_id=GENESIS_CHAIN_ID_TESTNET, **common)
    )

    assert mainnet.inputs.commitment() != testnet.inputs.commitment()
    assert mainnet.block_hash != testnet.block_hash
    assert mainnet.header_hash != testnet.header_hash
    assert artifact_from_json(mainnet.to_json()).to_json() == mainnet.to_json()

    bad_header_hash = mainnet.to_json()
    bad_header_hash["anchor_header_hash"] = "99" * 32
    with pytest.raises(GenesisError, match="anchor_header_hash"):
        artifact_from_json(bad_header_hash)


def test_genesis_inputs_require_canonical_empty_state_roots() -> None:
    inputs = GenesisInputs.create(
        chain_id=GENESIS_CHAIN_ID_TESTNET,
        whitepaper_hash=bytes.fromhex("11" * 32),
        bitcoin_block_hash=bytes.fromhex("22" * 32),
        ethereum_block_hash=bytes.fromhex("33" * 32),
        founder_pubkey_hash=bytes.fromhex("44" * 32),
    )

    with pytest.raises(GenesisError, match="empty_utxo_root"):
        replace(inputs, empty_utxo_root=bytes.fromhex("55" * 32))
    with pytest.raises(GenesisError, match="initial_shard_tree_root"):
        replace(inputs, initial_shard_tree_root=bytes.fromhex("66" * 32))
    with pytest.raises(GenesisError, match="initial_fee_floor_root"):
        replace(inputs, initial_fee_floor_root=bytes.fromhex("77" * 32))


def test_launch_policy_exports_spec_names_and_founder_key_length() -> None:
    public_key = bytes.fromhex("44" * 32)

    assert GENESIS_BTC_CONFIRMATIONS == GENESIS_BTC_MIN_CONFIRMATIONS
    assert GENESIS_BTC_SELECTION_RULE == GENESIS_BITCOIN_SELECTION_RULE
    assert GENESIS_ETH_SELECTION_RULE == GENESIS_ETHEREUM_SELECTION_RULE
    assert founder_pubkey_hash(public_key).hex() == (
        "b1323aa4cbe6cee3ba3722ac18e26cc9f615d771ba412031fd8564e63efed7c1"
    )
    with pytest.raises(GenesisError, match="32 bytes"):
        founder_pubkey_hash(public_key[:-1])


def test_genesis_ceremony_cli_records_public_founder_material(tmp_path: Path) -> None:
    out = tmp_path / "genesis.json"
    assert (
        genesis_main(
            [
                "--whitepaper-hash",
                WHITEPAPER_HASH,
                *_selection_args(),
                "--bitcoin-block-hash",
                "22" * 32,
                "--ethereum-block-hash",
                "33" * 32,
                "--founder-public-key-hex",
                "44" * 32,
                "--out",
                str(out),
            ]
        )
        == 0
    )
    document = json.loads(out.read_text(encoding="utf-8"))

    assert document["inputs"]["chain_id"] == GENESIS_CHAIN_ID_MAINNET
    assert document["inputs"]["whitepaper_hash"] == WHITEPAPER_HASH
    assert document["ceremony_start_ms"] == CEREMONY_START_MS
    assert document["selection_evidence"]["ceremony_start_ms"] == CEREMONY_START_MS
    assert document["selection_evidence"]["bitcoin"]["confirmations"] == (
        GENESIS_BTC_MIN_CONFIRMATIONS
    )
    assert document["selection_evidence"]["bitcoin"]["confirmation_tip_height"] == 840_005
    assert document["selection_evidence"]["bitcoin"]["source_content_blake3"] == "12" * 32
    assert document["selection_evidence"]["ethereum"]["finalized_head_number"] == 19_000_000
    assert document["selection_evidence"]["ethereum"]["source_content_blake3"] == "13" * 32
    assert document["selection_evidence"]["bitcoin"]["source_url"].startswith("https://")
    assert document["founder"]["public_key"] == "44" * 32
    assert artifact_from_json(document).block_hash.hex() == document["anchor_hash"]


def test_genesis_ceremony_cli_uses_persistent_founder_keystore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TENSORPOW_FOUNDER_KEYSTORE_PASSWORD", "ceremony-secret")
    keystore = tmp_path / "founder.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = [
        "--whitepaper-hash",
        WHITEPAPER_HASH,
        *_selection_args(),
        "--bitcoin-block-hash",
        "22" * 32,
        "--ethereum-block-hash",
        "33" * 32,
        "--founder-keystore",
        str(keystore),
    ]

    assert genesis_main([*args, "--out", str(first)]) == 0
    assert genesis_main([*args, "--out", str(second)]) == 0

    first_doc = json.loads(first.read_text(encoding="utf-8"))
    second_doc = json.loads(second.read_text(encoding="utf-8"))
    assert first_doc["founder"] == second_doc["founder"]
    assert first_doc["anchor_hash"] == second_doc["anchor_hash"]


def test_genesis_and_testnet_tools_reject_noncanonical_hex(tmp_path: Path) -> None:
    artifact = _genesis_artifact(GENESIS_CHAIN_ID_MAINNET)
    uppercase_input = artifact.to_json()
    uppercase_input["inputs"] = {
        **uppercase_input["inputs"],
        "whitepaper_hash": "aa" * 32,
    }
    uppercase_input["inputs"]["whitepaper_hash"] = str(
        uppercase_input["inputs"]["whitepaper_hash"]
    ).upper()
    with pytest.raises(GenesisError, match="canonical"):
        artifact_from_json(uppercase_input)

    spaced_anchor = artifact.to_json()
    anchor_hex = str(spaced_anchor["anchor_hex"])
    spaced_anchor["anchor_hex"] = f"{anchor_hex[:2]} {anchor_hex[2:]}"
    with pytest.raises(GenesisError, match="canonical"):
        artifact_from_json(spaced_anchor)

    out = tmp_path / "genesis.json"
    assert (
        genesis_main(
            [
                "--whitepaper-hash",
                ("aa" * 32).upper(),
                *_selection_args(),
                "--bitcoin-block-hash",
                "22" * 32,
                "--ethereum-block-hash",
                "33" * 32,
                "--founder-public-key-hex",
                "44" * 32,
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert (
        run_testnet_ops(
            [
                "genesis",
                "--whitepaper-hash",
                "11" * 32,
                "--bitcoin-block-hash",
                f"{'22' * 16} {'22' * 16}",
                "--ethereum-block-hash",
                "33" * 32,
                "--founder-pubkey-hash",
                "44" * 32,
                "--out",
                str(out),
            ]
        )
        == 1
    )


def test_genesis_ceremony_cli_requires_source_selection_evidence(tmp_path: Path) -> None:
    base_args = [
        "--whitepaper-hash",
        WHITEPAPER_HASH,
        "--bitcoin-block-hash",
        "22" * 32,
        "--ethereum-block-hash",
        "33" * 32,
        "--founder-public-key-hex",
        "44" * 32,
        "--out",
        str(tmp_path / "genesis.json"),
    ]

    assert genesis_main([*_selection_args(bitcoin_confirmations=5), *base_args]) == 1
    assert genesis_main([*_selection_args(bitcoin_confirmations=7), *base_args]) == 1
    assert genesis_main([*_selection_args(bitcoin_tip_height=840_006), *base_args]) == 1
    assert (
        genesis_main(
            [
                *_selection_args(bitcoin_observed_at_ms=CEREMONY_START_MS + 1),
                *base_args,
            ]
        )
        == 1
    )
    assert (
        genesis_main(
            [
                *_selection_args(ethereum_finalized_head_number=19_000_001),
                *base_args,
            ]
        )
        == 1
    )
    assert (
        genesis_main(
            [
                *_selection_args(ethereum_finalized_head_hash="88" * 32),
                *base_args,
            ]
        )
        == 1
    )
    assert (
        genesis_main(
            [
                *_selection_args(ethereum_finalized_at_ms=CEREMONY_START_MS + 1),
                *base_args,
            ]
        )
        == 1
    )
    assert (
        genesis_main(
            [
                *_selection_args(bitcoin_source_url="https://localhost/block/123"),
                *base_args,
            ]
        )
        == 1
    )
    assert (
        genesis_main(
            [
                *_selection_args(bitcoin_source_url="https://blocks.invalid/block/123"),
                *base_args,
            ]
        )
        == 1
    )


def test_testnet_genesis_bootstrap_and_faucet_transaction(tmp_path: Path) -> None:
    genesis_out = tmp_path / "testnet-genesis.json"
    assert (
        run_testnet_ops(
            [
                "genesis",
                "--whitepaper-hash",
                "11" * 32,
                "--bitcoin-block-hash",
                "22" * 32,
                "--ethereum-block-hash",
                "33" * 32,
                "--founder-pubkey-hash",
                "44" * 32,
                "--out",
                str(genesis_out),
            ]
        )
        == 0
    )
    genesis = json.loads(genesis_out.read_text(encoding="utf-8"))
    assert genesis["inputs"]["chain_id"] == GENESIS_CHAIN_ID_TESTNET
    assert genesis["public_testnet_gate"] == {
        "minimum_days": PUBLIC_TESTNET_MIN_DAYS,
        "minimum_unique_nodes": PUBLIC_TESTNET_MIN_NODES,
        "status": "not_satisfied_until_observed",
    }

    bootstrap_out = tmp_path / "bootstrap.json"
    assert (
        run_testnet_ops(
            [
                "bootstrap-config",
                "--data-dir",
                str(tmp_path / "node"),
                "--genesis",
                str(genesis_out),
                "--public-host",
                "bootstrap.tensorpow.org",
                "--p2p-port",
                "19000",
                "--rpc-port",
                "19001",
                "--out",
                str(bootstrap_out),
            ]
        )
        == 0
    )
    bootstrap = json.loads(bootstrap_out.read_text(encoding="utf-8"))
    assert bootstrap["chain_id"] == GENESIS_CHAIN_ID_TESTNET
    assert bootstrap["testnet_genesis_hash"] == genesis["anchor_hash"]
    assert bootstrap["public_endpoints"]["p2p"] == "/dns4/bootstrap.tensorpow.org/tcp/19000"
    assert '[chain]\nchain_id = "tensorpow-testnet"' in bootstrap["config_toml"]
    assert f'genesis_hash = "{genesis["anchor_hash"]}"' in bootstrap["config_toml"]

    for host in ("localhost", "bad host", "node.invalid", "bootstrap.example.com"):
        assert (
            run_testnet_ops(
                [
                    "bootstrap-config",
                    "--data-dir",
                    str(tmp_path / f"node-{host.replace(' ', '-')}"),
                    "--genesis",
                    str(genesis_out),
                    "--public-host",
                    host,
                    "--out",
                    str(tmp_path / f"bad-bootstrap-{host.replace(' ', '-')}.json"),
                ]
            )
            == 1
        )

    wallet = Wallet.recover("55" * 32)
    recipient = Wallet.recover("66" * 32)
    wallet_path = wallet.save(tmp_path / "faucet.json", "pw")
    utxo = UTXO(
        outpoint=Outpoint(bytes.fromhex("77" * 32), 0),
        amount_matoms=1_000,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=wallet.pubkey_hash(),
        payload=wallet.pubkey_hash(),
    )
    utxo_path = tmp_path / "utxos.json"
    utxo_path.write_text(utxos_to_json((utxo,)), encoding="utf-8")
    loaded_wallet = Wallet.load(wallet_path, "pw")

    tx = build_faucet_transaction(
        wallet=loaded_wallet,
        utxos_path=utxo_path,
        recipient_address=recipient.address,
        amount_matoms=100,
        fee_matoms=1,
    )

    assert tx.outputs[0].payload == recipient.pubkey_hash()
    assert tx.tx_id().hex()


def test_testnet_explorer_http_proxies_rpc() -> None:
    rpc_backend = InMemoryRpcBackend()
    rpc_server = JsonRpcServer(rpc_backend)
    rpc_http = create_http_server(rpc_server=rpc_server, port=0)
    rpc_thread = Thread(target=rpc_http.serve_forever, daemon=True)
    rpc_thread.start()
    rpc_port = int(rpc_http.server_address[1])

    testnet_http = create_testnet_http_server(
        port=0,
        rpc_url=f"http://127.0.0.1:{rpc_port}/rpc",
    )
    testnet_thread = Thread(target=testnet_http.serve_forever, daemon=True)
    testnet_thread.start()
    testnet_port = int(testnet_http.server_address[1])

    try:
        health = _get_json(testnet_port, "/health")
        mempool = _get_json(testnet_port, "/explorer/mempool")
        block = _get_json(testnet_port, f"/explorer/block?hash={'aa' * 32}")
        tx = _get_json(testnet_port, f"/explorer/tx?txid={'bb' * 32}")
        finality = _get_json(testnet_port, f"/explorer/finality?hash={'cc' * 32}")

        assert health == {"chain_id": GENESIS_CHAIN_ID_TESTNET, "healthy": True}
        assert mempool["count"] == 0
        assert block == {"block_hash": "aa" * 32, "found": False, "raw": None}
        assert tx["txid"] == "bb" * 32
        assert tx["found"] is False
        assert finality["block_hash"] == "cc" * 32
        assert finality["seen"] is False
    finally:
        testnet_http.shutdown()
        testnet_http.server_close()
        testnet_thread.join(timeout=5)
        rpc_http.shutdown()
        rpc_http.server_close()
        rpc_thread.join(timeout=5)


def test_testnet_faucet_dispenses_signed_transactions(tmp_path: Path) -> None:
    wallet = Wallet.recover("57" * 32)
    recipient = Wallet.recover("68" * 32)
    wallet_path = wallet.save(tmp_path / "faucet.json", "pw")
    utxo = UTXO(
        outpoint=Outpoint(bytes.fromhex("79" * 32), 0),
        amount_matoms=1_000,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=wallet.pubkey_hash(),
        payload=wallet.pubkey_hash(),
    )
    utxo_path = tmp_path / "utxos.json"
    utxo_path.write_text(utxos_to_json((utxo,)) + "\n", encoding="utf-8")

    rpc_backend = InMemoryRpcBackend(utxo_set=UTXOSet((utxo,)))
    rpc_server = JsonRpcServer(rpc_backend)
    rpc_http = create_http_server(rpc_server=rpc_server, port=0)
    rpc_thread = Thread(target=rpc_http.serve_forever, daemon=True)
    rpc_thread.start()
    rpc_port = int(rpc_http.server_address[1])

    testnet_http = create_testnet_http_server(
        port=0,
        rpc_url=f"http://127.0.0.1:{rpc_port}/rpc",
        faucet_state=FaucetState(
            wallet=Wallet.load(wallet_path, "pw"),
            utxos_path=utxo_path,
            amount_matoms=100,
            fee_matoms=100,
        ),
    )
    testnet_thread = Thread(target=testnet_http.serve_forever, daemon=True)
    testnet_thread.start()
    testnet_port = int(testnet_http.server_address[1])

    try:
        status, payload = _post_raw(
            testnet_port,
            "/faucet",
            json.dumps({"address": recipient.address}).encode("utf-8"),
        )
        mempool = _get_json(testnet_port, "/explorer/mempool")

        assert status == HTTPStatus.OK
        assert payload["accepted"] is True
        assert payload["amount_matoms"] == 100
        assert payload["fee_matoms"] == 100
        assert isinstance(payload["sendrawtx"], dict)
        assert payload["sendrawtx"]["accepted"] is True
        assert payload["sendrawtx"]["txid"] == payload["txid"]
        assert mempool["count"] == 1
        assert load_utxos_json(utxo_path) == ()
    finally:
        testnet_http.shutdown()
        testnet_http.server_close()
        testnet_thread.join(timeout=5)
        rpc_http.shutdown()
        rpc_http.server_close()
        rpc_thread.join(timeout=5)


def test_testnet_faucet_rejects_malformed_http_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(testnet_ops, "MAX_TESTNET_HTTP_BODY_BYTES", 8)
    testnet_http = create_testnet_http_server(port=0)
    testnet_thread = Thread(target=testnet_http.serve_forever, daemon=True)
    testnet_thread.start()
    testnet_port = int(testnet_http.server_address[1])

    try:
        invalid_status, invalid_payload = _post_raw(testnet_port, "/faucet", b"{")
        array_status, array_payload = _post_raw(testnet_port, "/faucet", b"[]")
        large_status, large_payload = _post_raw(
            testnet_port,
            "/faucet",
            b'{"address":"too-large"}',
        )

        assert invalid_status == HTTPStatus.BAD_REQUEST
        assert invalid_payload == {"error": "invalid json"}
        assert array_status == HTTPStatus.BAD_REQUEST
        assert array_payload == {"error": "json object required"}
        assert large_status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert large_payload == {"error": "request body too large"}
    finally:
        testnet_http.shutdown()
        testnet_http.server_close()
        testnet_thread.join(timeout=5)


def test_public_testnet_launch_gate_requires_real_evidence(tmp_path: Path) -> None:
    artifact = _genesis_artifact(GENESIS_CHAIN_ID_TESTNET)
    evidence = _public_testnet_evidence(artifact.to_json())

    result = validate_public_testnet_evidence(
        evidence,
        genesis_document=artifact.to_json(),
    )
    assert result["satisfied"] is True
    assert result["duration_days"] == PUBLIC_TESTNET_MIN_DAYS
    assert result["unique_node_count"] == PUBLIC_TESTNET_MIN_NODES
    assert result["testnet_genesis_hash"] == artifact.block_hash.hex()
    assert result["bootstrap_count"] == 1
    assert result["monitor_public_key_count"] == PUBLIC_TESTNET_MIN_MONITORS
    assert result["monitor_log_count"] == PUBLIC_TESTNET_MIN_MONITORS

    genesis_path = tmp_path / "testnet-genesis.json"
    evidence_path = tmp_path / "testnet-evidence.json"
    genesis_path.write_text(json.dumps(artifact.to_json()), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert (
        run_launch_gates(
            [
                "testnet",
                "--genesis",
                str(genesis_path),
                "--evidence",
                str(evidence_path),
            ]
        )
        == 0
    )

    mainnet_artifact = _genesis_artifact(GENESIS_CHAIN_ID_MAINNET)
    with pytest.raises(LaunchGateError, match="testnet"):
        validate_public_testnet_evidence(
            evidence,
            genesis_document=mainnet_artifact.to_json(),
        )

    missing_split_field = dict(evidence)
    del missing_split_field["consensus_splits"]
    with pytest.raises(LaunchGateError, match="consensus_splits"):
        validate_public_testnet_evidence(
            missing_split_field,
            genesis_document=artifact.to_json(),
        )

    split_evidence = dict(evidence, consensus_splits=["height 144 consensus split"])
    with pytest.raises(LaunchGateError, match="consensus splits"):
        validate_public_testnet_evidence(
            split_evidence,
            genesis_document=artifact.to_json(),
        )

    short_evidence = dict(
        evidence,
        end_time_ms=evidence["start_time_ms"] + PUBLIC_TESTNET_MIN_DAYS * 86_400_000 - 1,
    )
    with pytest.raises(LaunchGateError, match="duration"):
        validate_public_testnet_evidence(
            short_evidence,
            genesis_document=artifact.to_json(),
        )

    attack_evidence = dict(
        evidence,
        attack_scenarios={
            **evidence["attack_scenarios"],
            "eclipse": {"attempted": True, "succeeded": True},
        },
    )
    with pytest.raises(LaunchGateError, match="eclipse"):
        validate_public_testnet_evidence(
            attack_evidence,
            genesis_document=artifact.to_json(),
        )

    private_attack = dict(
        evidence,
        attack_scenarios={
            **evidence["attack_scenarios"],
            "long-range": {"attempted": True, "succeeded": False, "evidence_urls": []},
        },
    )
    with pytest.raises(LaunchGateError, match="public evidence"):
        validate_public_testnet_evidence(
            private_attack,
            genesis_document=artifact.to_json(),
        )

    tampered_attack = dict(evidence)
    tampered_scenarios = dict(evidence["attack_scenarios"])
    tampered_scenarios["double-spend"] = {
        **tampered_scenarios["double-spend"],
        "report_blake3": "99" * 32,
    }
    tampered_attack["attack_scenarios"] = tampered_scenarios
    with pytest.raises(LaunchGateError, match="invalid signature"):
        validate_public_testnet_evidence(
            tampered_attack,
            genesis_document=artifact.to_json(),
        )

    bad_endpoint = dict(evidence, bootstrap_multiaddrs=["127.0.0.1:28333"])
    with pytest.raises(LaunchGateError, match="bootstrap_multiaddrs"):
        validate_public_testnet_evidence(
            bad_endpoint,
            genesis_document=artifact.to_json(),
        )

    private_bootstrap = dict(evidence, bootstrap_multiaddrs=["/ip4/127.0.0.1/tcp/28333"])
    with pytest.raises(LaunchGateError, match="public"):
        validate_public_testnet_evidence(
            private_bootstrap,
            genesis_document=artifact.to_json(),
        )

    malformed_dns_bootstrap = dict(
        evidence,
        bootstrap_multiaddrs=["/dns4/bad host/tcp/notaport"],
    )
    with pytest.raises(LaunchGateError, match=r"DNS name|port"):
        validate_public_testnet_evidence(
            malformed_dns_bootstrap,
            genesis_document=artifact.to_json(),
        )

    reserved_dns_bootstrap = dict(
        evidence,
        bootstrap_multiaddrs=["/dns4/node.invalid/tcp/28333"],
    )
    with pytest.raises(LaunchGateError, match="public DNS"):
        validate_public_testnet_evidence(
            reserved_dns_bootstrap,
            genesis_document=artifact.to_json(),
        )

    private_urls = dict(
        evidence,
        faucet_url="https://localhost/faucet",
        explorer_url="https://127.0.0.1/explorer",
    )
    with pytest.raises(LaunchGateError, match="public"):
        validate_public_testnet_evidence(
            private_urls,
            genesis_document=artifact.to_json(),
        )

    reserved_dns_urls = dict(
        evidence,
        faucet_url="https://faucet.invalid/faucet",
        explorer_url="https://explorer.example.com/explorer",
    )
    with pytest.raises(LaunchGateError, match="public DNS"):
        validate_public_testnet_evidence(
            reserved_dns_urls,
            genesis_document=artifact.to_json(),
        )

    future_evidence = dict(evidence, end_time_ms=9_999_999_999_999)
    with pytest.raises(LaunchGateError, match="future"):
        validate_public_testnet_evidence(
            future_evidence,
            genesis_document=artifact.to_json(),
        )

    bad_peer_ids = dict(evidence, unique_nodes=["peer-000", *evidence["unique_nodes"][1:]])
    with pytest.raises(LaunchGateError, match="PeerIds"):
        validate_public_testnet_evidence(
            bad_peer_ids,
            genesis_document=artifact.to_json(),
        )

    unsigned_nodes = dict(evidence)
    unsigned_nodes["node_observations"] = []
    with pytest.raises(LaunchGateError, match="node_observations"):
        validate_public_testnet_evidence(
            unsigned_nodes,
            genesis_document=artifact.to_json(),
        )

    monitor_split_count = dict(evidence)
    split_count_logs = list(evidence["monitor_logs"])
    assert isinstance(split_count_logs[0], dict)
    split_count_logs[0] = _signed_monitor_log(
        monitor_index=0,
        testnet_genesis_hash=artifact.block_hash.hex(),
        start_time_ms=int(evidence["start_time_ms"]),
        end_time_ms=int(evidence["end_time_ms"]),
        head_anchor_hash=str(split_count_logs[0]["head_anchor_hash"]),
        final_state_root=str(split_count_logs[0]["final_state_root"]),
        split_count=1,
    )
    monitor_split_count["monitor_logs"] = split_count_logs
    with pytest.raises(LaunchGateError, match="consensus splits"):
        validate_public_testnet_evidence(
            monitor_split_count,
            genesis_document=artifact.to_json(),
        )

    stale_checkpoints = dict(evidence)
    stale_checkpoint_logs = list(evidence["monitor_logs"])
    assert isinstance(stale_checkpoint_logs[0], dict)
    stale_checkpoint_logs[0] = _signed_monitor_log(
        monitor_index=0,
        testnet_genesis_hash=artifact.block_hash.hex(),
        start_time_ms=int(evidence["start_time_ms"]),
        end_time_ms=int(evidence["end_time_ms"]),
        head_anchor_hash=str(stale_checkpoint_logs[0]["head_anchor_hash"]),
        final_state_root=str(stale_checkpoint_logs[0]["final_state_root"]),
        max_checkpoint_gap_ms=DAY_MS + 1,
    )
    stale_checkpoints["monitor_logs"] = stale_checkpoint_logs
    with pytest.raises(LaunchGateError, match="checkpoint"):
        validate_public_testnet_evidence(
            stale_checkpoints,
            genesis_document=artifact.to_json(),
        )

    split_monitor_logs = dict(evidence)
    split_logs = list(evidence["monitor_logs"])
    assert isinstance(split_logs[0], dict)
    split_logs[0] = _signed_monitor_log(
        monitor_index=0,
        testnet_genesis_hash=artifact.block_hash.hex(),
        start_time_ms=int(evidence["start_time_ms"]),
        end_time_ms=int(evidence["end_time_ms"]),
        head_anchor_hash="99" * 32,
        final_state_root=str(split_logs[0]["final_state_root"]),
    )
    split_monitor_logs["monitor_logs"] = split_logs
    with pytest.raises(LaunchGateError, match="agree"):
        validate_public_testnet_evidence(
            split_monitor_logs,
            genesis_document=artifact.to_json(),
        )


def test_launch_gate_sign_evidence_cli_signs_monitor_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _genesis_artifact(GENESIS_CHAIN_ID_TESTNET)
    evidence = _public_testnet_evidence(artifact.to_json())
    payload = dict(evidence["node_observations"][0])
    payload.pop("monitor_public_key")
    payload.pop("signature")
    payload_path = tmp_path / "observation-payload.json"
    signed_path = tmp_path / "observation-signed.json"
    wallet_path = _monitor_wallet(2).save(tmp_path / "monitor.json", "pw")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("TENSORPOW_MONITOR_KEYSTORE_PASSWORD", "pw")

    assert (
        run_launch_gates(
            [
                "sign-evidence",
                "--domain",
                "testnet-observation",
                "--payload",
                str(payload_path),
                "--wallet",
                str(wallet_path),
                "--out",
                str(signed_path),
            ]
        )
        == 0
    )
    signed = json.loads(signed_path.read_text(encoding="utf-8"))
    node_observations = list(evidence["node_observations"])
    node_observations[0] = signed
    evidence["node_observations"] = node_observations

    assert signed["monitor_public_key"] == _monitor_wallet(2).public_key.hex()
    assert (
        validate_public_testnet_evidence(
            evidence,
            genesis_document=artifact.to_json(),
        )["satisfied"]
        is True
    )


def test_genesis_publication_launch_gate_requires_publication_and_first_fruit(
    tmp_path: Path,
) -> None:
    founder_public_key = bytes.fromhex("44" * 32)
    artifact = build_genesis_artifact(
        GenesisInputs.create(
            chain_id=GENESIS_CHAIN_ID_MAINNET,
            whitepaper_hash=bytes.fromhex("11" * 32),
            bitcoin_block_hash=bytes.fromhex("22" * 32),
            ethereum_block_hash=bytes.fromhex("33" * 32),
            founder_pubkey_hash=founder_pubkey_hash(founder_public_key),
        )
    )
    ceremony_document = _ceremony_document(artifact.to_json(), founder_public_key)
    testnet_artifact = _genesis_artifact(GENESIS_CHAIN_ID_TESTNET)
    testnet_evidence = _public_testnet_evidence(testnet_artifact.to_json())
    first_fruit = _first_fruit_for_anchor(
        artifact.header_hash,
        timestamp_ms=PUBLICATION_PUBLISHED_AT_MS + FIRST_FRUIT_MAX_DELAY_MS,
    )
    evidence = _genesis_publication_evidence(ceremony_document, first_fruit)

    result = validate_genesis_publication_evidence(
        ceremony_document=ceremony_document,
        evidence=evidence,
        testnet_genesis_document=testnet_artifact.to_json(),
        testnet_evidence=testnet_evidence,
    )
    assert result["satisfied"] is True
    assert result["anchor_hash"] == artifact.block_hash.hex()
    assert result["arweave_id"] == evidence["arweave_id"]
    assert result["bitcoin_block_height"] == 840_000
    assert result["ceremony_start_ms"] == CEREMONY_START_MS
    assert result["ethereum_block_number"] == 19_000_000
    assert result["first_fruit_hash"] == first_fruit.block_hash().hex()
    assert result["first_fruit_delay_ms"] == FIRST_FRUIT_MAX_DELAY_MS
    assert result["founder_pubkey_hash"] == artifact.inputs.founder_pubkey_hash.hex()
    assert result["ipfs_cid"] == evidence["ipfs_cid"]
    assert result["whitepaper_hash"] == artifact.inputs.whitepaper_hash.hex()

    ceremony_path = tmp_path / "mainnet-genesis.json"
    evidence_path = tmp_path / "publication-evidence.json"
    testnet_genesis_path = tmp_path / "testnet-genesis.json"
    testnet_evidence_path = tmp_path / "testnet-evidence.json"
    ceremony_path.write_text(
        json.dumps(ceremony_document, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    testnet_genesis_path.write_text(json.dumps(testnet_artifact.to_json()), encoding="utf-8")
    testnet_evidence_path.write_text(json.dumps(testnet_evidence), encoding="utf-8")
    assert (
        run_launch_gates(
            [
                "genesis-publication",
                "--ceremony",
                str(ceremony_path),
                "--evidence",
                str(evidence_path),
                "--testnet-genesis",
                str(testnet_genesis_path),
                "--testnet-evidence",
                str(testnet_evidence_path),
                "--expected-whitepaper-hash",
                artifact.inputs.whitepaper_hash.hex(),
            ]
        )
        == 0
    )

    bad_release = dict(evidence, github_release_title="release-1.2.3")
    with pytest.raises(LaunchGateError, match=r"vX\.Y\.Z"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=bad_release,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    bad_host = dict(evidence, github_release_url="https://evilgithub.com/releases/tag/v1.2.3")
    with pytest.raises(LaunchGateError, match=r"github\.com"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=bad_host,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    subdomain_host = dict(
        evidence,
        github_release_url="https://attacker.github.com/aravhawk/TensorPoW/releases/tag/v1.2.3",
    )
    with pytest.raises(LaunchGateError, match=r"github\.com"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=subdomain_host,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    wrong_repo = dict(
        evidence,
        github_release_url="https://github.com/aravhawk/Other/releases/tag/v1.2.3",
    )
    with pytest.raises(LaunchGateError, match="aravhawk/TensorPoW"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=wrong_repo,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    mismatched_release_tag = dict(
        evidence,
        github_release_url="https://github.com/aravhawk/TensorPoW/releases/tag/v1.2.4",
    )
    with pytest.raises(LaunchGateError, match="release tag"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=mismatched_release_tag,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    bad_ipfs = dict(evidence, ipfs_cid="not-a-cid")
    with pytest.raises(LaunchGateError, match="ipfs_cid"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=bad_ipfs,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    bad_arweave = dict(evidence, arweave_id="short")
    with pytest.raises(LaunchGateError, match="arweave_id"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=bad_arweave,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    missing_attestations = dict(evidence, publication_attestations=[])
    with pytest.raises(LaunchGateError, match="publication_attestations"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=missing_attestations,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    wrong_ceremony_bytes_hash = dict(evidence, ceremony_json_blake3="99" * 32)
    with pytest.raises(LaunchGateError, match="ceremony_json_blake3"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=wrong_ceremony_bytes_hash,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    with pytest.raises(LaunchGateError, match="whitepaper"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
            expected_whitepaper_hash="99" * 32,
        )

    github_release_url = str(evidence["github_release_url"])
    single_attester_target = dict(evidence)
    single_attester_target["publication_attestations"] = [
        attestation
        for attestation in evidence["publication_attestations"]
        if not (
            isinstance(attestation, dict)
            and attestation["target"] == github_release_url
            and attestation["monitor_public_key"] == _monitor_wallet(1).public_key.hex()
        )
    ]
    with pytest.raises(LaunchGateError, match="per target"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=single_attester_target,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    unsigned_publication_time = dict(evidence, published_at_ms=evidence["published_at_ms"] + 1)
    with pytest.raises(LaunchGateError, match="invalid signature"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=unsigned_publication_time,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    anchor_hash = str(evidence["published_anchor_hash"])
    spaced_anchor_hash = dict(
        evidence, published_anchor_hash=f"{anchor_hash[:2]} {anchor_hash[2:]}"
    )
    with pytest.raises(LaunchGateError, match="canonical"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=spaced_anchor_hash,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    bad_delay = dict(
        evidence,
        first_fruit_at_ms=evidence["published_at_ms"] + FIRST_FRUIT_MAX_DELAY_MS + 1,
    )
    with pytest.raises(LaunchGateError, match="one minute"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=bad_delay,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    timestamp_mismatch = dict(evidence, first_fruit_at_ms=evidence["first_fruit_at_ms"] - 1)
    with pytest.raises(LaunchGateError, match="timestamp"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=timestamp_mismatch,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    wrong_fruit = _first_fruit_for_anchor(bytes.fromhex("99" * 32))
    bad_first_fruit = dict(
        evidence,
        first_fruit_hex=wrong_fruit.serialize().hex(),
        first_fruit_hash=wrong_fruit.block_hash().hex(),
    )
    with pytest.raises(LaunchGateError, match="latest_anchor"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=bad_first_fruit,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    unmined_fruit = _unmined_first_fruit_for_anchor(artifact.header_hash)
    bad_pow = dict(
        evidence,
        first_fruit_hex=unmined_fruit.serialize().hex(),
        first_fruit_hash=unmined_fruit.block_hash().hex(),
    )
    with pytest.raises(LaunchGateError, match="PoW"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=bad_pow,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    noncanonical_first_fruit = dict(evidence, first_fruit_hex=evidence["first_fruit_hex"].upper())
    with pytest.raises(LaunchGateError, match="canonical"):
        validate_genesis_publication_evidence(
            ceremony_document=ceremony_document,
            evidence=noncanonical_first_fruit,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    missing_founder = dict(ceremony_document)
    del missing_founder["founder"]
    with pytest.raises(LaunchGateError, match="founder"):
        validate_genesis_publication_evidence(
            ceremony_document=missing_founder,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    wrong_founder = {
        **ceremony_document,
        "founder": {**ceremony_document["founder"], "public_key": "55" * 32},
    }
    with pytest.raises(LaunchGateError, match="founder public_key"):
        validate_genesis_publication_evidence(
            ceremony_document=wrong_founder,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    wrong_founder_address = {
        **ceremony_document,
        "founder": {
            **ceremony_document["founder"],
            "address": founder_address(bytes.fromhex("66" * 32)),
        },
    }
    with pytest.raises(LaunchGateError, match="founder address"):
        validate_genesis_publication_evidence(
            ceremony_document=wrong_founder_address,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    missing_selection = dict(ceremony_document)
    del missing_selection["selection_evidence"]
    with pytest.raises(LaunchGateError, match="selection_evidence"):
        validate_genesis_publication_evidence(
            ceremony_document=missing_selection,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    wrong_selection_hash = _with_nested(
        ceremony_document,
        "selection_evidence",
        "bitcoin",
        "block_hash",
        value="99" * 32,
    )
    with pytest.raises(LaunchGateError, match="bitcoin block_hash"):
        validate_genesis_publication_evidence(
            ceremony_document=wrong_selection_hash,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    stale_bitcoin_selection = _with_nested(
        ceremony_document,
        "selection_evidence",
        "bitcoin",
        "confirmation_tip_height",
        value=840_006,
    )
    with pytest.raises(LaunchGateError, match="bitcoin confirmation tip"):
        validate_genesis_publication_evidence(
            ceremony_document=stale_bitcoin_selection,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    insufficient_confirmations = _with_nested(
        ceremony_document,
        "selection_evidence",
        "bitcoin",
        "confirmations",
        value=GENESIS_BTC_MIN_CONFIRMATIONS - 1,
    )
    with pytest.raises(LaunchGateError, match="bitcoin confirmations"):
        validate_genesis_publication_evidence(
            ceremony_document=insufficient_confirmations,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    stale_ethereum_selection = _with_nested(
        ceremony_document,
        "selection_evidence",
        "ethereum",
        "finalized_head_number",
        value=19_000_001,
    )
    with pytest.raises(LaunchGateError, match="ethereum finalized head"):
        validate_genesis_publication_evidence(
            ceremony_document=stale_ethereum_selection,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    future_finalized = _with_nested(
        ceremony_document,
        "selection_evidence",
        "ethereum",
        "finalized_at_ms",
        value=CEREMONY_START_MS + 1,
    )
    with pytest.raises(LaunchGateError, match="ethereum finalized"):
        validate_genesis_publication_evidence(
            ceremony_document=future_finalized,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    private_selection_source = _with_nested(
        ceremony_document,
        "selection_evidence",
        "bitcoin",
        "source_url",
        value="https://127.0.0.1/block/840000",
    )
    with pytest.raises(LaunchGateError, match="public"):
        validate_genesis_publication_evidence(
            ceremony_document=private_selection_source,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    unsigned_source_selection = _with_nested(
        ceremony_document,
        "selection_evidence",
        "bitcoin",
        "source_content_blake3",
        value="99" * 32,
    )
    with pytest.raises(LaunchGateError, match="invalid signature"):
        validate_genesis_publication_evidence(
            ceremony_document=unsigned_source_selection,
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )

    with pytest.raises(LaunchGateError, match="mainnet"):
        validate_genesis_publication_evidence(
            ceremony_document=testnet_artifact.to_json(),
            evidence=evidence,
            testnet_genesis_document=testnet_artifact.to_json(),
            testnet_evidence=testnet_evidence,
        )


def _get_json(port: int, path: str) -> dict[str, object]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    assert response.status == 200
    parsed = json.loads(payload)
    assert isinstance(parsed, dict)
    return parsed


def _post_raw(port: int, path: str, body: bytes) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(
        "POST",
        path,
        body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    parsed = json.loads(payload)
    assert isinstance(parsed, dict)
    return response.status, parsed


def _monitor_wallet(index: int) -> Wallet:
    return Wallet.recover(f"{index + 1:02x}" * 32)


def _selection_args(
    *,
    bitcoin_confirmations: int = GENESIS_BTC_MIN_CONFIRMATIONS,
    bitcoin_tip_hash: str = "77" * 32,
    bitcoin_tip_height: int = 840_005,
    bitcoin_tip_observed_at_ms: int = BITCOIN_CONFIRMATION_TIP_OBSERVED_AT_MS,
    bitcoin_observed_at_ms: int = BITCOIN_OBSERVED_AT_MS,
    bitcoin_source_content_blake3: str = "12" * 32,
    bitcoin_source_url: str = "https://mempool.space/block/22",
    ethereum_finalized_head_hash: str = "33" * 32,
    ethereum_finalized_head_number: int = 19_000_000,
    ethereum_finalized_at_ms: int = ETHEREUM_FINALIZED_AT_MS,
    ethereum_source_content_blake3: str = "13" * 32,
    ethereum_source_url: str = "https://etherscan.io/block/33",
) -> list[str]:
    return [
        "--ceremony-start-ms",
        str(CEREMONY_START_MS),
        "--bitcoin-block-height",
        "840000",
        "--bitcoin-confirmations",
        str(bitcoin_confirmations),
        "--bitcoin-confirmation-tip-height",
        str(bitcoin_tip_height),
        "--bitcoin-confirmation-tip-hash",
        bitcoin_tip_hash,
        "--bitcoin-confirmation-tip-observed-at-ms",
        str(bitcoin_tip_observed_at_ms),
        "--bitcoin-observed-at-ms",
        str(bitcoin_observed_at_ms),
        "--bitcoin-source-content-blake3",
        bitcoin_source_content_blake3,
        "--bitcoin-source-url",
        bitcoin_source_url,
        "--ethereum-block-number",
        "19000000",
        "--ethereum-finalized-head-number",
        str(ethereum_finalized_head_number),
        "--ethereum-finalized-head-hash",
        ethereum_finalized_head_hash,
        "--ethereum-finalized-at-ms",
        str(ethereum_finalized_at_ms),
        "--ethereum-source-content-blake3",
        ethereum_source_content_blake3,
        "--ethereum-source-url",
        ethereum_source_url,
    ]


def _signed_node_observation(
    *,
    node_id: str,
    testnet_genesis_hash: str,
    start_time_ms: int,
    end_time_ms: int,
    monitor_index: int,
) -> dict[str, object]:
    payload = {
        "evidence_url": f"https://testnet.tensorpow.org/nodes/{node_id}.json",
        "first_seen_ms": start_time_ms,
        "last_seen_ms": end_time_ms,
        "node_id": node_id,
        "testnet_genesis_hash": testnet_genesis_hash,
        "uptime_bps": 10_000,
    }
    return _signed_payload(SIGNATURE_DOMAIN_TESTNET_OBSERVATION, payload, monitor_index)


def _signed_monitor_log(
    *,
    checkpoint_count: int = PUBLIC_TESTNET_MIN_DAYS,
    final_anchor_height: int = PUBLIC_TESTNET_MIN_DAYS,
    fruit_count: int = PUBLIC_TESTNET_MIN_DAYS * 10,
    monitor_index: int,
    testnet_genesis_hash: str,
    start_time_ms: int,
    end_time_ms: int,
    head_anchor_hash: str,
    final_state_root: str,
    max_checkpoint_gap_ms: int = DAY_MS,
    start_anchor_height: int = 0,
    start_head_timestamp_ms: int | None = None,
    final_head_timestamp_ms: int | None = None,
    split_count: int = 0,
) -> dict[str, object]:
    checkpoint_hashes = [
        hash_bytes(f"monitor-checkpoint-{index}".encode()).hex()
        for index in range(checkpoint_count)
    ]
    payload = {
        "anchor_count": final_anchor_height - start_anchor_height,
        "checkpoint_count": checkpoint_count,
        "checkpoint_hashes": checkpoint_hashes,
        "content_blake3": hash_bytes(f"monitor-log-{monitor_index}".encode()).hex(),
        "end_time_ms": end_time_ms,
        "final_anchor_height": final_anchor_height,
        "final_head_timestamp_ms": end_time_ms
        if final_head_timestamp_ms is None
        else final_head_timestamp_ms,
        "final_state_root": final_state_root,
        "fruit_count": fruit_count,
        "head_anchor_hash": head_anchor_hash,
        "max_checkpoint_gap_ms": max_checkpoint_gap_ms,
        "split_count": split_count,
        "start_anchor_height": start_anchor_height,
        "start_head_timestamp_ms": start_time_ms
        if start_head_timestamp_ms is None
        else start_head_timestamp_ms,
        "start_time_ms": start_time_ms,
        "testnet_genesis_hash": testnet_genesis_hash,
        "url": f"https://testnet.tensorpow.org/monitor/{monitor_index}.json",
    }
    return _signed_payload(SIGNATURE_DOMAIN_TESTNET_LOG, payload, monitor_index)


def _signed_source_observation(
    *,
    payload: dict[str, object],
    monitor_index: int,
) -> dict[str, object]:
    return _signed_payload(SIGNATURE_DOMAIN_SOURCE_SELECTION, payload, monitor_index)


def _signed_publication_attestation(
    *,
    target: str,
    ceremony_document: dict[str, object],
    first_fruit_at_ms: int,
    first_fruit_hash: str,
    monitor_index: int,
    published_at_ms: int,
) -> dict[str, object]:
    payload = {
        "anchor_hash": str(ceremony_document["anchor_hash"]),
        "content_blake3": _canonical_json_hash(ceremony_document),
        "evidence_url": f"https://tensorpow.org/genesis/attestations/{monitor_index}.json",
        "first_fruit_at_ms": first_fruit_at_ms,
        "first_fruit_hash": first_fruit_hash,
        "published_at_ms": published_at_ms,
        "target": target,
    }
    return _signed_payload(SIGNATURE_DOMAIN_PUBLICATION, payload, monitor_index)


def _signed_payload(
    domain: bytes,
    payload: dict[str, object],
    monitor_index: int,
) -> dict[str, object]:
    wallet = _monitor_wallet(monitor_index)
    message = domain + json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        **payload,
        "monitor_public_key": wallet.public_key.hex(),
        "signature": sign(message, wallet.private_key).hex(),
    }


def _canonical_json_hash(document: dict[str, object]) -> str:
    encoded = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hash_bytes(encoded).hex()


BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _peer_id(index: int) -> str:
    return (
        "12D3KooW"
        + ("1" * 42)
        + BASE58_ALPHABET[index // len(BASE58_ALPHABET)]
        + BASE58_ALPHABET[index % len(BASE58_ALPHABET)]
    )


def _signed_attack_report(
    *,
    end_time_ms: int,
    monitor_index: int,
    scenario: str,
    start_time_ms: int,
    testnet_genesis_hash: str,
) -> dict[str, object]:
    payload = {
        "attempted": True,
        "ended_at_ms": start_time_ms + 1_000 + monitor_index,
        "evidence_urls": [f"https://testnet.tensorpow.org/attacks/{scenario}.json"],
        "report_blake3": hash_bytes(f"attack-{scenario}".encode()).hex(),
        "scenario": scenario,
        "started_at_ms": start_time_ms + monitor_index,
        "succeeded": False,
        "testnet_genesis_hash": testnet_genesis_hash,
        "testnet_window_end_ms": end_time_ms,
        "testnet_window_start_ms": start_time_ms,
    }
    return _signed_payload(SIGNATURE_DOMAIN_ATTACK_REPORT, payload, monitor_index)


def _public_testnet_evidence(testnet_genesis: dict[str, object]) -> dict[str, object]:
    start_time_ms = TESTNET_START_MS
    end_time_ms = start_time_ms + PUBLIC_TESTNET_MIN_DAYS * 86_400_000
    testnet_genesis_hash = str(testnet_genesis["anchor_hash"])
    unique_nodes = [_peer_id(index) for index in range(PUBLIC_TESTNET_MIN_NODES)]
    head_anchor_hash = hash_bytes(b"testnet-final-head").hex()
    final_state_root = hash_bytes(b"testnet-final-state").hex()
    return {
        "testnet_genesis_hash": testnet_genesis["anchor_hash"],
        "bootstrap_multiaddrs": ["/dns4/bootstrap.tensorpow.org/tcp/28333"],
        "faucet_url": "https://testnet.tensorpow.org/faucet",
        "explorer_url": "https://testnet.tensorpow.org/explorer",
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "unique_nodes": unique_nodes,
        "node_observations": [
            _signed_node_observation(
                node_id=node_id,
                testnet_genesis_hash=testnet_genesis_hash,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                monitor_index=index % PUBLIC_TESTNET_MIN_MONITORS,
            )
            for index, node_id in enumerate(unique_nodes)
        ],
        "monitor_logs": [
            _signed_monitor_log(
                monitor_index=index,
                testnet_genesis_hash=testnet_genesis_hash,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                head_anchor_hash=head_anchor_hash,
                final_state_root=final_state_root,
            )
            for index in range(PUBLIC_TESTNET_MIN_MONITORS)
        ],
        "consensus_splits": [],
        "attack_scenarios": {
            scenario: _signed_attack_report(
                end_time_ms=end_time_ms,
                monitor_index=index % PUBLIC_TESTNET_MIN_MONITORS,
                scenario=scenario,
                start_time_ms=start_time_ms,
                testnet_genesis_hash=testnet_genesis_hash,
            )
            for index, scenario in enumerate(REQUIRED_ATTACK_SCENARIOS)
        },
    }


def _genesis_artifact(chain_id: str):
    return build_genesis_artifact(
        GenesisInputs.create(
            chain_id=chain_id,
            whitepaper_hash=bytes.fromhex("11" * 32),
            bitcoin_block_hash=bytes.fromhex("22" * 32),
            ethereum_block_hash=bytes.fromhex("33" * 32),
            founder_pubkey_hash=bytes.fromhex("44" * 32),
        )
    )


def _ceremony_document(
    genesis_document: dict[str, object],
    founder_public_key: bytes,
) -> dict[str, object]:
    document = dict(genesis_document)
    document["founder"] = {
        "address": founder_address(founder_public_key),
        "public_key": founder_public_key.hex(),
        "pubkey_hash": founder_pubkey_hash(founder_public_key).hex(),
    }
    inputs = document["inputs"]
    assert isinstance(inputs, dict)
    document["ceremony_start_ms"] = CEREMONY_START_MS
    document["selection_rules"] = {
        "bitcoin": GENESIS_BITCOIN_SELECTION_RULE,
        "ethereum": GENESIS_ETHEREUM_SELECTION_RULE,
    }
    bitcoin_source_payload: dict[str, object] = {
        "block_hash": inputs["bitcoin_block_hash"],
        "block_height": 840_000,
        "chain": "bitcoin",
        "confirmation_tip_hash": "77" * 32,
        "confirmation_tip_height": 840_005,
        "confirmation_tip_observed_at_ms": BITCOIN_CONFIRMATION_TIP_OBSERVED_AT_MS,
        "confirmations": GENESIS_BTC_MIN_CONFIRMATIONS,
        "observed_at_ms": BITCOIN_OBSERVED_AT_MS,
        "selection_rule": GENESIS_BITCOIN_SELECTION_RULE,
        "source_content_blake3": "12" * 32,
        "source_url": "https://mempool.space/block/22",
    }
    ethereum_source_payload: dict[str, object] = {
        "block_hash": inputs["ethereum_block_hash"],
        "block_number": 19_000_000,
        "chain": "ethereum",
        "finalized_at_ms": ETHEREUM_FINALIZED_AT_MS,
        "finalized_head_hash": inputs["ethereum_block_hash"],
        "finalized_head_number": 19_000_000,
        "selection_rule": GENESIS_ETHEREUM_SELECTION_RULE,
        "source_content_blake3": "13" * 32,
        "source_url": "https://etherscan.io/block/33",
    }
    document["selection_evidence"] = {
        "ceremony_start_ms": CEREMONY_START_MS,
        "bitcoin": {
            **bitcoin_source_payload,
            "source_observations": [
                _signed_source_observation(payload=bitcoin_source_payload, monitor_index=index)
                for index in range(PUBLICATION_MIN_ATTESTERS)
            ],
        },
        "ethereum": {
            **ethereum_source_payload,
            "source_observations": [
                _signed_source_observation(payload=ethereum_source_payload, monitor_index=index)
                for index in range(PUBLICATION_MIN_ATTESTERS)
            ],
        },
    }
    return document


def _with_nested(
    document: dict[str, object],
    first_key: str,
    second_key: str,
    third_key: str,
    *,
    value: object,
) -> dict[str, object]:
    outer = dict(document)
    first = dict(outer[first_key])
    second = dict(first[second_key])
    second[third_key] = value
    first[second_key] = second
    outer[first_key] = first
    return outer


def _genesis_publication_evidence(
    ceremony_document: dict[str, object],
    first_fruit: Fruit,
) -> dict[str, object]:
    published_at_ms = PUBLICATION_PUBLISHED_AT_MS
    github_release_url = "https://github.com/aravhawk/TensorPoW/releases/tag/v1.2.3"
    ipfs_cid = "bafybeigdyrztensorpowgenesisfixture"
    arweave_id = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg"
    mirror_urls = ["https://tensorpow.org/genesis/mainnet-genesis.json"]
    first_fruit_at_ms = published_at_ms + FIRST_FRUIT_MAX_DELAY_MS
    first_fruit_hash = first_fruit.block_hash().hex()
    ceremony_json_blake3 = _canonical_json_hash(ceremony_document)
    return {
        "published_anchor_hash": ceremony_document["anchor_hash"],
        "published_anchor_hex": ceremony_document["anchor_hex"],
        "ceremony_json_blake3": ceremony_json_blake3,
        "github_release_title": "v1.2.3",
        "github_release_url": github_release_url,
        "ipfs_cid": ipfs_cid,
        "arweave_id": arweave_id,
        "mirror_urls": mirror_urls,
        "publication_attestations": [
            _signed_publication_attestation(
                target=target,
                ceremony_document=ceremony_document,
                first_fruit_at_ms=first_fruit_at_ms,
                first_fruit_hash=first_fruit_hash,
                monitor_index=monitor_index,
                published_at_ms=published_at_ms,
            )
            for target in (github_release_url, ipfs_cid, arweave_id, *mirror_urls)
            for monitor_index in range(PUBLICATION_MIN_ATTESTERS)
        ],
        "published_at_ms": published_at_ms,
        "first_fruit_at_ms": first_fruit_at_ms,
        "first_fruit_hex": first_fruit.serialize().hex(),
        "first_fruit_hash": first_fruit_hash,
    }


def _first_fruit_for_anchor(
    latest_anchor: bytes,
    *,
    timestamp_ms: int = PUBLICATION_PUBLISHED_AT_MS + FIRST_FRUIT_MAX_DELAY_MS,
) -> Fruit:
    coinbase = Transaction.coinbase(
        (Output(amount_matoms=1, template_id=TEMPLATE_PKH, payload=bytes.fromhex("aa" * 32)),)
    )
    tx_bytes = coinbase.to_bytes()
    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=GENESIS_PARENT_HASH,
        parent_bitmap=b"",
        latest_anchor=latest_anchor,
        tx_merkle_root=tx_merkle_root((tx_bytes,)),
        timestamp_ms=timestamp_ms,
        shard_id=0,
        nonce=0,
    )
    result = mine(
        header.to_pow_header(()),
        FRUIT_TARGET_LE,
        Event(),
        backend="cpu",
        start_nonce=0,
        max_nonce=512,
    )
    assert result is not None
    header = FruitHeader(
        version=header.version,
        sig_type_supported=header.sig_type_supported,
        parent_selected=header.parent_selected,
        parent_bitmap=header.parent_bitmap,
        latest_anchor=header.latest_anchor,
        tx_merkle_root=header.tx_merkle_root,
        timestamp_ms=header.timestamp_ms,
        shard_id=header.shard_id,
        nonce=result.nonce,
    )
    return Fruit(header=header, transactions=(tx_bytes,))


def _unmined_first_fruit_for_anchor(latest_anchor: bytes) -> Fruit:
    fruit = _first_fruit_for_anchor(latest_anchor)
    for offset in range(1, 129):
        header = FruitHeader(
            version=fruit.header.version,
            sig_type_supported=fruit.header.sig_type_supported,
            parent_selected=fruit.header.parent_selected,
            parent_bitmap=fruit.header.parent_bitmap,
            latest_anchor=fruit.header.latest_anchor,
            tx_merkle_root=fruit.header.tx_merkle_root,
            timestamp_ms=fruit.header.timestamp_ms,
            shard_id=fruit.header.shard_id,
            nonce=fruit.header.nonce + offset,
        )
        if not verify_pow(header.to_pow_header(()), FRUIT_TARGET_LE, backend="cpu"):
            return Fruit(header=header, transactions=fruit.transactions)
    raise AssertionError("could not find an invalid first-fruit nonce")
