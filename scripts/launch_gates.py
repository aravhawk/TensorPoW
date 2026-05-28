#!/usr/bin/env python
"""Validate public testnet and genesis launch-gate evidence."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast
from urllib.parse import urlparse

from tensorpow.chain.blocks import BlockDecodeError, Fruit
from tensorpow.chain.headers import HeaderDecodeError
from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes
from tensorpow.crypto.signatures import (
    ED25519_PUBLIC_KEY_BYTES,
    ED25519_SIGNATURE_BYTES,
    sign,
    verify,
)
from tensorpow.genesis import (
    GENESIS_CHAIN_ID_MAINNET,
    GENESIS_CHAIN_ID_TESTNET,
    GenesisError,
    artifact_from_json,
    founder_address,
    founder_pubkey_hash,
)
from tensorpow.launch_policy import (
    GENESIS_BITCOIN_SELECTION_RULE,
    GENESIS_BTC_MIN_CONFIRMATIONS,
    GENESIS_ETHEREUM_SELECTION_RULE,
    PUBLIC_TESTNET_MIN_DAYS,
    PUBLIC_TESTNET_MIN_NODES,
)
from tensorpow.pow.challenge import GENESIS_PARENT_HASH
from tensorpow.pow.kernel import FRUIT_TARGET_LE
from tensorpow.pow.verify import verify_pow
from tensorpow.wallet import load_wallet

DAY_MS: Final[int] = 24 * 60 * 60 * 1000
FIRST_FRUIT_MAX_DELAY_MS: Final[int] = 60_000
PUBLIC_TESTNET_MIN_MONITORS: Final[int] = 3
PUBLICATION_MIN_ATTESTERS: Final[int] = 2
SOURCE_SELECTION_MIN_ATTESTERS: Final[int] = 2
PUBLIC_TESTNET_MIN_UPTIME_BPS: Final[int] = 9_500
PUBLIC_TESTNET_MIN_CHECKPOINTS: Final[int] = PUBLIC_TESTNET_MIN_DAYS
SIGNATURE_DOMAIN_TESTNET_OBSERVATION: Final[bytes] = b"TensorPoW:testnet-observation"
SIGNATURE_DOMAIN_TESTNET_LOG: Final[bytes] = b"TensorPoW:testnet-monitor-log"
SIGNATURE_DOMAIN_PUBLICATION: Final[bytes] = b"TensorPoW:genesis-publication"
SIGNATURE_DOMAIN_SOURCE_SELECTION: Final[bytes] = b"TensorPoW:genesis-source-selection"
SIGNATURE_DOMAIN_ATTACK_REPORT: Final[bytes] = b"TensorPoW:testnet-attack-report"
SIGNATURE_DOMAIN_NAMES: Final[dict[str, bytes]] = {
    "attack-report": SIGNATURE_DOMAIN_ATTACK_REPORT,
    "publication": SIGNATURE_DOMAIN_PUBLICATION,
    "source-selection": SIGNATURE_DOMAIN_SOURCE_SELECTION,
    "testnet-log": SIGNATURE_DOMAIN_TESTNET_LOG,
    "testnet-observation": SIGNATURE_DOMAIN_TESTNET_OBSERVATION,
}
REQUIRED_ATTACK_SCENARIOS: Final[tuple[str, ...]] = (
    "double-spend",
    "selfish-mining",
    "eclipse",
    "long-range",
    "spam-fee-floor",
    "shard-fork",
)
RELEASE_TITLE_RE: Final[re.Pattern[str]] = re.compile(r"^v\d+\.\d+\.\d+$")
CID_V0_RE: Final[re.Pattern[str]] = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")
CID_BASE32_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{9,127}$")
ARWEAVE_TX_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9A-Za-z_-]{43}$")
DNS_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
PEER_ID_RE: Final[re.Pattern[str]] = re.compile(r"^12D3KooW[1-9A-HJ-NP-Za-km-z]{44}$")


class LaunchGateError(ValueError):
    """Raised when launch-gate evidence is malformed or insufficient."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the launch-gate evidence validator."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = cast(dict[str, object], args.handler(args))
        print(json.dumps(payload, sort_keys=True))
    except (OSError, TypeError, ValueError, GenesisError, LaunchGateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="launch_gates")
    subparsers = parser.add_subparsers(dest="command", required=True)

    testnet = subparsers.add_parser("testnet")
    testnet.add_argument("--evidence", type=Path, required=True)
    testnet.add_argument("--genesis", type=Path, required=True)
    testnet.set_defaults(handler=_cmd_testnet)

    genesis = subparsers.add_parser("genesis-publication")
    genesis.add_argument("--ceremony", type=Path, required=True)
    genesis.add_argument("--evidence", type=Path, required=True)
    genesis.add_argument("--testnet-genesis", type=Path, required=True)
    genesis.add_argument("--testnet-evidence", type=Path, required=True)
    genesis.add_argument("--expected-whitepaper-hash", required=True)
    genesis.set_defaults(handler=_cmd_genesis_publication)

    sign_evidence = subparsers.add_parser("sign-evidence")
    sign_evidence.add_argument(
        "--domain",
        choices=tuple(sorted(SIGNATURE_DOMAIN_NAMES)),
        required=True,
    )
    sign_evidence.add_argument("--payload", type=Path, required=True)
    sign_evidence.add_argument("--wallet", type=Path, required=True)
    sign_evidence.add_argument(
        "--password-env",
        default="TENSORPOW_MONITOR_KEYSTORE_PASSWORD",
    )
    sign_evidence.add_argument("--out", type=Path)
    sign_evidence.set_defaults(handler=_cmd_sign_evidence)
    return parser


def _cmd_testnet(args: argparse.Namespace) -> dict[str, object]:
    return validate_public_testnet_evidence(
        _read_json_object(Path(args.evidence)),
        genesis_document=_read_json_object(Path(args.genesis)),
    )


def _cmd_genesis_publication(args: argparse.Namespace) -> dict[str, object]:
    ceremony_document, ceremony_json_blake3 = _read_json_object_with_blake3(Path(args.ceremony))
    return validate_genesis_publication_evidence(
        ceremony_document=ceremony_document,
        evidence=_read_json_object(Path(args.evidence)),
        testnet_genesis_document=_read_json_object(Path(args.testnet_genesis)),
        testnet_evidence=_read_json_object(Path(args.testnet_evidence)),
        ceremony_json_blake3=ceremony_json_blake3,
        expected_whitepaper_hash=args.expected_whitepaper_hash,
    )


def _cmd_sign_evidence(args: argparse.Namespace) -> dict[str, object]:
    payload = dict(_read_json_object(Path(args.payload)))
    payload.pop("monitor_public_key", None)
    payload.pop("signature", None)
    password = os.environ.get(args.password_env)
    if not password:
        raise LaunchGateError(f"{args.password_env} must be set")
    wallet = load_wallet(Path(args.wallet), password)
    signed_payload = {
        **payload,
        "monitor_public_key": wallet.public_key.hex(),
        "signature": sign(
            _signature_message(SIGNATURE_DOMAIN_NAMES[args.domain], payload),
            wallet.private_key,
        ).hex(),
    }
    if args.out is not None:
        _write_json_object(Path(args.out), signed_payload)
    return signed_payload


def validate_public_testnet_evidence(
    evidence: Mapping[str, object],
    *,
    genesis_document: Mapping[str, object],
    current_time_ms: int | None = None,
) -> dict[str, object]:
    """Validate that public testnet evidence satisfies the pre-mainnet gate."""

    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    artifact = artifact_from_json(genesis_document)
    if artifact.inputs.chain_id != GENESIS_CHAIN_ID_TESTNET:
        raise LaunchGateError("testnet genesis must use tensorpow-testnet chain_id")
    if _require_hash_hex(evidence, "testnet_genesis_hash") != artifact.block_hash.hex():
        raise LaunchGateError("testnet_genesis_hash does not match testnet genesis")

    bootstrap_multiaddrs = _require_unique_strings(evidence, "bootstrap_multiaddrs")
    if not bootstrap_multiaddrs:
        raise LaunchGateError("bootstrap_multiaddrs must contain at least one address")
    for multiaddr in bootstrap_multiaddrs:
        _require_bootstrap_multiaddr(multiaddr)
    _require_url(evidence, "faucet_url")
    _require_url(evidence, "explorer_url")

    start_time_ms = _require_nonnegative_int(evidence, "start_time_ms")
    end_time_ms = _require_nonnegative_int(evidence, "end_time_ms")
    if end_time_ms <= start_time_ms:
        raise LaunchGateError("end_time_ms must be after start_time_ms")
    duration_ms = end_time_ms - start_time_ms
    now_ms = (
        _current_time_ms()
        if current_time_ms is None
        else _require_nonnegative_int_value(
            "current_time_ms",
            current_time_ms,
        )
    )
    if end_time_ms > now_ms:
        raise LaunchGateError("public testnet end_time_ms is in the future")
    minimum_ms = PUBLIC_TESTNET_MIN_DAYS * DAY_MS
    if duration_ms < minimum_ms:
        raise LaunchGateError("public testnet duration is below minimum")

    unique_nodes = _require_unique_strings(evidence, "unique_nodes")
    for node_id in unique_nodes:
        _require_peer_id(node_id)
    if len(unique_nodes) < PUBLIC_TESTNET_MIN_NODES:
        raise LaunchGateError("public testnet unique node count is below minimum")
    observation_summary = _validate_node_observations(
        evidence,
        unique_nodes=unique_nodes,
        testnet_genesis_hash=artifact.block_hash.hex(),
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )
    monitor_log_count = _validate_monitor_logs(
        evidence,
        testnet_genesis_hash=artifact.block_hash.hex(),
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )

    consensus_splits = evidence.get("consensus_splits")
    if not isinstance(consensus_splits, Sequence) or isinstance(consensus_splits, str | bytes):
        raise LaunchGateError("consensus_splits must be a sequence")
    if consensus_splits:
        raise LaunchGateError("public testnet evidence records consensus splits")

    attacks = evidence.get("attack_scenarios")
    if not isinstance(attacks, Mapping):
        raise LaunchGateError("attack_scenarios must be an object")
    passed_attacks = []
    attack_monitor_public_keys: set[str] = set()
    for scenario in REQUIRED_ATTACK_SCENARIOS:
        attack = attacks.get(scenario)
        if not isinstance(attack, Mapping):
            raise LaunchGateError(f"missing attack scenario {scenario}")
        if attack.get("attempted") is not True:
            raise LaunchGateError(f"attack scenario {scenario} was not attempted")
        if attack.get("succeeded") is not False:
            raise LaunchGateError(f"attack scenario {scenario} succeeded")
        evidence_urls = _require_unique_strings(
            cast(Mapping[str, object], attack),
            "evidence_urls",
        )
        if not evidence_urls:
            raise LaunchGateError(f"attack scenario {scenario} has no public evidence URLs")
        for evidence_url in evidence_urls:
            _require_url_value(f"attack scenario {scenario} evidence_urls entry", evidence_url)
        attack_report = cast(Mapping[str, object], attack)
        report_hash = _require_hash_hex(attack_report, "report_blake3")
        started_at_ms = _require_nonnegative_int(attack_report, "started_at_ms")
        ended_at_ms = _require_nonnegative_int(attack_report, "ended_at_ms")
        if (
            started_at_ms < start_time_ms
            or ended_at_ms > end_time_ms
            or ended_at_ms <= started_at_ms
        ):
            raise LaunchGateError(f"attack scenario {scenario} timing is outside testnet window")
        payload = {
            "attempted": True,
            "ended_at_ms": ended_at_ms,
            "evidence_urls": list(evidence_urls),
            "report_blake3": report_hash,
            "scenario": scenario,
            "started_at_ms": started_at_ms,
            "succeeded": False,
            "testnet_genesis_hash": artifact.block_hash.hex(),
            "testnet_window_end_ms": end_time_ms,
            "testnet_window_start_ms": start_time_ms,
        }
        attack_monitor_public_keys.add(
            _require_signature(
                attack_report,
                domain=SIGNATURE_DOMAIN_ATTACK_REPORT,
                payload=payload,
            )
        )
        passed_attacks.append(scenario)
    if len(attack_monitor_public_keys) < PUBLIC_TESTNET_MIN_MONITORS:
        raise LaunchGateError("attack_scenarios must be signed by independent monitors")

    return {
        "attack_scenarios": passed_attacks,
        "bootstrap_count": len(bootstrap_multiaddrs),
        "duration_days": duration_ms // DAY_MS,
        "explorer_url": _require_str(evidence, "explorer_url"),
        "faucet_url": _require_str(evidence, "faucet_url"),
        "minimum_days": PUBLIC_TESTNET_MIN_DAYS,
        "minimum_monitors": PUBLIC_TESTNET_MIN_MONITORS,
        "minimum_unique_nodes": PUBLIC_TESTNET_MIN_NODES,
        "monitor_log_count": monitor_log_count,
        "monitor_public_key_count": observation_summary["monitor_public_key_count"],
        "satisfied": True,
        "testnet_genesis_hash": artifact.block_hash.hex(),
        "unique_node_count": len(unique_nodes),
    }


def validate_genesis_publication_evidence(
    *,
    ceremony_document: Mapping[str, object],
    evidence: Mapping[str, object],
    testnet_genesis_document: Mapping[str, object],
    testnet_evidence: Mapping[str, object],
    ceremony_json_blake3: str | None = None,
    expected_whitepaper_hash: str | None = None,
    current_time_ms: int | None = None,
) -> dict[str, object]:
    """Validate publication and first-fruit evidence for mainnet genesis."""

    now_ms = (
        _current_time_ms()
        if current_time_ms is None
        else _require_nonnegative_int_value(
            "current_time_ms",
            current_time_ms,
        )
    )
    testnet_result = validate_public_testnet_evidence(
        testnet_evidence,
        genesis_document=testnet_genesis_document,
        current_time_ms=now_ms,
    )
    artifact = artifact_from_json(ceremony_document)
    if artifact.inputs.chain_id != GENESIS_CHAIN_ID_MAINNET:
        raise LaunchGateError("mainnet genesis publication must use tensorpow-mainnet chain_id")
    if (
        expected_whitepaper_hash is not None
        and artifact.inputs.whitepaper_hash.hex()
        != _require_hash_hex_value("expected_whitepaper_hash", expected_whitepaper_hash)
    ):
        raise LaunchGateError("whitepaper hash does not match expected whitepaper hash")
    _require_founder_record(ceremony_document, artifact.inputs.founder_pubkey_hash)
    selection_summary = _validate_genesis_selection_evidence(
        ceremony_document,
        bitcoin_block_hash=artifact.inputs.bitcoin_block_hash.hex(),
        ethereum_block_hash=artifact.inputs.ethereum_block_hash.hex(),
    )
    published_anchor_hash = _require_hash_hex(evidence, "published_anchor_hash")
    if published_anchor_hash != artifact.block_hash.hex():
        raise LaunchGateError("published_anchor_hash does not match ceremony anchor")
    if _require_str(evidence, "published_anchor_hex") != artifact.anchor.serialize().hex():
        raise LaunchGateError("published_anchor_hex does not match ceremony anchor")

    release_title = _require_str(evidence, "github_release_title")
    if RELEASE_TITLE_RE.fullmatch(release_title) is None:
        raise LaunchGateError("github_release_title must use vX.Y.Z format")
    github_release_url = _require_url(evidence, "github_release_url")
    _require_github_release_url(github_release_url, release_title)
    ipfs_cid = _require_ipfs_cid(evidence, "ipfs_cid")
    arweave_id = _require_arweave_tx_id(evidence, "arweave_id")
    mirror_urls = _require_unique_strings(evidence, "mirror_urls")
    if not mirror_urls:
        raise LaunchGateError("mirror_urls must contain at least one mirror")
    for mirror_url in mirror_urls:
        _require_url_value("mirror_urls entry", mirror_url)

    published_at_ms = _require_nonnegative_int(evidence, "published_at_ms")
    first_fruit_at_ms = _require_nonnegative_int(evidence, "first_fruit_at_ms")
    if first_fruit_at_ms > now_ms or published_at_ms > now_ms:
        raise LaunchGateError("genesis publication timestamps must not be in the future")
    delay_ms = first_fruit_at_ms - published_at_ms
    if delay_ms < 0:
        raise LaunchGateError("first fruit timestamp precedes publication")
    if delay_ms > FIRST_FRUIT_MAX_DELAY_MS:
        raise LaunchGateError("first fruit was not mined within one minute")

    first_fruit = _decode_first_fruit(_require_str(evidence, "first_fruit_hex"))
    if first_fruit.header.latest_anchor != artifact.header_hash:
        raise LaunchGateError("first fruit latest_anchor does not reference genesis")
    if (
        first_fruit.header.parent_selected != GENESIS_PARENT_HASH
        or first_fruit.header.parent_bitmap
    ):
        raise LaunchGateError("first fruit must be the first genesis-parent fruit")
    if first_fruit.header.timestamp_ms != first_fruit_at_ms:
        raise LaunchGateError("first fruit timestamp does not match first_fruit_at_ms")
    if not verify_pow(first_fruit.header.to_pow_header(()), FRUIT_TARGET_LE, backend="cpu"):
        raise LaunchGateError("first fruit PoW does not satisfy FRUIT_TARGET_LE")
    first_fruit_hash = _require_hash_hex(evidence, "first_fruit_hash")
    if first_fruit.block_hash().hex() != first_fruit_hash:
        raise LaunchGateError("first_fruit_hash does not match first_fruit_hex")
    expected_ceremony_json_blake3 = (
        _canonical_json_hash(ceremony_document)
        if ceremony_json_blake3 is None
        else _require_hash_hex_value("ceremony_json_blake3", ceremony_json_blake3)
    )
    if _require_hash_hex(evidence, "ceremony_json_blake3") != expected_ceremony_json_blake3:
        raise LaunchGateError("ceremony_json_blake3 does not match ceremony bytes")
    publication_attester_count = _validate_publication_attestations(
        evidence,
        anchor_hash=artifact.block_hash.hex(),
        ceremony_json_blake3=expected_ceremony_json_blake3,
        first_fruit_at_ms=first_fruit_at_ms,
        first_fruit_hash=first_fruit_hash,
        publication_targets=(
            _require_str(evidence, "github_release_url"),
            ipfs_cid,
            arweave_id,
            *mirror_urls,
        ),
        published_at_ms=published_at_ms,
    )

    return {
        "anchor_hash": artifact.block_hash.hex(),
        "bitcoin_block_height": selection_summary["bitcoin_block_height"],
        "chain_id": artifact.inputs.chain_id,
        "ceremony_start_ms": selection_summary["ceremony_start_ms"],
        "ethereum_block_number": selection_summary["ethereum_block_number"],
        "first_fruit_delay_ms": delay_ms,
        "first_fruit_hash": first_fruit_hash,
        "founder_pubkey_hash": artifact.inputs.founder_pubkey_hash.hex(),
        "github_release_title": release_title,
        "arweave_id": arweave_id,
        "ipfs_cid": ipfs_cid,
        "mirror_count": len(mirror_urls),
        "publication_attester_count": publication_attester_count,
        "satisfied": True,
        "testnet_unique_node_count": testnet_result["unique_node_count"],
        "whitepaper_hash": artifact.inputs.whitepaper_hash.hex(),
    }


def _decode_first_fruit(first_fruit_hex: str) -> Fruit:
    _require_canonical_hex_string("first_fruit_hex", first_fruit_hex)
    try:
        return Fruit.deserialize(bytes.fromhex(first_fruit_hex))
    except (TypeError, ValueError, BlockDecodeError, HeaderDecodeError) as exc:
        raise LaunchGateError("first_fruit_hex is malformed") from exc


def _read_json_object(path: Path) -> Mapping[str, object]:
    with path.open("rb") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise LaunchGateError(f"{path} must contain a JSON object")
    return payload


def _read_json_object_with_blake3(path: Path) -> tuple[Mapping[str, object], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise LaunchGateError(f"{path} must contain a JSON object")
    return payload, hash_bytes(raw).hex()


def _write_json_object(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_hash_hex(document: Mapping[str, object], key: str) -> str:
    value = _require_str(document, key)
    return _require_hash_hex_value(key, value)


def _require_hash_hex_value(name: str, value: str) -> str:
    _require_canonical_hex_string(name, value)
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise LaunchGateError(f"{name} must be hex") from exc
    if len(decoded) != HASH_LEN_BYTES:
        raise LaunchGateError(f"{name} must decode to {HASH_LEN_BYTES} bytes")
    return value


def _require_nonnegative_int(document: Mapping[str, object], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LaunchGateError(f"{key} must be int")
    if value < 0:
        raise LaunchGateError(f"{key} must be nonnegative")
    return value


def _require_unique_strings(document: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise LaunchGateError(f"{key} must be a sequence")
    items = tuple(_require_nonempty_string_value(f"{key} entry", item) for item in value)
    if len(set(items)) != len(items):
        raise LaunchGateError(f"{key} entries must be unique")
    return items


def _require_url(
    document: Mapping[str, object],
    key: str,
    *,
    required_host_suffix: str | None = None,
) -> str:
    value = _require_str(document, key)
    _require_url_value(key, value, required_host_suffix=required_host_suffix)
    return value


def _require_url_value(
    name: str,
    value: str,
    *,
    required_host_suffix: str | None = None,
) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise LaunchGateError(f"{name} must be an https URL")
    _require_public_host(f"{name} host", parsed.hostname)
    if required_host_suffix is not None:
        hostname = parsed.hostname
        if hostname != required_host_suffix and not hostname.endswith(f".{required_host_suffix}"):
            raise LaunchGateError(f"{name} must be hosted under {required_host_suffix}")
    if parsed.path in ("", "/"):
        raise LaunchGateError(f"{name} must include a path")


def _require_github_release_url(value: str, release_title: str) -> None:
    parsed = urlparse(value)
    if parsed.hostname != "github.com":
        raise LaunchGateError("github_release_url must be hosted on github.com")
    expected_path = f"/aravhawk/TensorPoW/releases/tag/{release_title}"
    if parsed.path != expected_path:
        raise LaunchGateError("github_release_url must target aravhawk/TensorPoW release tag")


def _require_ipfs_cid(document: Mapping[str, object], key: str) -> str:
    value = _require_nonempty_str(document, key)
    if CID_V0_RE.fullmatch(value) is None and CID_BASE32_RE.fullmatch(value) is None:
        raise LaunchGateError(f"{key} must be a CIDv0 or base32 CID string")
    return value


def _require_arweave_tx_id(document: Mapping[str, object], key: str) -> str:
    value = _require_nonempty_str(document, key)
    if ARWEAVE_TX_ID_RE.fullmatch(value) is None:
        raise LaunchGateError(f"{key} must be a 43-character base64url transaction id")
    return value


def _require_bootstrap_multiaddr(value: str) -> None:
    parts = tuple(part for part in value.split("/") if part)
    if len(parts) < 2 or parts[0] not in ("ip4", "ip6", "dns4", "dns6", "dnsaddr"):
        raise LaunchGateError("bootstrap_multiaddrs entries must start with a public address")
    if parts[0] in ("ip4", "ip6"):
        try:
            address = ipaddress.ip_address(parts[1])
        except ValueError as exc:
            raise LaunchGateError("bootstrap_multiaddrs IP address is malformed") from exc
        if not address.is_global:
            raise LaunchGateError("bootstrap_multiaddrs IP address must be public")
    else:
        _require_public_dns_name("bootstrap_multiaddrs DNS name", parts[1])
    if "tcp" in parts:
        transport_index = parts.index("tcp")
        if transport_index + 1 >= len(parts):
            raise LaunchGateError("bootstrap_multiaddrs TCP transport must include a port")
        _require_port("bootstrap_multiaddrs TCP port", parts[transport_index + 1])
    elif "udp" in parts and any(part.startswith("quic") for part in parts):
        transport_index = parts.index("udp")
        if transport_index + 1 >= len(parts):
            raise LaunchGateError("bootstrap_multiaddrs QUIC transport must include a UDP port")
        _require_port("bootstrap_multiaddrs UDP port", parts[transport_index + 1])
    else:
        raise LaunchGateError("bootstrap_multiaddrs entries must include a transport")


def _require_public_host(name: str, host: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        _require_public_dns_name(name, host)
        return
    if not address.is_global:
        raise LaunchGateError(f"{name} must be public")


def _require_public_dns_name(name: str, host: str) -> None:
    if host == "" or host.strip() != host or host.lower() != host:
        raise LaunchGateError(f"{name} must be a canonical public DNS name")
    candidate = host.removeprefix("_dnsaddr.")
    if candidate.endswith("."):
        raise LaunchGateError(f"{name} must be a canonical public DNS name")
    if (
        candidate in ("localhost",)
        or candidate.endswith(".localhost")
        or candidate.endswith(".local")
    ):
        raise LaunchGateError(f"{name} must be public")
    if "." not in candidate:
        raise LaunchGateError(f"{name} must contain a public suffix")
    labels = candidate.rstrip(".").split(".")
    if any(DNS_LABEL_RE.fullmatch(label) is None for label in labels):
        raise LaunchGateError(f"{name} is malformed")
    if _is_reserved_dns_name(candidate):
        raise LaunchGateError(f"{name} must use a globally delegated public DNS name")


def _is_reserved_dns_name(host: str) -> bool:
    labels = tuple(host.split("."))
    reserved_suffixes = (
        ("localhost",),
        ("local",),
        ("test",),
        ("invalid",),
        ("example",),
        ("onion",),
        ("example", "com"),
        ("example", "net"),
        ("example", "org"),
    )
    return any(labels[-len(suffix) :] == suffix for suffix in reserved_suffixes)


def _require_peer_id(value: str) -> None:
    if PEER_ID_RE.fullmatch(value) is None:
        raise LaunchGateError("unique_nodes must contain canonical libp2p Ed25519 PeerIds")


def _require_port(name: str, value: str) -> None:
    if value == "" or not value.isascii() or not value.isdecimal():
        raise LaunchGateError(f"{name} must be numeric")
    port = int(value)
    if port < 1 or port > 65_535:
        raise LaunchGateError(f"{name} must be in range 1..65535")


def _validate_node_observations(
    evidence: Mapping[str, object],
    *,
    unique_nodes: tuple[str, ...],
    testnet_genesis_hash: str,
    start_time_ms: int,
    end_time_ms: int,
) -> dict[str, int]:
    observations = evidence.get("node_observations")
    if not isinstance(observations, Sequence) or isinstance(observations, str | bytes):
        raise LaunchGateError("node_observations must be a sequence")
    expected_nodes = set(unique_nodes)
    observed_nodes: set[str] = set()
    monitor_public_keys: set[str] = set()
    for item in observations:
        if not isinstance(item, Mapping):
            raise LaunchGateError("node_observations entries must be objects")
        observation = cast(Mapping[str, object], item)
        node_id = _require_nonempty_str(observation, "node_id")
        if node_id not in expected_nodes:
            raise LaunchGateError("node_observations contains unknown node_id")
        if node_id in observed_nodes:
            raise LaunchGateError("node_observations must contain each node once")
        first_seen_ms = _require_nonnegative_int(observation, "first_seen_ms")
        last_seen_ms = _require_nonnegative_int(observation, "last_seen_ms")
        uptime_bps = _require_bps(observation, "uptime_bps")
        evidence_url = _require_url(observation, "evidence_url")
        if first_seen_ms > start_time_ms or last_seen_ms < end_time_ms:
            raise LaunchGateError("node observation does not cover the full testnet window")
        if uptime_bps < PUBLIC_TESTNET_MIN_UPTIME_BPS:
            raise LaunchGateError("node observation uptime is below minimum")
        payload = {
            "evidence_url": evidence_url,
            "first_seen_ms": first_seen_ms,
            "last_seen_ms": last_seen_ms,
            "node_id": node_id,
            "testnet_genesis_hash": testnet_genesis_hash,
            "uptime_bps": uptime_bps,
        }
        monitor_public_keys.add(
            _require_signature(
                observation,
                domain=SIGNATURE_DOMAIN_TESTNET_OBSERVATION,
                payload=payload,
            )
        )
        observed_nodes.add(node_id)
    if observed_nodes != expected_nodes:
        raise LaunchGateError("node_observations must cover every unique node")
    if len(monitor_public_keys) < PUBLIC_TESTNET_MIN_MONITORS:
        raise LaunchGateError("node_observations must be signed by independent monitors")
    return {"monitor_public_key_count": len(monitor_public_keys)}


def _validate_monitor_logs(
    evidence: Mapping[str, object],
    *,
    testnet_genesis_hash: str,
    start_time_ms: int,
    end_time_ms: int,
) -> int:
    logs = evidence.get("monitor_logs")
    if not isinstance(logs, Sequence) or isinstance(logs, str | bytes):
        raise LaunchGateError("monitor_logs must be a sequence")
    monitor_public_keys: set[str] = set()
    seen_urls: set[str] = set()
    head_anchor_hashes: set[str] = set()
    final_state_roots: set[str] = set()
    progress_records: set[tuple[int, int, int, int, int, int, tuple[str, ...]]] = set()
    for item in logs:
        if not isinstance(item, Mapping):
            raise LaunchGateError("monitor_logs entries must be objects")
        log = cast(Mapping[str, object], item)
        url = _require_url(log, "url")
        if url in seen_urls:
            raise LaunchGateError("monitor_logs URLs must be unique")
        seen_urls.add(url)
        content_hash = _require_hash_hex(log, "content_blake3")
        checkpoint_count = _require_positive_int(log, "checkpoint_count")
        head_anchor_hash = _require_hash_hex(log, "head_anchor_hash")
        final_state_root = _require_hash_hex(log, "final_state_root")
        max_checkpoint_gap_ms = _require_positive_int(log, "max_checkpoint_gap_ms")
        if max_checkpoint_gap_ms > DAY_MS:
            raise LaunchGateError("monitor_logs max_checkpoint_gap_ms exceeds one day")
        split_count = _require_nonnegative_int(log, "split_count")
        if split_count != 0:
            raise LaunchGateError("monitor_logs recorded consensus splits")
        start_anchor_height = _require_nonnegative_int(log, "start_anchor_height")
        final_anchor_height = _require_nonnegative_int(log, "final_anchor_height")
        if final_anchor_height <= start_anchor_height:
            raise LaunchGateError("monitor_logs final_anchor_height must advance")
        start_head_timestamp_ms = _require_nonnegative_int(log, "start_head_timestamp_ms")
        final_head_timestamp_ms = _require_nonnegative_int(log, "final_head_timestamp_ms")
        if start_head_timestamp_ms > start_time_ms:
            raise LaunchGateError("monitor_logs start head timestamp is after testnet start")
        if final_head_timestamp_ms < end_time_ms:
            raise LaunchGateError("monitor_logs final head timestamp is before testnet end")
        fruit_count = _require_positive_int(log, "fruit_count")
        anchor_count = _require_positive_int(log, "anchor_count")
        checkpoint_hashes = tuple(
            _require_hash_hex_value("monitor_logs checkpoint_hashes entry", checkpoint)
            for checkpoint in _require_unique_strings(log, "checkpoint_hashes")
        )
        if len(checkpoint_hashes) < PUBLIC_TESTNET_MIN_CHECKPOINTS:
            raise LaunchGateError("monitor_logs checkpoint_hashes are below minimum")
        if checkpoint_count != len(checkpoint_hashes):
            raise LaunchGateError("monitor_logs checkpoint_count mismatch")
        payload = {
            "anchor_count": anchor_count,
            "checkpoint_count": checkpoint_count,
            "checkpoint_hashes": list(checkpoint_hashes),
            "content_blake3": content_hash,
            "end_time_ms": end_time_ms,
            "final_anchor_height": final_anchor_height,
            "final_head_timestamp_ms": final_head_timestamp_ms,
            "final_state_root": final_state_root,
            "fruit_count": fruit_count,
            "head_anchor_hash": head_anchor_hash,
            "max_checkpoint_gap_ms": max_checkpoint_gap_ms,
            "split_count": split_count,
            "start_anchor_height": start_anchor_height,
            "start_head_timestamp_ms": start_head_timestamp_ms,
            "start_time_ms": start_time_ms,
            "testnet_genesis_hash": testnet_genesis_hash,
            "url": url,
        }
        monitor_public_keys.add(
            _require_signature(
                log,
                domain=SIGNATURE_DOMAIN_TESTNET_LOG,
                payload=payload,
            )
        )
        head_anchor_hashes.add(head_anchor_hash)
        final_state_roots.add(final_state_root)
        progress_records.add(
            (
                start_anchor_height,
                final_anchor_height,
                start_head_timestamp_ms,
                final_head_timestamp_ms,
                fruit_count,
                anchor_count,
                checkpoint_hashes,
            )
        )
    if len(head_anchor_hashes) != 1 or len(final_state_roots) != 1:
        raise LaunchGateError("monitor_logs must agree on final head anchor and state root")
    if len(progress_records) != 1:
        raise LaunchGateError("monitor_logs must agree on chain progress checkpoints")
    if len(monitor_public_keys) < PUBLIC_TESTNET_MIN_MONITORS:
        raise LaunchGateError("monitor_logs must be signed by independent monitors")
    return len(seen_urls)


def _validate_publication_attestations(
    evidence: Mapping[str, object],
    *,
    anchor_hash: str,
    ceremony_json_blake3: str,
    first_fruit_at_ms: int,
    first_fruit_hash: str,
    publication_targets: tuple[str, ...],
    published_at_ms: int,
) -> int:
    attestations = evidence.get("publication_attestations")
    if not isinstance(attestations, Sequence) or isinstance(attestations, str | bytes):
        raise LaunchGateError("publication_attestations must be a sequence")
    expected_targets = set(publication_targets)
    attested_targets: set[str] = set()
    attester_keys: set[str] = set()
    target_attester_keys: dict[str, set[str]] = {target: set() for target in expected_targets}
    for item in attestations:
        if not isinstance(item, Mapping):
            raise LaunchGateError("publication_attestations entries must be objects")
        attestation = cast(Mapping[str, object], item)
        target = _require_nonempty_str(attestation, "target")
        if target not in expected_targets:
            raise LaunchGateError("publication_attestations contains unknown target")
        content_hash = _require_hash_hex(attestation, "content_blake3")
        if content_hash != ceremony_json_blake3:
            raise LaunchGateError("publication attestation content_blake3 mismatch")
        if _require_hash_hex(attestation, "anchor_hash") != anchor_hash:
            raise LaunchGateError("publication attestation anchor_hash mismatch")
        evidence_url = _require_url(attestation, "evidence_url")
        payload = {
            "anchor_hash": anchor_hash,
            "content_blake3": content_hash,
            "evidence_url": evidence_url,
            "first_fruit_at_ms": first_fruit_at_ms,
            "first_fruit_hash": first_fruit_hash,
            "published_at_ms": published_at_ms,
            "target": target,
        }
        attester_key = _require_signature(
            attestation,
            domain=SIGNATURE_DOMAIN_PUBLICATION,
            payload=payload,
        )
        if attester_key in target_attester_keys[target]:
            raise LaunchGateError("publication_attestations duplicate target attester")
        attester_keys.add(attester_key)
        target_attester_keys[target].add(attester_key)
        attested_targets.add(target)
    missing = expected_targets - attested_targets
    if missing:
        raise LaunchGateError("publication_attestations missing publication target")
    under_attested = [
        target
        for target, target_keys in target_attester_keys.items()
        if len(target_keys) < PUBLICATION_MIN_ATTESTERS
    ]
    if under_attested:
        raise LaunchGateError("publication_attestations need independent attestations per target")
    return len(attester_keys)


def _validate_genesis_selection_evidence(
    ceremony_document: Mapping[str, object],
    *,
    bitcoin_block_hash: str,
    ethereum_block_hash: str,
) -> dict[str, int]:
    ceremony_start_ms = _require_nonnegative_int(ceremony_document, "ceremony_start_ms")
    selection_rules = ceremony_document.get("selection_rules")
    if not isinstance(selection_rules, Mapping):
        raise LaunchGateError("selection_rules must be an object")
    rules = cast(Mapping[str, object], selection_rules)
    if _require_str(rules, "bitcoin") != GENESIS_BITCOIN_SELECTION_RULE:
        raise LaunchGateError("selection_rules bitcoin rule mismatch")
    if _require_str(rules, "ethereum") != GENESIS_ETHEREUM_SELECTION_RULE:
        raise LaunchGateError("selection_rules ethereum rule mismatch")

    evidence = ceremony_document.get("selection_evidence")
    if not isinstance(evidence, Mapping):
        raise LaunchGateError("selection_evidence must be an object")
    selection = cast(Mapping[str, object], evidence)
    if _require_nonnegative_int(selection, "ceremony_start_ms") != ceremony_start_ms:
        raise LaunchGateError("selection_evidence ceremony_start_ms mismatch")

    bitcoin = selection.get("bitcoin")
    if not isinstance(bitcoin, Mapping):
        raise LaunchGateError("selection_evidence bitcoin must be an object")
    bitcoin_evidence = cast(Mapping[str, object], bitcoin)
    if _require_hash_hex(bitcoin_evidence, "block_hash") != bitcoin_block_hash:
        raise LaunchGateError("selection_evidence bitcoin block_hash mismatch")
    bitcoin_block_height = _require_positive_int(bitcoin_evidence, "block_height")
    bitcoin_tip_height = _require_positive_int(bitcoin_evidence, "confirmation_tip_height")
    _require_hash_hex(bitcoin_evidence, "confirmation_tip_hash")
    bitcoin_confirmations = _require_nonnegative_int(bitcoin_evidence, "confirmations")
    if bitcoin_confirmations != GENESIS_BTC_MIN_CONFIRMATIONS:
        raise LaunchGateError(
            f"selection_evidence bitcoin confirmations must equal {GENESIS_BTC_MIN_CONFIRMATIONS}"
        )
    if bitcoin_tip_height - bitcoin_block_height + 1 != bitcoin_confirmations:
        raise LaunchGateError("selection_evidence bitcoin confirmation tip is stale")
    if _require_nonnegative_int(bitcoin_evidence, "observed_at_ms") > ceremony_start_ms:
        raise LaunchGateError("selection_evidence bitcoin observed_at_ms is after ceremony start")
    if (
        _require_nonnegative_int(bitcoin_evidence, "confirmation_tip_observed_at_ms")
        > ceremony_start_ms
    ):
        raise LaunchGateError(
            "selection_evidence bitcoin confirmation_tip_observed_at_ms is after ceremony start"
        )
    if _require_str(bitcoin_evidence, "selection_rule") != GENESIS_BITCOIN_SELECTION_RULE:
        raise LaunchGateError("selection_evidence bitcoin selection_rule mismatch")
    bitcoin_source_url = _require_url(bitcoin_evidence, "source_url")
    bitcoin_source_content_hash = _require_hash_hex(bitcoin_evidence, "source_content_blake3")
    _validate_source_observations(
        bitcoin_evidence,
        payload={
            "block_hash": bitcoin_block_hash,
            "block_height": bitcoin_block_height,
            "chain": "bitcoin",
            "confirmation_tip_hash": _require_hash_hex(
                bitcoin_evidence,
                "confirmation_tip_hash",
            ),
            "confirmation_tip_height": bitcoin_tip_height,
            "confirmation_tip_observed_at_ms": _require_nonnegative_int(
                bitcoin_evidence,
                "confirmation_tip_observed_at_ms",
            ),
            "confirmations": bitcoin_confirmations,
            "observed_at_ms": _require_nonnegative_int(bitcoin_evidence, "observed_at_ms"),
            "selection_rule": GENESIS_BITCOIN_SELECTION_RULE,
            "source_content_blake3": bitcoin_source_content_hash,
            "source_url": bitcoin_source_url,
        },
    )

    ethereum = selection.get("ethereum")
    if not isinstance(ethereum, Mapping):
        raise LaunchGateError("selection_evidence ethereum must be an object")
    ethereum_evidence = cast(Mapping[str, object], ethereum)
    if _require_hash_hex(ethereum_evidence, "block_hash") != ethereum_block_hash:
        raise LaunchGateError("selection_evidence ethereum block_hash mismatch")
    ethereum_block_number = _require_positive_int(ethereum_evidence, "block_number")
    ethereum_finalized_head_number = _require_positive_int(
        ethereum_evidence,
        "finalized_head_number",
    )
    ethereum_finalized_head_hash = _require_hash_hex(ethereum_evidence, "finalized_head_hash")
    if ethereum_finalized_head_number != ethereum_block_number:
        raise LaunchGateError("selection_evidence ethereum finalized head is stale")
    if ethereum_finalized_head_hash != ethereum_block_hash:
        raise LaunchGateError("selection_evidence ethereum finalized head hash mismatch")
    if _require_nonnegative_int(ethereum_evidence, "finalized_at_ms") > ceremony_start_ms:
        raise LaunchGateError("selection_evidence ethereum finalized_at_ms is after ceremony start")
    if _require_str(ethereum_evidence, "selection_rule") != GENESIS_ETHEREUM_SELECTION_RULE:
        raise LaunchGateError("selection_evidence ethereum selection_rule mismatch")
    ethereum_source_url = _require_url(ethereum_evidence, "source_url")
    ethereum_source_content_hash = _require_hash_hex(ethereum_evidence, "source_content_blake3")
    _validate_source_observations(
        ethereum_evidence,
        payload={
            "block_hash": ethereum_block_hash,
            "block_number": ethereum_block_number,
            "chain": "ethereum",
            "finalized_at_ms": _require_nonnegative_int(
                ethereum_evidence,
                "finalized_at_ms",
            ),
            "finalized_head_hash": ethereum_finalized_head_hash,
            "finalized_head_number": ethereum_finalized_head_number,
            "selection_rule": GENESIS_ETHEREUM_SELECTION_RULE,
            "source_content_blake3": ethereum_source_content_hash,
            "source_url": ethereum_source_url,
        },
    )

    return {
        "bitcoin_block_height": bitcoin_block_height,
        "ceremony_start_ms": ceremony_start_ms,
        "ethereum_block_number": ethereum_block_number,
    }


def _validate_source_observations(
    document: Mapping[str, object],
    *,
    payload: Mapping[str, object],
) -> None:
    observations = document.get("source_observations")
    if not isinstance(observations, Sequence) or isinstance(observations, str | bytes):
        raise LaunchGateError("selection_evidence source_observations must be a sequence")
    monitor_public_keys: set[str] = set()
    for item in observations:
        if not isinstance(item, Mapping):
            raise LaunchGateError("selection_evidence source_observations entries must be objects")
        monitor_public_keys.add(
            _require_signature(
                cast(Mapping[str, object], item),
                domain=SIGNATURE_DOMAIN_SOURCE_SELECTION,
                payload=payload,
            )
        )
    if len(monitor_public_keys) < SOURCE_SELECTION_MIN_ATTESTERS:
        raise LaunchGateError("selection_evidence source observations need independent monitors")


def _require_canonical_hex_string(name: str, value: str) -> None:
    if value == "" or value.strip() != value:
        raise LaunchGateError(f"{name} must be canonical lowercase hex")
    if value.lower() != value:
        raise LaunchGateError(f"{name} must be canonical lowercase hex")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise LaunchGateError(f"{name} must be hex") from exc
    if decoded.hex() != value:
        raise LaunchGateError(f"{name} must be canonical lowercase hex")


def _require_hex_bytes(document: Mapping[str, object], key: str, expected_len: int) -> bytes:
    value = _require_str(document, key)
    _require_canonical_hex_string(key, value)
    decoded = bytes.fromhex(value)
    if len(decoded) != expected_len:
        raise LaunchGateError(f"{key} must decode to {expected_len} bytes")
    return decoded


def _require_signature(
    document: Mapping[str, object],
    *,
    domain: bytes,
    payload: Mapping[str, object],
) -> str:
    public_key = _require_hex_bytes(document, "monitor_public_key", ED25519_PUBLIC_KEY_BYTES)
    signature = _require_hex_bytes(document, "signature", ED25519_SIGNATURE_BYTES)
    if not verify(_signature_message(domain, payload), signature, public_key):
        raise LaunchGateError("signed launch-gate evidence has an invalid signature")
    return public_key.hex()


def _signature_message(domain: bytes, payload: Mapping[str, object]) -> bytes:
    return domain + _canonical_json_bytes(payload)


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _canonical_json_hash(document: Mapping[str, object]) -> str:
    return hash_bytes(_canonical_json_bytes(document)).hex()


def _require_bps(document: Mapping[str, object], key: str) -> int:
    value = _require_nonnegative_int(document, key)
    if value > 10_000:
        raise LaunchGateError(f"{key} must be <= 10000")
    return value


def _require_positive_int(document: Mapping[str, object], key: str) -> int:
    value = _require_nonnegative_int(document, key)
    if value <= 0:
        raise LaunchGateError(f"{key} must be positive")
    return value


def _require_nonnegative_int_value(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LaunchGateError(f"{name} must be int")
    if value < 0:
        raise LaunchGateError(f"{name} must be nonnegative")
    return value


def _current_time_ms() -> int:
    return int(time.time() * 1000)


def _require_founder_record(
    ceremony_document: Mapping[str, object],
    expected_pubkey_hash: bytes,
) -> None:
    founder = ceremony_document.get("founder")
    if not isinstance(founder, Mapping):
        raise LaunchGateError("founder must be an object")
    founder_mapping = cast(Mapping[str, object], founder)
    public_key_hex = _require_hash_hex(founder_mapping, "public_key")
    public_key = bytes.fromhex(public_key_hex)
    if founder_pubkey_hash(public_key) != expected_pubkey_hash:
        raise LaunchGateError("founder public_key does not match committed pubkey hash")
    if _require_hash_hex(founder_mapping, "pubkey_hash") != expected_pubkey_hash.hex():
        raise LaunchGateError("founder pubkey_hash does not match committed pubkey hash")
    if _require_nonempty_str(founder_mapping, "address") != founder_address(public_key):
        raise LaunchGateError("founder address does not match founder public_key")


def _require_nonempty_str(document: Mapping[str, object], key: str) -> str:
    return _require_nonempty_string_value(key, document.get(key))


def _require_str(document: Mapping[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise LaunchGateError(f"{key} must be str")
    return value


def _require_nonempty_string_value(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise LaunchGateError(f"{name} must be str")
    if value == "" or value.strip() != value:
        raise LaunchGateError(f"{name} must be nonempty without surrounding whitespace")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
