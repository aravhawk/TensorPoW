"""Build deterministic TensorPoW genesis ceremony artifacts."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes
from tensorpow.genesis import (
    GENESIS_CHAIN_ID_MAINNET,
    GENESIS_CHAIN_ID_TESTNET,
    GenesisError,
    GenesisInputs,
    build_genesis_artifact,
    founder_address,
    founder_pubkey_hash,
)
from tensorpow.launch_policy import (
    GENESIS_BITCOIN_SELECTION_RULE,
    GENESIS_BTC_MIN_CONFIRMATIONS,
    GENESIS_ETHEREUM_SELECTION_RULE,
)
from tensorpow.wallet import create_keystore, load_keystore

DNS_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the genesis ceremony artifact builder."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        artifact_json = build_ceremony_document(args)
        encoded = json.dumps(artifact_json, indent=2, sort_keys=True) + "\n"
        if args.out is not None:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    except (OSError, TypeError, ValueError, GenesisError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genesis_ceremony")
    parser.add_argument(
        "--chain-id",
        choices=(GENESIS_CHAIN_ID_MAINNET, GENESIS_CHAIN_ID_TESTNET),
        default=GENESIS_CHAIN_ID_MAINNET,
    )
    whitepaper = parser.add_mutually_exclusive_group(required=True)
    whitepaper.add_argument("--whitepaper-file", type=Path)
    whitepaper.add_argument("--whitepaper-hash")
    parser.add_argument("--ceremony-start-ms", type=int, required=True)
    parser.add_argument("--bitcoin-block-hash", required=True)
    parser.add_argument("--bitcoin-block-height", type=int, required=True)
    parser.add_argument("--bitcoin-confirmations", type=int, required=True)
    parser.add_argument("--bitcoin-confirmation-tip-height", type=int, required=True)
    parser.add_argument("--bitcoin-confirmation-tip-hash", required=True)
    parser.add_argument("--bitcoin-confirmation-tip-observed-at-ms", type=int, required=True)
    parser.add_argument("--bitcoin-observed-at-ms", type=int, required=True)
    parser.add_argument("--bitcoin-source-content-blake3", required=True)
    parser.add_argument("--bitcoin-source-url", required=True)
    parser.add_argument("--ethereum-block-hash", required=True)
    parser.add_argument("--ethereum-block-number", type=int, required=True)
    parser.add_argument("--ethereum-finalized-head-number", type=int, required=True)
    parser.add_argument("--ethereum-finalized-head-hash", required=True)
    parser.add_argument("--ethereum-finalized-at-ms", type=int, required=True)
    parser.add_argument("--ethereum-source-content-blake3", required=True)
    parser.add_argument("--ethereum-source-url", required=True)
    founder = parser.add_mutually_exclusive_group(required=True)
    founder.add_argument("--founder-public-key-hex")
    founder.add_argument("--founder-keystore", type=Path)
    parser.add_argument(
        "--founder-password-env",
        default="TENSORPOW_FOUNDER_KEYSTORE_PASSWORD",
        help="environment variable used to create or unlock --founder-keystore",
    )
    parser.add_argument("--timestamp-ms", type=int, default=0)
    parser.add_argument("--nonce", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser


def build_ceremony_document(args: argparse.Namespace) -> dict[str, object]:
    """Return the complete public ceremony document for parsed CLI args."""

    founder_public_key = _founder_public_key(args)
    ceremony_start_ms = _nonnegative_int("ceremony_start_ms", args.ceremony_start_ms)
    bitcoin_hash = _hash_arg("bitcoin_block_hash", args.bitcoin_block_hash)
    ethereum_hash = _hash_arg("ethereum_block_hash", args.ethereum_block_hash)
    inputs = GenesisInputs.create(
        chain_id=args.chain_id,
        whitepaper_hash=_whitepaper_hash(args),
        bitcoin_block_hash=bitcoin_hash,
        ethereum_block_hash=ethereum_hash,
        founder_pubkey_hash=founder_pubkey_hash(founder_public_key),
    )
    artifact = build_genesis_artifact(inputs, timestamp_ms=args.timestamp_ms, nonce=args.nonce)
    document = artifact.to_json()
    document["founder"] = {
        "address": founder_address(founder_public_key),
        "public_key": founder_public_key.hex(),
        "pubkey_hash": inputs.founder_pubkey_hash.hex(),
    }
    document["publication"] = {
        "github_release_title_format": "vX.Y.Z",
        "mirror_payload": "anchor_hex and this JSON document",
        "permanent_storage_targets": ["IPFS", "Arweave"],
    }
    document["selection_rules"] = {
        "bitcoin": GENESIS_BITCOIN_SELECTION_RULE,
        "ethereum": GENESIS_ETHEREUM_SELECTION_RULE,
    }
    document["ceremony_start_ms"] = ceremony_start_ms
    document["selection_evidence"] = _selection_evidence(
        args,
        ceremony_start_ms=ceremony_start_ms,
        bitcoin_hash=bitcoin_hash,
        ethereum_hash=ethereum_hash,
    )
    return document


def _whitepaper_hash(args: argparse.Namespace) -> bytes:
    if args.whitepaper_hash is not None:
        return _hash_arg("whitepaper_hash", args.whitepaper_hash)
    return hash_bytes(Path(args.whitepaper_file).read_bytes())


def _founder_public_key(args: argparse.Namespace) -> bytes:
    if args.founder_public_key_hex is not None:
        return _hash_arg("founder_public_key", args.founder_public_key_hex)
    password = os.environ.get(args.founder_password_env)
    if not password:
        raise GenesisError(f"{args.founder_password_env} must be set for --founder-keystore")
    path = Path(args.founder_keystore)
    wallet = load_keystore(path, password) if path.exists() else create_keystore(path, password)
    return wallet.public_key


def _hash_arg(name: str, value: str) -> bytes:
    if value == "" or value.strip() != value or value.lower() != value:
        raise GenesisError(f"{name} must be canonical lowercase hex")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise GenesisError(f"{name} must be hex") from exc
    if decoded.hex() != value:
        raise GenesisError(f"{name} must be canonical lowercase hex")
    if len(decoded) != HASH_LEN_BYTES:
        raise GenesisError(f"{name} must decode to {HASH_LEN_BYTES} bytes")
    return decoded


def _selection_evidence(
    args: argparse.Namespace,
    *,
    ceremony_start_ms: int,
    bitcoin_hash: bytes,
    ethereum_hash: bytes,
) -> dict[str, object]:
    bitcoin_height = _positive_int("bitcoin_block_height", args.bitcoin_block_height)
    bitcoin_confirmations = _positive_int("bitcoin_confirmations", args.bitcoin_confirmations)
    bitcoin_tip_height = _positive_int(
        "bitcoin_confirmation_tip_height",
        args.bitcoin_confirmation_tip_height,
    )
    bitcoin_tip_hash = _hash_arg(
        "bitcoin_confirmation_tip_hash",
        args.bitcoin_confirmation_tip_hash,
    )
    bitcoin_source_content_hash = _hash_arg(
        "bitcoin_source_content_blake3",
        args.bitcoin_source_content_blake3,
    )
    bitcoin_tip_observed_at_ms = _nonnegative_int(
        "bitcoin_confirmation_tip_observed_at_ms",
        args.bitcoin_confirmation_tip_observed_at_ms,
    )
    bitcoin_observed_at_ms = _nonnegative_int("bitcoin_observed_at_ms", args.bitcoin_observed_at_ms)
    if bitcoin_confirmations != GENESIS_BTC_MIN_CONFIRMATIONS:
        raise GenesisError(f"bitcoin_confirmations must equal {GENESIS_BTC_MIN_CONFIRMATIONS}")
    if bitcoin_tip_height - bitcoin_height + 1 != bitcoin_confirmations:
        raise GenesisError("bitcoin confirmation tip height does not match confirmations")
    if bitcoin_observed_at_ms > ceremony_start_ms:
        raise GenesisError("bitcoin_observed_at_ms must not be after ceremony_start_ms")
    if bitcoin_tip_observed_at_ms > ceremony_start_ms:
        raise GenesisError(
            "bitcoin_confirmation_tip_observed_at_ms must not be after ceremony_start_ms"
        )

    ethereum_block_number = _positive_int("ethereum_block_number", args.ethereum_block_number)
    ethereum_finalized_head_number = _positive_int(
        "ethereum_finalized_head_number",
        args.ethereum_finalized_head_number,
    )
    ethereum_finalized_head_hash = _hash_arg(
        "ethereum_finalized_head_hash",
        args.ethereum_finalized_head_hash,
    )
    ethereum_source_content_hash = _hash_arg(
        "ethereum_source_content_blake3",
        args.ethereum_source_content_blake3,
    )
    ethereum_finalized_at_ms = _nonnegative_int(
        "ethereum_finalized_at_ms",
        args.ethereum_finalized_at_ms,
    )
    if ethereum_finalized_head_number != ethereum_block_number:
        raise GenesisError("ethereum_finalized_head_number must equal ethereum_block_number")
    if ethereum_finalized_head_hash != ethereum_hash:
        raise GenesisError("ethereum_finalized_head_hash must equal ethereum_block_hash")
    if ethereum_finalized_at_ms > ceremony_start_ms:
        raise GenesisError("ethereum_finalized_at_ms must not be after ceremony_start_ms")

    return {
        "ceremony_start_ms": ceremony_start_ms,
        "bitcoin": {
            "block_hash": bitcoin_hash.hex(),
            "block_height": bitcoin_height,
            "confirmation_tip_hash": bitcoin_tip_hash.hex(),
            "confirmation_tip_height": bitcoin_tip_height,
            "confirmation_tip_observed_at_ms": bitcoin_tip_observed_at_ms,
            "confirmations": bitcoin_confirmations,
            "observed_at_ms": bitcoin_observed_at_ms,
            "selection_rule": GENESIS_BITCOIN_SELECTION_RULE,
            "source_content_blake3": bitcoin_source_content_hash.hex(),
            "source_observations": [],
            "source_url": _require_public_https_url("bitcoin_source_url", args.bitcoin_source_url),
        },
        "ethereum": {
            "block_hash": ethereum_hash.hex(),
            "block_number": ethereum_block_number,
            "finalized_head_hash": ethereum_finalized_head_hash.hex(),
            "finalized_head_number": ethereum_finalized_head_number,
            "finalized_at_ms": ethereum_finalized_at_ms,
            "selection_rule": GENESIS_ETHEREUM_SELECTION_RULE,
            "source_content_blake3": ethereum_source_content_hash.hex(),
            "source_observations": [],
            "source_url": _require_public_https_url(
                "ethereum_source_url", args.ethereum_source_url
            ),
        },
    }


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GenesisError(f"{name} must be int")
    if value <= 0:
        raise GenesisError(f"{name} must be positive")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GenesisError(f"{name} must be int")
    if value < 0:
        raise GenesisError(f"{name} must be nonnegative")
    return value


def _require_public_https_url(name: str, value: str) -> str:
    if not isinstance(value, str) or value == "" or value.strip() != value:
        raise GenesisError(f"{name} must be a nonempty URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname is None or parsed.path in ("", "/"):
        raise GenesisError(f"{name} must be an https URL with a path")
    _require_public_host(name, parsed.hostname)
    return value


def _require_public_host(name: str, host: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        _require_public_dns_name(name, host)
        return
    if not address.is_global:
        raise GenesisError(f"{name} host must be public")


def _require_public_dns_name(name: str, host: str) -> None:
    if host == "" or host.strip() != host or host.lower() != host:
        raise GenesisError(f"{name} host must be a canonical public DNS name")
    if host.endswith("."):
        raise GenesisError(f"{name} host must be a canonical public DNS name")
    if host in ("localhost",) or host.endswith(".localhost") or host.endswith(".local"):
        raise GenesisError(f"{name} host must be public")
    if "." not in host:
        raise GenesisError(f"{name} host must contain a public suffix")
    labels = host.rstrip(".").split(".")
    if any(DNS_LABEL_RE.fullmatch(label) is None for label in labels):
        raise GenesisError(f"{name} host is malformed")
    if _is_reserved_dns_name(host):
        raise GenesisError(f"{name} host must use a globally delegated public DNS name")


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


if __name__ == "__main__":
    raise SystemExit(main())
