"""Deterministic genesis and public-testnet artifact construction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, cast

from tensorpow.chain.blocks import (
    Anchor,
    FeeFloorEntry,
    anchor_reward_root,
    fee_floor_set_root,
    fruit_set_root,
    parent_candidate_root,
)
from tensorpow.chain.headers import AnchorHeader
from tensorpow.crypto.address import pubkey_to_address
from tensorpow.crypto.hash import (
    DOMAIN_ADDRESS,
    DOMAIN_GENESIS,
    HASH_LEN_BYTES,
    domain_hash,
    hash_bytes,
)
from tensorpow.mempool import ROOT_SHARD_ID, ShardTree
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH
from tensorpow.state import UTXOSet

GENESIS_CHAIN_ID_MAINNET: Final[str] = "tensorpow-mainnet"
GENESIS_CHAIN_ID_TESTNET: Final[str] = "tensorpow-testnet"
GENESIS_ERA_MARKER: Final[str] = "tensorpow-2026"
GENESIS_ROOT_FEE_FLOOR_MATOMS_PER_KB: Final[int] = 0
GENESIS_TIMESTAMP_MS: Final[int] = 0
GENESIS_NONCE: Final[int] = 0


class GenesisError(ValueError):
    """Raised when ceremony inputs are malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class GenesisInputs:
    """Public inputs committed into a TensorPoW genesis anchor."""

    chain_id: str
    era_marker: str
    whitepaper_hash: bytes
    bitcoin_block_hash: bytes
    ethereum_block_hash: bytes
    founder_pubkey_hash: bytes
    empty_utxo_root: bytes
    initial_shard_tree_root: bytes
    initial_fee_floor_root: bytes

    def __post_init__(self) -> None:
        _require_chain_id(self.chain_id)
        if self.era_marker != GENESIS_ERA_MARKER:
            raise GenesisError(f"era_marker must be {GENESIS_ERA_MARKER!r}")
        _require_hash("whitepaper_hash", self.whitepaper_hash)
        _require_hash("bitcoin_block_hash", self.bitcoin_block_hash)
        _require_hash("ethereum_block_hash", self.ethereum_block_hash)
        _require_hash("founder_pubkey_hash", self.founder_pubkey_hash)
        _require_hash("empty_utxo_root", self.empty_utxo_root)
        _require_hash("initial_shard_tree_root", self.initial_shard_tree_root)
        _require_hash("initial_fee_floor_root", self.initial_fee_floor_root)

    @classmethod
    def create(
        cls,
        *,
        chain_id: str,
        whitepaper_hash: bytes,
        bitcoin_block_hash: bytes,
        ethereum_block_hash: bytes,
        founder_pubkey_hash: bytes,
    ) -> GenesisInputs:
        """Build inputs with canonical empty-state roots."""

        shard_tree = ShardTree()
        fee_entries = _genesis_fee_floor_entries()
        return cls(
            chain_id=chain_id,
            era_marker=GENESIS_ERA_MARKER,
            whitepaper_hash=whitepaper_hash,
            bitcoin_block_hash=bitcoin_block_hash,
            ethereum_block_hash=ethereum_block_hash,
            founder_pubkey_hash=founder_pubkey_hash,
            empty_utxo_root=UTXOSet().merkle_root(),
            initial_shard_tree_root=shard_tree.state_root(),
            initial_fee_floor_root=fee_floor_set_root(fee_entries),
        )

    def commitment_preimage(self) -> bytes:
        """Return the exact bytes after the genesis domain byte."""

        return b"".join(
            (
                self.chain_id.encode("utf-8"),
                self.era_marker.encode("utf-8"),
                self.whitepaper_hash,
                self.bitcoin_block_hash,
                self.ethereum_block_hash,
                self.founder_pubkey_hash,
                self.empty_utxo_root,
                self.initial_shard_tree_root,
                self.initial_fee_floor_root,
            )
        )

    def commitment(self) -> bytes:
        """Return BLAKE3(DOMAIN_GENESIS || public genesis inputs)."""

        return domain_hash(DOMAIN_GENESIS, self.commitment_preimage())

    def to_json(self) -> dict[str, str]:
        """Return public ceremony inputs as deterministic JSON fields."""

        return {
            "chain_id": self.chain_id,
            "era_marker": self.era_marker,
            "whitepaper_hash": self.whitepaper_hash.hex(),
            "bitcoin_block_hash": self.bitcoin_block_hash.hex(),
            "ethereum_block_hash": self.ethereum_block_hash.hex(),
            "founder_pubkey_hash": self.founder_pubkey_hash.hex(),
            "empty_utxo_root": self.empty_utxo_root.hex(),
            "initial_shard_tree_root": self.initial_shard_tree_root.hex(),
            "initial_fee_floor_root": self.initial_fee_floor_root.hex(),
        }


@dataclass(frozen=True, slots=True)
class GenesisArtifact:
    """Deterministic genesis anchor plus its public metadata."""

    inputs: GenesisInputs
    anchor: Anchor

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, GenesisInputs):
            raise TypeError("inputs must be GenesisInputs")
        if not isinstance(self.anchor, Anchor):
            raise TypeError("anchor must be Anchor")
        if self.anchor.genesis_commitment != self.inputs.commitment():
            raise GenesisError("anchor genesis commitment does not match inputs")

    @property
    def block_hash(self) -> bytes:
        """Return the full serialized genesis anchor hash for public commitment."""

        return hash_bytes(self.anchor.serialize())

    @property
    def header_hash(self) -> bytes:
        """Return the chain-separated genesis anchor identifier used by descendants."""

        return self.anchor.block_hash()

    def to_json(self) -> dict[str, object]:
        """Return a deterministic public genesis artifact."""

        return {
            "anchor_hash": self.block_hash.hex(),
            "anchor_header_hash": self.header_hash.hex(),
            "anchor_hex": self.anchor.serialize().hex(),
            "genesis_commitment": self.inputs.commitment().hex(),
            "inputs": self.inputs.to_json(),
        }

    def to_json_bytes(self) -> bytes:
        """Return canonical JSON bytes for publication."""

        return json.dumps(self.to_json(), indent=2, sort_keys=True).encode("utf-8") + b"\n"


def founder_pubkey_hash(public_key: bytes) -> bytes:
    """Return the Tensorcoin owner pubkey hash for a founder public key."""

    if not isinstance(public_key, bytes) or len(public_key) != HASH_LEN_BYTES:
        raise GenesisError(f"public_key must be {HASH_LEN_BYTES} bytes")
    return domain_hash(DOMAIN_ADDRESS, public_key)


def founder_address(public_key: bytes) -> str:
    """Return the public Tensorcoin address for a founder public key."""

    return pubkey_to_address(public_key)


def build_genesis_anchor(
    inputs: GenesisInputs,
    *,
    timestamp_ms: int = GENESIS_TIMESTAMP_MS,
    nonce: int = GENESIS_NONCE,
) -> Anchor:
    """Build the deterministic genesis anchor for ``inputs``."""

    if not isinstance(inputs, GenesisInputs):
        raise TypeError("inputs must be GenesisInputs")
    shard_tree = ShardTree()
    fee_entries = _genesis_fee_floor_entries()
    header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=GENESIS_PARENT_HASH,
        fruit_set_root=fruit_set_root(()),
        parent_candidate_root=parent_candidate_root(()),
        shard_tree_state_root=shard_tree.state_root(),
        fee_floor_set_root=fee_floor_set_root(fee_entries),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=_require_nonnegative_int("timestamp_ms", timestamp_ms),
        nonce=_require_nonnegative_int("nonce", nonce),
    )
    return Anchor(
        header=header,
        covered_fruit_hashes=(),
        parent_candidate_hashes=(),
        shard_tree_bytes=shard_tree.serialize(),
        fee_floor_entries=fee_entries,
        genesis_commitment=inputs.commitment(),
    )


def build_genesis_artifact(
    inputs: GenesisInputs,
    *,
    timestamp_ms: int = GENESIS_TIMESTAMP_MS,
    nonce: int = GENESIS_NONCE,
) -> GenesisArtifact:
    """Build the deterministic public genesis artifact."""

    return GenesisArtifact(
        inputs=inputs,
        anchor=build_genesis_anchor(inputs, timestamp_ms=timestamp_ms, nonce=nonce),
    )


def artifact_from_json(document: Mapping[str, object]) -> GenesisArtifact:
    """Parse and verify a published genesis artifact."""

    inputs = inputs_from_json(_expect_mapping(document, "inputs"))
    anchor_hex = _expect_str(document, "anchor_hex")
    _require_canonical_hex_string("anchor_hex", anchor_hex)
    try:
        anchor = Anchor.deserialize(bytes.fromhex(anchor_hex))
    except (TypeError, ValueError) as exc:
        raise GenesisError("anchor_hex is malformed") from exc
    artifact = GenesisArtifact(inputs=inputs, anchor=anchor)
    expected_hash = _expect_str(document, "anchor_hash")
    _require_canonical_hex_string("anchor_hash", expected_hash)
    if artifact.block_hash.hex() != expected_hash:
        raise GenesisError("anchor_hash does not match anchor_hex")
    expected_header_hash = _expect_str(document, "anchor_header_hash")
    _require_canonical_hex_string("anchor_header_hash", expected_header_hash)
    if artifact.header_hash.hex() != expected_header_hash:
        raise GenesisError("anchor_header_hash does not match anchor_hex")
    expected_commitment = _expect_str(document, "genesis_commitment")
    _require_canonical_hex_string("genesis_commitment", expected_commitment)
    if artifact.inputs.commitment().hex() != expected_commitment:
        raise GenesisError("genesis_commitment does not match inputs")
    return artifact


def inputs_from_json(document: Mapping[str, object]) -> GenesisInputs:
    """Parse public genesis inputs from JSON fields."""

    return GenesisInputs(
        chain_id=_expect_str(document, "chain_id"),
        era_marker=_expect_str(document, "era_marker"),
        whitepaper_hash=_expect_hash_hex(document, "whitepaper_hash"),
        bitcoin_block_hash=_expect_hash_hex(document, "bitcoin_block_hash"),
        ethereum_block_hash=_expect_hash_hex(document, "ethereum_block_hash"),
        founder_pubkey_hash=_expect_hash_hex(document, "founder_pubkey_hash"),
        empty_utxo_root=_expect_hash_hex(document, "empty_utxo_root"),
        initial_shard_tree_root=_expect_hash_hex(document, "initial_shard_tree_root"),
        initial_fee_floor_root=_expect_hash_hex(document, "initial_fee_floor_root"),
    )


def _genesis_fee_floor_entries() -> tuple[FeeFloorEntry, ...]:
    return (FeeFloorEntry(ROOT_SHARD_ID, GENESIS_ROOT_FEE_FLOOR_MATOMS_PER_KB),)


def _require_chain_id(chain_id: str) -> None:
    if chain_id not in (GENESIS_CHAIN_ID_MAINNET, GENESIS_CHAIN_ID_TESTNET):
        raise GenesisError("chain_id must be a TensorPoW mainnet or testnet chain id")


def _expect_mapping(document: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise GenesisError(f"{key} must be an object")
    return cast(Mapping[str, object], value)


def _expect_str(document: Mapping[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or value == "":
        raise GenesisError(f"{key} must be a non-empty string")
    return value


def _expect_hash_hex(document: Mapping[str, object], key: str) -> bytes:
    value_text = _expect_str(document, key)
    _require_canonical_hex_string(key, value_text)
    try:
        value = bytes.fromhex(value_text)
    except ValueError as exc:
        raise GenesisError(f"{key} must be hex") from exc
    _require_hash(key, value)
    return value


def _require_hash(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != HASH_LEN_BYTES:
        raise GenesisError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _require_nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise GenesisError(f"{name} must be nonnegative")
    return value


def _require_canonical_hex_string(name: str, value: str) -> None:
    if value == "" or value.strip() != value or value.lower() != value:
        raise GenesisError(f"{name} must be canonical lowercase hex")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise GenesisError(f"{name} must be hex") from exc
    if decoded.hex() != value:
        raise GenesisError(f"{name} must be canonical lowercase hex")


__all__ = [
    "GENESIS_CHAIN_ID_MAINNET",
    "GENESIS_CHAIN_ID_TESTNET",
    "GENESIS_ERA_MARKER",
    "GENESIS_NONCE",
    "GENESIS_ROOT_FEE_FLOOR_MATOMS_PER_KB",
    "GENESIS_TIMESTAMP_MS",
    "GenesisArtifact",
    "GenesisError",
    "GenesisInputs",
    "artifact_from_json",
    "build_genesis_anchor",
    "build_genesis_artifact",
    "founder_address",
    "founder_pubkey_hash",
    "inputs_from_json",
]
