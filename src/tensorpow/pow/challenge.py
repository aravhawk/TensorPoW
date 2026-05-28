"""Deterministic PoW challenge preimages and matrix construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import torch
from blake3 import blake3

from tensorpow.crypto.hash import (
    DOMAIN_POW_CHALLENGE_ANCHOR,
    DOMAIN_POW_CHALLENGE_FRUIT,
    DOMAIN_POW_MATRIX_A,
    DOMAIN_POW_MATRIX_B,
    HASH_LEN_BYTES,
)
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.kernel import POW_MATRIX_DIM

FORMAT_EPOCH: Final[int] = 0
U16_MAX: Final[int] = 0xFFFF
U32_MAX: Final[int] = 0xFFFFFFFF
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF
GENESIS_PARENT_HASH: Final[bytes] = bytes(HASH_LEN_BYTES)


@dataclass(frozen=True, slots=True)
class FruitPowHeader:
    """Fruit header fields needed to reconstruct the PoW challenge."""

    version: int
    sig_type_supported: int
    effective_parent_hashes: tuple[bytes, ...]
    latest_anchor: bytes
    tx_merkle_root: bytes
    timestamp_ms: int
    shard_id: int
    nonce: int

    def __post_init__(self) -> None:
        _require_u16("version", self.version)
        _require_u16("sig_type_supported", self.sig_type_supported)
        if self.sig_type_supported & SIG_TYPE_ED25519_BIT == 0:
            raise ValueError("sig_type_supported must include SIG_TYPE_ED25519_BIT")
        _require_hash_tuple("effective_parent_hashes", self.effective_parent_hashes)
        _require_hash("latest_anchor", self.latest_anchor)
        _require_hash("tx_merkle_root", self.tx_merkle_root)
        _require_u64("timestamp_ms", self.timestamp_ms)
        _require_u32("shard_id", self.shard_id)
        _require_u64("nonce", self.nonce)


@dataclass(frozen=True, slots=True)
class AnchorPowHeader:
    """Anchor header fields needed to reconstruct the PoW challenge."""

    version: int
    parent_anchor: bytes
    fruit_set_root: bytes
    parent_candidate_root: bytes
    shard_tree_state_root: bytes
    fee_floor_set_root: bytes
    anchor_reward_root: bytes
    timestamp_ms: int
    nonce: int

    def __post_init__(self) -> None:
        _require_u16("version", self.version)
        _require_hash("parent_anchor", self.parent_anchor)
        _require_hash("fruit_set_root", self.fruit_set_root)
        _require_hash("parent_candidate_root", self.parent_candidate_root)
        _require_hash("shard_tree_state_root", self.shard_tree_state_root)
        _require_hash("fee_floor_set_root", self.fee_floor_set_root)
        _require_hash("anchor_reward_root", self.anchor_reward_root)
        _require_u64("timestamp_ms", self.timestamp_ms)
        _require_u64("nonce", self.nonce)


type PowHeader = FruitPowHeader | AnchorPowHeader


def fruit_pow_preimage(header: FruitPowHeader) -> bytes:
    """Encode the normative fruit PoW preimage."""

    if not isinstance(header, FruitPowHeader):
        raise TypeError("header must be FruitPowHeader")
    return b"".join(
        (
            bytes((DOMAIN_POW_CHALLENGE_FRUIT,)),
            _u16(header.version),
            _u16(header.sig_type_supported),
            _u16(len(header.effective_parent_hashes)),
            b"".join(header.effective_parent_hashes),
            header.latest_anchor,
            header.tx_merkle_root,
            _u64(header.timestamp_ms),
            _u32(header.shard_id),
            _u64(header.nonce),
        )
    )


def anchor_pow_preimage(header: AnchorPowHeader) -> bytes:
    """Encode the normative anchor PoW preimage."""

    if not isinstance(header, AnchorPowHeader):
        raise TypeError("header must be AnchorPowHeader")
    return b"".join(
        (
            bytes((DOMAIN_POW_CHALLENGE_ANCHOR,)),
            _u16(header.version),
            header.parent_anchor,
            header.fruit_set_root,
            header.parent_candidate_root,
            header.shard_tree_state_root,
            header.fee_floor_set_root,
            header.anchor_reward_root,
            _u64(header.timestamp_ms),
            _u64(header.nonce),
        )
    )


def pow_preimage(header: PowHeader) -> bytes:
    """Encode the PoW preimage for a fruit or anchor header."""

    if isinstance(header, FruitPowHeader):
        return fruit_pow_preimage(header)
    if isinstance(header, AnchorPowHeader):
        return anchor_pow_preimage(header)
    raise TypeError("header must be FruitPowHeader or AnchorPowHeader")


def with_nonce(header: PowHeader, nonce: int) -> PowHeader:
    """Return a copy of a PoW header with nonce replaced."""

    _require_u64("nonce", nonce)
    return replace(header, nonce=nonce)


def build_challenge_matrices(
    header: PowHeader, *, matrix_dim: int = POW_MATRIX_DIM
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the deterministic A and B INT8 challenge matrices for a header."""

    preimage = pow_preimage(header)
    return (
        _matrix_from_xof(DOMAIN_POW_MATRIX_A, preimage, matrix_dim=matrix_dim),
        _matrix_from_xof(DOMAIN_POW_MATRIX_B, preimage, matrix_dim=matrix_dim),
    )


def build_challenge(
    parent_hashes: list[bytes],
    tx_merkle_root: bytes,
    timestamp: int,
    nonce: int,
    domain: int,
    *,
    matrix_dim: int = POW_MATRIX_DIM,
) -> torch.Tensor:
    """Compatibility helper for the milestone API; returns matrix A."""

    if domain == DOMAIN_POW_CHALLENGE_FRUIT:
        header: PowHeader = FruitPowHeader(
            version=FORMAT_EPOCH,
            sig_type_supported=SIG_TYPE_ED25519_BIT,
            effective_parent_hashes=tuple(parent_hashes),
            latest_anchor=GENESIS_PARENT_HASH,
            tx_merkle_root=tx_merkle_root,
            timestamp_ms=timestamp,
            shard_id=0,
            nonce=nonce,
        )
    elif domain == DOMAIN_POW_CHALLENGE_ANCHOR:
        if len(parent_hashes) != 1:
            raise ValueError("anchor compatibility challenge requires one parent hash")
        header = AnchorPowHeader(
            version=FORMAT_EPOCH,
            parent_anchor=parent_hashes[0],
            fruit_set_root=tx_merkle_root,
            parent_candidate_root=GENESIS_PARENT_HASH,
            shard_tree_state_root=GENESIS_PARENT_HASH,
            fee_floor_set_root=GENESIS_PARENT_HASH,
            anchor_reward_root=GENESIS_PARENT_HASH,
            timestamp_ms=timestamp,
            nonce=nonce,
        )
    else:
        raise ValueError("domain must be a PoW challenge domain")
    return build_challenge_matrices(header, matrix_dim=matrix_dim)[0]


def _matrix_from_xof(domain: int, preimage: bytes, *, matrix_dim: int) -> torch.Tensor:
    if not isinstance(matrix_dim, int):
        raise TypeError("matrix_dim must be int")
    if matrix_dim <= 0:
        raise ValueError("matrix_dim must be positive")
    byte_len = matrix_dim * matrix_dim
    raw = blake3(bytes((domain,)) + preimage).digest(length=byte_len)
    values = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int16)
    values = torch.where(values >= 128, values - 256, values).to(torch.int8)
    return values.reshape(matrix_dim, matrix_dim).contiguous()


def _require_hash(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _require_hash_tuple(name: str, value: tuple[bytes, ...]) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(value) > U16_MAX:
        raise ValueError(f"{name} must fit in uint16 count")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    for item in value:
        _require_hash(name, item)


def _require_u16(name: str, value: int) -> None:
    _require_uint(name, value, U16_MAX)


def _require_u32(name: str, value: int) -> None:
    _require_uint(name, value, U32_MAX)


def _require_u64(name: str, value: int) -> None:
    _require_uint(name, value, U64_MAX)


def _require_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} outside uint range")


def _u16(value: int) -> bytes:
    return value.to_bytes(2, "little")


def _u32(value: int) -> bytes:
    return value.to_bytes(4, "little")


def _u64(value: int) -> bytes:
    return value.to_bytes(8, "little")
