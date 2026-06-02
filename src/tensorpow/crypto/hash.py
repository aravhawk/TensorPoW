"""BLAKE3 hashing and Merkle helpers for TensorPoW."""

from __future__ import annotations

from typing import Final

from blake3 import blake3

HASH_LEN_BYTES: Final[int] = 32
MERKLE_DEPTH_BITS: Final[int] = HASH_LEN_BYTES * 8

DOMAIN_POW_CHALLENGE_FRUIT: Final[int] = 0x00
DOMAIN_POW_CHALLENGE_ANCHOR: Final[int] = 0x01
DOMAIN_POW_OUTPUT: Final[int] = 0x02
DOMAIN_POW_MATRIX_A: Final[int] = 0x03
DOMAIN_POW_MATRIX_B: Final[int] = 0x04
DOMAIN_FRUIT_HEADER: Final[int] = 0x10
DOMAIN_ANCHOR_HEADER: Final[int] = 0x11
DOMAIN_TX_MERKLE_ROOT: Final[int] = 0x12
DOMAIN_FRUIT_SET_ROOT: Final[int] = 0x13
DOMAIN_PARENT_CANDIDATE_ROOT: Final[int] = 0x14
DOMAIN_ANCHOR_REWARD_ROOT: Final[int] = 0x15
DOMAIN_TX_ID: Final[int] = 0x20
DOMAIN_TX_SIGHASH: Final[int] = 0x21
DOMAIN_OUTPOINT: Final[int] = 0x22
DOMAIN_UTXO: Final[int] = 0x23
DOMAIN_MERKLE_LEAF: Final[int] = 0x30
DOMAIN_MERKLE_NODE: Final[int] = 0x31
DOMAIN_MERKLE_EMPTY: Final[int] = 0x32
MERKLE_LEAF_TAG: Final[int] = 0x00
MERKLE_NODE_TAG: Final[int] = 0x01
MERKLE_EMPTY_TAG: Final[int] = 0x02
DOMAIN_ADDRESS: Final[int] = 0x40
DOMAIN_GENESIS: Final[int] = 0x50
DOMAIN_SHARD_TREE: Final[int] = 0x60
DOMAIN_FEE_FLOOR: Final[int] = 0x61
DOMAIN_DAS_SAMPLE: Final[int] = 0x70


def _require_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")


def _require_hash_len(name: str, value: bytes) -> None:
    _require_bytes(name, value)
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _domain_byte(domain: int) -> bytes:
    if not isinstance(domain, int):
        raise TypeError("domain must be int")
    if not 0 <= domain <= 0xFF:
        raise ValueError("domain must fit in one byte")
    return bytes((domain,))


def hash_bytes(data: bytes) -> bytes:
    """Return the 32-byte BLAKE3 hash of data."""

    _require_bytes("data", data)
    return blake3(data).digest(length=HASH_LEN_BYTES)


def domain_hash(domain: int, data: bytes) -> bytes:
    """Return BLAKE3(domain_byte || data)."""

    _require_bytes("data", data)
    return hash_bytes(_domain_byte(domain) + data)


def keyed_hash(key: bytes, data: bytes) -> bytes:
    """Return a 32-byte BLAKE3 keyed hash."""

    _require_hash_len("key", key)
    _require_bytes("data", data)
    return blake3(data, key=key).digest(length=HASH_LEN_BYTES)


def derive_key(context: str, key_material: bytes) -> bytes:
    """Derive a 32-byte key with BLAKE3 KDF mode."""

    if not isinstance(context, str):
        raise TypeError("context must be str")
    if context == "":
        raise ValueError("context must not be empty")
    _require_bytes("key_material", key_material)
    return blake3(key_material, derive_key_context=context).digest(length=HASH_LEN_BYTES)
