"""BLAKE3 hashing and Merkle helpers for TensorPoW."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
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


@dataclass(frozen=True)
class MerkleProof:
    """Sparse Merkle inclusion proof for one 32-byte key."""

    key: bytes
    value: bytes
    siblings: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_hash_len("key", self.key)
        _require_hash_len("value", self.value)
        if len(self.siblings) != MERKLE_DEPTH_BITS:
            raise ValueError(f"siblings must contain {MERKLE_DEPTH_BITS} hashes")
        for sibling in self.siblings:
            _require_hash_len("sibling", sibling)


class MerkleTree:
    """Compact sparse Merkle tree over 32-byte keys and values."""

    def __init__(self, items: dict[bytes, bytes] | None = None) -> None:
        self._items: dict[bytes, bytes] = {}
        for key, value in (items or {}).items():
            self.set(key, value)

    @classmethod
    def from_items(cls, items: dict[bytes, bytes]) -> MerkleTree:
        return cls(items)

    def __len__(self) -> int:
        return len(self._items)

    def set(self, key: bytes, value: bytes) -> None:
        _require_hash_len("key", key)
        _require_hash_len("value", value)
        self._items[key] = value

    def delete(self, key: bytes) -> None:
        _require_hash_len("key", key)
        del self._items[key]

    def get(self, key: bytes) -> bytes | None:
        _require_hash_len("key", key)
        return self._items.get(key)

    def root(self) -> bytes:
        return _subtree_root(tuple(sorted(self._items.items())), 0)

    def inclusion_proof(self, key: bytes) -> MerkleProof:
        _require_hash_len("key", key)
        value = self._items.get(key)
        if value is None:
            raise KeyError("key is not present in the Merkle tree")

        siblings = []
        sorted_items = tuple(sorted(self._items.items()))
        for bit_index in range(MERKLE_DEPTH_BITS):
            sibling_bit = 1 - _bit_at(key, bit_index)
            prefix = (*_prefix_bits(key, bit_index), sibling_bit)
            sibling_items = tuple(item for item in sorted_items if _matches_prefix(item[0], prefix))
            siblings.append(_subtree_root(sibling_items, bit_index + 1))

        return MerkleProof(key=key, value=value, siblings=tuple(siblings))

    @staticmethod
    def verify_proof(proof: MerkleProof, root: bytes) -> bool:
        _require_hash_len("root", root)
        node = _leaf_hash(proof.key, proof.value)
        for bit_index in range(MERKLE_DEPTH_BITS - 1, -1, -1):
            sibling = proof.siblings[bit_index]
            if _bit_at(proof.key, bit_index) == 0:
                node = _node_hash(node, sibling)
            else:
                node = _node_hash(sibling, node)
        return node == root


def _leaf_hash(key: bytes, value: bytes) -> bytes:
    return domain_hash(DOMAIN_MERKLE_LEAF, key + value)


def _node_hash(left: bytes, right: bytes) -> bytes:
    return domain_hash(DOMAIN_MERKLE_NODE, left + right)


@cache
def _empty_hash(height: int) -> bytes:
    if not 0 <= height <= MERKLE_DEPTH_BITS:
        raise ValueError("height outside sparse Merkle tree")
    if height == 0:
        return domain_hash(DOMAIN_MERKLE_EMPTY, b"\x00\x00")
    child = _empty_hash(height - 1)
    return _node_hash(child, child)


def _subtree_root(items: tuple[tuple[bytes, bytes], ...], bit_index: int) -> bytes:
    if not items:
        return _empty_hash(MERKLE_DEPTH_BITS - bit_index)
    if bit_index == MERKLE_DEPTH_BITS:
        if len(items) != 1:
            raise ValueError("duplicate Merkle key")
        key, value = items[0]
        return _leaf_hash(key, value)

    left = tuple(item for item in items if _bit_at(item[0], bit_index) == 0)
    right = tuple(item for item in items if _bit_at(item[0], bit_index) == 1)
    return _node_hash(_subtree_root(left, bit_index + 1), _subtree_root(right, bit_index + 1))


def _bit_at(key: bytes, bit_index: int) -> int:
    byte_index, offset = divmod(bit_index, 8)
    return (key[byte_index] >> (7 - offset)) & 1


def _prefix_bits(key: bytes, bit_count: int) -> tuple[int, ...]:
    return tuple(_bit_at(key, bit_index) for bit_index in range(bit_count))


def _matches_prefix(key: bytes, prefix: tuple[int, ...]) -> bool:
    return all(_bit_at(key, bit_index) == bit for bit_index, bit in enumerate(prefix))
