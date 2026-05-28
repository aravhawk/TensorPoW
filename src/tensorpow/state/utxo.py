"""UTXO state and sparse Merkle commitments."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Final

from tensorpow.crypto.hash import (
    DOMAIN_MERKLE_EMPTY,
    DOMAIN_MERKLE_LEAF,
    DOMAIN_MERKLE_NODE,
    DOMAIN_OUTPOINT,
    DOMAIN_UTXO,
    HASH_LEN_BYTES,
    MERKLE_DEPTH_BITS,
    domain_hash,
)

U16_BYTES: Final[int] = 2
U32_BYTES: Final[int] = 4
U64_BYTES: Final[int] = 8
U16_MAX: Final[int] = 0xFFFF
U32_MAX: Final[int] = 0xFFFFFFFF
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF

OUTPOINT_BYTES: Final[int] = HASH_LEN_BYTES + U32_BYTES
MAX_SUPPLY_MATOMS: Final[int] = 2_100_000_000_000_000
TX_OUTPUT_PAYLOAD_MAX_BYTES: Final[int] = 2048

TEMPLATE_PKH: Final[int] = 0
TEMPLATE_MULTISIG: Final[int] = 1
TEMPLATE_HASHLOCK: Final[int] = 2
ACTIVE_TEMPLATES: Final[frozenset[int]] = frozenset(
    (TEMPLATE_PKH, TEMPLATE_MULTISIG, TEMPLATE_HASHLOCK)
)

UTXO_FIXED_BYTES: Final[int] = (
    OUTPOINT_BYTES + U64_BYTES + U16_BYTES + HASH_LEN_BYTES + U64_BYTES + U64_BYTES + U16_BYTES
)
KEY_SPACE_BITS: Final[int] = HASH_LEN_BYTES * 8
KEY_SPACE_SIZE: Final[int] = 1 << KEY_SPACE_BITS

type _MerkleItem = tuple[int, bytes, bytes]


@dataclass(frozen=True, slots=True)
class Outpoint:
    """Transaction output reference."""

    tx_id: bytes
    output_index: int

    def __post_init__(self) -> None:
        _require_hash("tx_id", self.tx_id)
        _require_u32("output_index", self.output_index)

    @classmethod
    def from_bytes(cls, data: bytes) -> Outpoint:
        """Decode canonical `tx_id || output_index_le` bytes."""

        _require_bytes("data", data)
        if len(data) != OUTPOINT_BYTES:
            raise ValueError(f"outpoint must be {OUTPOINT_BYTES} bytes")
        return cls(
            tx_id=data[:HASH_LEN_BYTES],
            output_index=int.from_bytes(data[HASH_LEN_BYTES:], "little"),
        )

    def to_bytes(self) -> bytes:
        """Encode as `tx_id || output_index_le`."""

        return self.tx_id + self.output_index.to_bytes(U32_BYTES, "little")

    def key(self) -> bytes:
        """Return the section-8 sparse Merkle key for this outpoint."""

        return domain_hash(DOMAIN_OUTPOINT, self.to_bytes())


@dataclass(frozen=True, slots=True)
class UTXO:
    """Unspent output payload committed into the UTXO set."""

    outpoint: Outpoint
    amount_matoms: int
    template_id: int
    owner_pubkey_hash: bytes
    locktime_ms: int = 0
    lockheight: int = 0
    payload: bytes = b""

    def __post_init__(self) -> None:
        if not isinstance(self.outpoint, Outpoint):
            raise TypeError("outpoint must be Outpoint")
        _require_u64("amount_matoms", self.amount_matoms)
        if self.amount_matoms == 0:
            raise ValueError("amount_matoms must be nonzero")
        if self.amount_matoms > MAX_SUPPLY_MATOMS:
            raise ValueError("amount_matoms exceeds MAX_SUPPLY_MATOMS")
        _require_u16("template_id", self.template_id)
        if self.template_id not in ACTIVE_TEMPLATES:
            raise ValueError("template_id must be an active output template")
        _require_hash("owner_pubkey_hash", self.owner_pubkey_hash)
        _require_u64("locktime_ms", self.locktime_ms)
        _require_u64("lockheight", self.lockheight)
        _require_bytes("payload", self.payload)
        if len(self.payload) > TX_OUTPUT_PAYLOAD_MAX_BYTES:
            raise ValueError(f"payload must be <= {TX_OUTPUT_PAYLOAD_MAX_BYTES} bytes")

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_outpoint_key: bytes | None = None) -> UTXO:
        """Decode canonical UTXO bytes, optionally checking the storage key."""

        _require_bytes("data", data)
        if len(data) < UTXO_FIXED_BYTES:
            raise ValueError("UTXO bytes are truncated")

        offset = 0
        outpoint = Outpoint.from_bytes(data[offset : offset + OUTPOINT_BYTES])
        offset += OUTPOINT_BYTES
        amount_matoms = int.from_bytes(data[offset : offset + U64_BYTES], "little")
        offset += U64_BYTES
        template_id = int.from_bytes(data[offset : offset + U16_BYTES], "little")
        offset += U16_BYTES
        owner_pubkey_hash = data[offset : offset + HASH_LEN_BYTES]
        offset += HASH_LEN_BYTES
        locktime_ms = int.from_bytes(data[offset : offset + U64_BYTES], "little")
        offset += U64_BYTES
        lockheight = int.from_bytes(data[offset : offset + U64_BYTES], "little")
        offset += U64_BYTES
        payload_len = int.from_bytes(data[offset : offset + U16_BYTES], "little")
        offset += U16_BYTES

        expected_len = UTXO_FIXED_BYTES + payload_len
        if len(data) != expected_len:
            raise ValueError("UTXO payload length does not match encoded bytes")

        utxo = cls(
            outpoint=outpoint,
            amount_matoms=amount_matoms,
            template_id=template_id,
            owner_pubkey_hash=owner_pubkey_hash,
            locktime_ms=locktime_ms,
            lockheight=lockheight,
            payload=data[offset:],
        )
        if expected_outpoint_key is not None:
            _require_hash("expected_outpoint_key", expected_outpoint_key)
            if utxo.outpoint_key() != expected_outpoint_key:
                raise ValueError("UTXO outpoint does not match expected key")
        return utxo

    def to_bytes(self) -> bytes:
        """Encode the section-8 UTXO byte layout."""

        payload_len = len(self.payload)
        return b"".join(
            (
                self.outpoint.to_bytes(),
                self.amount_matoms.to_bytes(U64_BYTES, "little"),
                self.template_id.to_bytes(U16_BYTES, "little"),
                self.owner_pubkey_hash,
                self.locktime_ms.to_bytes(U64_BYTES, "little"),
                self.lockheight.to_bytes(U64_BYTES, "little"),
                payload_len.to_bytes(U16_BYTES, "little"),
                self.payload,
            )
        )

    def outpoint_key(self) -> bytes:
        """Return the sparse Merkle key for this UTXO."""

        return self.outpoint.key()

    def value_hash(self) -> bytes:
        """Return `BLAKE3(DOMAIN_UTXO || utxo_bytes)`."""

        return domain_hash(DOMAIN_UTXO, self.to_bytes())


@dataclass(frozen=True, slots=True)
class UTXOInclusionProof:
    """Sparse Merkle proof that an outpoint is present."""

    outpoint: Outpoint
    utxo: UTXO
    siblings: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_outpoint("outpoint", self.outpoint)
        if not isinstance(self.utxo, UTXO):
            raise TypeError("utxo must be UTXO")
        if self.utxo.outpoint != self.outpoint:
            raise ValueError("proof UTXO must match proof outpoint")
        _require_siblings(self.siblings, expected_len=MERKLE_DEPTH_BITS)


@dataclass(frozen=True, slots=True)
class UTXONonInclusionProof:
    """Sparse Merkle proof that an outpoint is absent."""

    outpoint: Outpoint
    empty_depth: int
    siblings: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_outpoint("outpoint", self.outpoint)
        _require_u16("empty_depth", self.empty_depth)
        if self.empty_depth > MERKLE_DEPTH_BITS:
            raise ValueError("empty_depth outside sparse Merkle tree")
        _require_siblings(self.siblings, expected_len=self.empty_depth)


type UTXOProof = UTXOInclusionProof | UTXONonInclusionProof


class UTXOSet:
    """In-memory UTXO set with a section-8 sparse Merkle commitment."""

    def __init__(self, utxos: Iterable[UTXO] | None = None) -> None:
        self._utxos: dict[Outpoint, UTXO] = {}
        self._outpoint_by_key: dict[bytes, Outpoint] = {}
        self._items_cache: tuple[_MerkleItem, ...] | None = None
        self._key_ints_cache: tuple[int, ...] | None = None
        self._subtree_cache: dict[tuple[int, int], bytes] = {}
        for utxo in utxos or ():
            self.add(utxo)

    def __len__(self) -> int:
        return len(self._utxos)

    def __contains__(self, outpoint: object) -> bool:
        return isinstance(outpoint, Outpoint) and self.contains(outpoint)

    def add(self, utxo: UTXO) -> None:
        """Add one unspent output, rejecting duplicate outpoints."""

        if not isinstance(utxo, UTXO):
            raise TypeError("utxo must be UTXO")
        if utxo.outpoint in self._utxos:
            raise KeyError("outpoint already exists in UTXO set")
        outpoint_key = utxo.outpoint_key()
        if outpoint_key in self._outpoint_by_key:
            raise ValueError("duplicate UTXO Merkle key")
        self._utxos[utxo.outpoint] = utxo
        self._outpoint_by_key[outpoint_key] = utxo.outpoint
        self._invalidate_merkle_cache()

    def remove(self, outpoint: Outpoint) -> UTXO:
        """Remove and return one unspent output."""

        _require_outpoint("outpoint", outpoint)
        try:
            utxo = self._utxos.pop(outpoint)
        except KeyError as exc:
            raise KeyError("outpoint is not present in UTXO set") from exc
        del self._outpoint_by_key[utxo.outpoint_key()]
        self._invalidate_merkle_cache()
        return utxo

    def get(self, outpoint: Outpoint) -> UTXO | None:
        """Return the UTXO for an outpoint, if present."""

        _require_outpoint("outpoint", outpoint)
        return self._utxos.get(outpoint)

    def contains(self, outpoint: Outpoint) -> bool:
        """Return True when the outpoint is unspent."""

        _require_outpoint("outpoint", outpoint)
        return outpoint in self._utxos

    def utxos(self) -> tuple[UTXO, ...]:
        """Return all UTXOs in canonical outpoint order."""

        return tuple(sorted(self._utxos.values(), key=lambda utxo: utxo.outpoint.to_bytes()))

    def merkle_root(self) -> bytes:
        """Return the compact sparse Merkle root of the current UTXO set."""

        return _subtree_root(
            self._merkle_items(),
            self._merkle_key_ints(),
            0,
            0,
            self._subtree_cache,
        )

    def inclusion_proof(self, outpoint: Outpoint) -> UTXOInclusionProof:
        """Build an inclusion proof for an unspent outpoint."""

        _require_outpoint("outpoint", outpoint)
        utxo = self._utxos.get(outpoint)
        if utxo is None:
            raise KeyError("outpoint is not present in UTXO set")
        return UTXOInclusionProof(
            outpoint=outpoint,
            utxo=utxo,
            siblings=_proof_siblings(
                outpoint.key(),
                self._merkle_items(),
                self._merkle_key_ints(),
                self._subtree_cache,
            ),
        )

    def non_inclusion_proof(self, outpoint: Outpoint) -> UTXONonInclusionProof:
        """Build a proof that an outpoint is absent from the set."""

        _require_outpoint("outpoint", outpoint)
        if outpoint in self._utxos:
            raise KeyError("outpoint is present in UTXO set")
        items = self._merkle_items()
        key_ints = self._merkle_key_ints()
        empty_depth = _first_empty_depth(outpoint.key(), key_ints)
        return UTXONonInclusionProof(
            outpoint=outpoint,
            empty_depth=empty_depth,
            siblings=_proof_siblings(
                outpoint.key(),
                items,
                key_ints,
                self._subtree_cache,
                depth_limit=empty_depth,
            ),
        )

    @staticmethod
    def verify_proof(outpoint: Outpoint, proof: UTXOProof, root: bytes) -> bool:
        """Return True when a UTXO inclusion or absence proof matches a root."""

        _require_outpoint("outpoint", outpoint)
        _require_hash("root", root)
        if not isinstance(proof, UTXOInclusionProof | UTXONonInclusionProof):
            raise TypeError("proof must be UTXOInclusionProof or UTXONonInclusionProof")
        if proof.outpoint != outpoint:
            return False

        try:
            if isinstance(proof, UTXOInclusionProof):
                _require_siblings(proof.siblings, expected_len=MERKLE_DEPTH_BITS)
            else:
                _require_siblings(proof.siblings, expected_len=proof.empty_depth)
        except (TypeError, ValueError):
            return False

        key = outpoint.key()
        if isinstance(proof, UTXOInclusionProof):
            if proof.utxo.outpoint != outpoint:
                return False
            node = _leaf_hash(key, proof.utxo.value_hash())
            start_depth = MERKLE_DEPTH_BITS
        else:
            if not 0 <= proof.empty_depth <= MERKLE_DEPTH_BITS:
                return False
            node = _empty_hash(proof.empty_depth)
            start_depth = proof.empty_depth

        for bit_index in range(start_depth - 1, -1, -1):
            sibling = proof.siblings[bit_index]
            if _bit_at(key, bit_index) == 0:
                node = _node_hash(bit_index, node, sibling)
            else:
                node = _node_hash(bit_index, sibling, node)
        return node == root

    def _merkle_items(self) -> tuple[_MerkleItem, ...]:
        cached = self._items_cache
        if cached is None:
            cached = tuple(
                sorted(
                    (
                        int.from_bytes(outpoint_key, "big"),
                        outpoint_key,
                        utxo.value_hash(),
                    )
                    for utxo in self._utxos.values()
                    for outpoint_key in (utxo.outpoint_key(),)
                )
            )
            self._items_cache = cached
        return cached

    def _merkle_key_ints(self) -> tuple[int, ...]:
        cached = self._key_ints_cache
        if cached is None:
            cached = tuple(item[0] for item in self._merkle_items())
            self._key_ints_cache = cached
        return cached

    def _invalidate_merkle_cache(self) -> None:
        self._items_cache = None
        self._key_ints_cache = None
        self._subtree_cache.clear()


def _require_outpoint(name: str, value: Outpoint) -> None:
    if not isinstance(value, Outpoint):
        raise TypeError(f"{name} must be Outpoint")


def _require_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")


def _require_hash(name: str, value: bytes) -> None:
    _require_bytes(name, value)
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _require_u16(name: str, value: int) -> None:
    _require_uint(name, value, U16_MAX)


def _require_u32(name: str, value: int) -> None:
    _require_uint(name, value, U32_MAX)


def _require_u64(name: str, value: int) -> None:
    _require_uint(name, value, U64_MAX)


def _require_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} outside uint range")


def _require_siblings(siblings: tuple[bytes, ...], *, expected_len: int) -> None:
    if not isinstance(siblings, tuple):
        raise TypeError("siblings must be a tuple")
    if len(siblings) != expected_len:
        raise ValueError(f"siblings must contain {expected_len} hashes")
    for sibling in siblings:
        _require_hash("sibling", sibling)
    if len(set(siblings)) != len(siblings):
        raise ValueError("siblings must not contain duplicates")


def _leaf_hash(key: bytes, value_hash: bytes) -> bytes:
    _require_hash("key", key)
    _require_hash("value_hash", value_hash)
    return domain_hash(DOMAIN_MERKLE_LEAF, key + value_hash)


def _node_hash(depth: int, left: bytes, right: bytes) -> bytes:
    _require_u16("depth", depth)
    _require_hash("left", left)
    _require_hash("right", right)
    return domain_hash(DOMAIN_MERKLE_NODE, depth.to_bytes(U16_BYTES, "little") + left + right)


@cache
def _empty_hash(depth: int) -> bytes:
    _require_u16("depth", depth)
    if depth > MERKLE_DEPTH_BITS:
        raise ValueError("depth outside sparse Merkle tree")
    return domain_hash(DOMAIN_MERKLE_EMPTY, depth.to_bytes(U16_BYTES, "little"))


def _subtree_root(
    items: tuple[_MerkleItem, ...],
    key_ints: tuple[int, ...],
    depth: int,
    prefix: int,
    subtree_cache: dict[tuple[int, int], bytes],
) -> bytes:
    cache_key = (depth, prefix)
    cached = subtree_cache.get(cache_key)
    if cached is not None:
        return cached

    start, end = _range_for_prefix(key_ints, depth, prefix)
    if start == end:
        return _empty_hash(depth)
    if depth == MERKLE_DEPTH_BITS:
        if end - start != 1:
            raise ValueError("duplicate UTXO Merkle key")
        _, key, value_hash = items[start]
        node = _leaf_hash(key, value_hash)
        subtree_cache[cache_key] = node
        return node

    node = _node_hash(
        depth,
        _subtree_root(items, key_ints, depth + 1, prefix << 1, subtree_cache),
        _subtree_root(items, key_ints, depth + 1, (prefix << 1) | 1, subtree_cache),
    )
    subtree_cache[cache_key] = node
    return node


def _proof_siblings(
    key: bytes,
    items: tuple[_MerkleItem, ...],
    key_ints: tuple[int, ...],
    subtree_cache: dict[tuple[int, int], bytes],
    *,
    depth_limit: int = MERKLE_DEPTH_BITS,
) -> tuple[bytes, ...]:
    _require_hash("key", key)
    _require_u16("depth_limit", depth_limit)
    if depth_limit > MERKLE_DEPTH_BITS:
        raise ValueError("depth_limit outside sparse Merkle tree")
    siblings: list[bytes] = []
    key_int = int.from_bytes(key, "big")
    for bit_index in range(depth_limit):
        prefix = (_prefix_int(key_int, bit_index) << 1) | (1 - _bit_at(key, bit_index))
        siblings.append(_subtree_root(items, key_ints, bit_index + 1, prefix, subtree_cache))
    return tuple(siblings)


def _first_empty_depth(key: bytes, key_ints: tuple[int, ...]) -> int:
    _require_hash("key", key)
    key_int = int.from_bytes(key, "big")
    for depth in range(MERKLE_DEPTH_BITS + 1):
        prefix = _prefix_int(key_int, depth)
        start, end = _range_for_prefix(key_ints, depth, prefix)
        if start == end:
            return depth
    raise ValueError("key is already present in sparse Merkle tree")


def _range_for_prefix(key_ints: tuple[int, ...], depth: int, prefix: int) -> tuple[int, int]:
    _require_u16("depth", depth)
    if depth > MERKLE_DEPTH_BITS:
        raise ValueError("depth outside sparse Merkle tree")
    if not isinstance(prefix, int):
        raise TypeError("prefix must be int")
    if not 0 <= prefix < (1 << depth):
        raise ValueError("prefix outside sparse Merkle tree")

    shift = KEY_SPACE_BITS - depth
    lower = prefix << shift
    upper = KEY_SPACE_SIZE if depth == 0 else (prefix + 1) << shift
    start = bisect_left(key_ints, lower)
    end = bisect_left(key_ints, upper, start)
    return start, end


def _prefix_int(key_int: int, bit_count: int) -> int:
    _require_u16("bit_count", bit_count)
    if bit_count > MERKLE_DEPTH_BITS:
        raise ValueError("bit_count outside sparse Merkle tree")
    return key_int >> (KEY_SPACE_BITS - bit_count) if bit_count else 0


def _bit_at(key: bytes, bit_index: int) -> int:
    return (key[bit_index // 8] >> (7 - (bit_index % 8))) & 1


__all__ = [
    "ACTIVE_TEMPLATES",
    "MAX_SUPPLY_MATOMS",
    "OUTPOINT_BYTES",
    "TEMPLATE_HASHLOCK",
    "TEMPLATE_MULTISIG",
    "TEMPLATE_PKH",
    "TX_OUTPUT_PAYLOAD_MAX_BYTES",
    "UTXO",
    "UTXO_FIXED_BYTES",
    "Outpoint",
    "UTXOInclusionProof",
    "UTXONonInclusionProof",
    "UTXOProof",
    "UTXOSet",
]
