"""Canonical fruit and anchor header structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tensorpow.chain.merkle import require_hash
from tensorpow.crypto.hash import (
    DOMAIN_ANCHOR_HEADER,
    DOMAIN_FRUIT_HEADER,
    HASH_LEN_BYTES,
    domain_hash,
)
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.challenge import (
    FORMAT_EPOCH,
    U16_MAX,
    U32_MAX,
    U64_MAX,
    AnchorPowHeader,
    FruitPowHeader,
)

U16_BYTES: Final[int] = 2
U32_BYTES: Final[int] = 4
U64_BYTES: Final[int] = 8
PARENT_BITMAP_MAX_BYTES: Final[int] = 1250
SHARD_MAX_DEPTH: Final[int] = 16
SHARD_ID_DEPTH_SHIFT: Final[int] = 16


class HeaderDecodeError(ValueError):
    """Raised when header bytes are malformed or non-canonical."""


@dataclass(frozen=True, slots=True)
class FruitHeader:
    """Canonical fruit header."""

    version: int
    sig_type_supported: int
    parent_selected: bytes
    parent_bitmap: bytes
    latest_anchor: bytes
    tx_merkle_root: bytes
    timestamp_ms: int
    shard_id: int
    nonce: int

    def __post_init__(self) -> None:
        _require_format_epoch(self.version)
        _require_sig_mask(self.sig_type_supported)
        require_hash("parent_selected", self.parent_selected)
        _require_bytes("parent_bitmap", self.parent_bitmap, max_len=PARENT_BITMAP_MAX_BYTES)
        require_hash("latest_anchor", self.latest_anchor)
        require_hash("tx_merkle_root", self.tx_merkle_root)
        _require_uint("timestamp_ms", self.timestamp_ms, U64_MAX)
        _require_shard_id(self.shard_id)
        _require_uint("nonce", self.nonce, U64_MAX)

    def serialize(self) -> bytes:
        return b"".join(
            (
                _u16(self.version),
                _u16(self.sig_type_supported),
                self.parent_selected,
                _u16(len(self.parent_bitmap)),
                self.parent_bitmap,
                self.latest_anchor,
                self.tx_merkle_root,
                _u64(self.timestamp_ms),
                _u32(self.shard_id),
                _u64(self.nonce),
            )
        )

    @classmethod
    def deserialize(cls, data: bytes) -> FruitHeader:
        reader = _Reader(data)
        version = reader.u16()
        sig_type_supported = reader.u16()
        parent_selected = reader.bytes(HASH_LEN_BYTES)
        parent_bitmap = reader.bytes(reader.u16())
        latest_anchor = reader.bytes(HASH_LEN_BYTES)
        tx_merkle_root = reader.bytes(HASH_LEN_BYTES)
        timestamp_ms = reader.u64()
        shard_id = reader.u32()
        nonce = reader.u64()
        reader.finish()
        try:
            return cls(
                version=version,
                sig_type_supported=sig_type_supported,
                parent_selected=parent_selected,
                parent_bitmap=parent_bitmap,
                latest_anchor=latest_anchor,
                tx_merkle_root=tx_merkle_root,
                timestamp_ms=timestamp_ms,
                shard_id=shard_id,
                nonce=nonce,
            )
        except (TypeError, ValueError) as exc:
            raise HeaderDecodeError(str(exc)) from exc

    def header_hash(self) -> bytes:
        return domain_hash(DOMAIN_FRUIT_HEADER, self.serialize())

    def effective_parent_hashes(self, parent_candidates: tuple[bytes, ...]) -> tuple[bytes, ...]:
        for parent_hash in parent_candidates:
            require_hash("parent_candidate", parent_hash)
        selected = [self.parent_selected]
        for index, candidate in enumerate(parent_candidates):
            if index // 8 < len(self.parent_bitmap) and (
                self.parent_bitmap[index // 8] & (1 << (index % 8))
            ):
                selected.append(candidate)
        if _has_bits_beyond(self.parent_bitmap, len(parent_candidates)):
            raise ValueError("parent_bitmap has bits beyond parent candidate list")
        if len(set(selected)) != len(selected):
            raise ValueError("effective parent set contains duplicates")
        return tuple(selected)

    def to_pow_header(self, parent_candidates: tuple[bytes, ...]) -> FruitPowHeader:
        return FruitPowHeader(
            version=self.version,
            sig_type_supported=self.sig_type_supported,
            effective_parent_hashes=self.effective_parent_hashes(parent_candidates),
            latest_anchor=self.latest_anchor,
            tx_merkle_root=self.tx_merkle_root,
            timestamp_ms=self.timestamp_ms,
            shard_id=self.shard_id,
            nonce=self.nonce,
        )


@dataclass(frozen=True, slots=True)
class AnchorHeader:
    """Canonical anchor header."""

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
        _require_format_epoch(self.version)
        require_hash("parent_anchor", self.parent_anchor)
        require_hash("fruit_set_root", self.fruit_set_root)
        require_hash("parent_candidate_root", self.parent_candidate_root)
        require_hash("shard_tree_state_root", self.shard_tree_state_root)
        require_hash("fee_floor_set_root", self.fee_floor_set_root)
        require_hash("anchor_reward_root", self.anchor_reward_root)
        _require_uint("timestamp_ms", self.timestamp_ms, U64_MAX)
        _require_uint("nonce", self.nonce, U64_MAX)

    def serialize(self) -> bytes:
        return b"".join(
            (
                _u16(self.version),
                self.parent_anchor,
                self.fruit_set_root,
                self.parent_candidate_root,
                self.shard_tree_state_root,
                self.fee_floor_set_root,
                self.anchor_reward_root,
                _u64(self.timestamp_ms),
                _u64(self.nonce),
            )
        )

    @classmethod
    def deserialize(cls, data: bytes) -> AnchorHeader:
        reader = _Reader(data)
        version = reader.u16()
        parent_anchor = reader.bytes(HASH_LEN_BYTES)
        fruit_set_root = reader.bytes(HASH_LEN_BYTES)
        parent_candidate_root = reader.bytes(HASH_LEN_BYTES)
        shard_tree_state_root = reader.bytes(HASH_LEN_BYTES)
        fee_floor_set_root = reader.bytes(HASH_LEN_BYTES)
        anchor_reward_root = reader.bytes(HASH_LEN_BYTES)
        timestamp_ms = reader.u64()
        nonce = reader.u64()
        reader.finish()
        try:
            return cls(
                version=version,
                parent_anchor=parent_anchor,
                fruit_set_root=fruit_set_root,
                parent_candidate_root=parent_candidate_root,
                shard_tree_state_root=shard_tree_state_root,
                fee_floor_set_root=fee_floor_set_root,
                anchor_reward_root=anchor_reward_root,
                timestamp_ms=timestamp_ms,
                nonce=nonce,
            )
        except (TypeError, ValueError) as exc:
            raise HeaderDecodeError(str(exc)) from exc

    def header_hash(self) -> bytes:
        return domain_hash(DOMAIN_ANCHOR_HEADER, self.serialize())

    def to_pow_header(self) -> AnchorPowHeader:
        return AnchorPowHeader(
            version=self.version,
            parent_anchor=self.parent_anchor,
            fruit_set_root=self.fruit_set_root,
            parent_candidate_root=self.parent_candidate_root,
            shard_tree_state_root=self.shard_tree_state_root,
            fee_floor_set_root=self.fee_floor_set_root,
            anchor_reward_root=self.anchor_reward_root,
            timestamp_ms=self.timestamp_ms,
            nonce=self.nonce,
        )


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = _require_bytes("data", data)
        self._offset = 0

    def bytes(self, length: int) -> bytes:
        _require_uint("length", length, U32_MAX)
        end = self._offset + length
        if end > len(self._data):
            raise HeaderDecodeError("truncated header")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def u16(self) -> int:
        return int.from_bytes(self.bytes(U16_BYTES), "little")

    def u32(self) -> int:
        return int.from_bytes(self.bytes(U32_BYTES), "little")

    def u64(self) -> int:
        return int.from_bytes(self.bytes(U64_BYTES), "little")

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise HeaderDecodeError("trailing header bytes")


def _require_format_epoch(value: int) -> None:
    _require_uint("version", value, U16_MAX)
    if value != FORMAT_EPOCH:
        raise ValueError("version must equal FORMAT_EPOCH")


def _require_sig_mask(value: int) -> None:
    _require_uint("sig_type_supported", value, U16_MAX)
    if value != SIG_TYPE_ED25519_BIT:
        raise ValueError("sig_type_supported must contain only the active Ed25519 bit")


def _require_shard_id(value: int) -> None:
    _require_uint("shard_id", value, U32_MAX)
    depth = value >> SHARD_ID_DEPTH_SHIFT
    path = value & ((1 << SHARD_ID_DEPTH_SHIFT) - 1)
    if depth > SHARD_MAX_DEPTH:
        raise ValueError("shard_id depth exceeds SHARD_MAX_DEPTH")
    if path >= (1 << depth):
        raise ValueError("shard_id path is outside depth")


def _has_bits_beyond(bitmap: bytes, candidate_count: int) -> bool:
    for bit_index in range(candidate_count, len(bitmap) * 8):
        if bitmap[bit_index // 8] & (1 << (bit_index % 8)):
            return True
    return False


def _require_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} outside uint range")


def _require_bytes(name: str, value: bytes, *, max_len: int | None = None) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if max_len is not None and len(value) > max_len:
        raise ValueError(f"{name} exceeds max length")
    return value


def _u16(value: int) -> bytes:
    return value.to_bytes(U16_BYTES, "little")


def _u32(value: int) -> bytes:
    return value.to_bytes(U32_BYTES, "little")


def _u64(value: int) -> bytes:
    return value.to_bytes(U64_BYTES, "little")
