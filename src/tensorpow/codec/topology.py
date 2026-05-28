"""Deterministic compression for anchor topology commitments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tensorpow.chain.blocks import PARENT_CANDIDATE_MAX_COUNT
from tensorpow.chain.merkle import require_hash
from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes

CODEC_ID_BYTES: Final[int] = 2
CODEC_RAW: Final[int] = 0x0000
CODEC_TOPOLOGY: Final[int] = 0x0005

U8_BYTES: Final[int] = 1
U32_BYTES: Final[int] = 4
U32_MAX: Final[int] = 0xFFFFFFFF

COMPRESSED_OBJECT_HEADER_BYTES: Final[int] = CODEC_ID_BYTES + (2 * U32_BYTES)
TOPOLOGY_CODEC_MAGIC: Final[bytes] = b"TPTF"
TOPOLOGY_AFFINE_INT8: Final[int] = 0x01
TOPOLOGY_CODEC_COMPRESSION_PCT: Final[int] = 20
MAX_TOPOLOGY_COMMITMENTS: Final[int] = PARENT_CANDIDATE_MAX_COUNT
MAX_TOPOLOGY_RAW_BYTES: Final[int] = U32_BYTES + (MAX_TOPOLOGY_COMMITMENTS * HASH_LEN_BYTES)
TOPOLOGY_FACTOR_BODY_BYTES: Final[int] = (
    len(TOPOLOGY_CODEC_MAGIC)
    + U8_BYTES
    + U32_BYTES
    + HASH_LEN_BYTES
    + HASH_LEN_BYTES
    + HASH_LEN_BYTES
)
MAX_TOPOLOGY_COMPRESSED_BYTES: Final[int] = COMPRESSED_OBJECT_HEADER_BYTES + MAX_TOPOLOGY_RAW_BYTES


class TopologyCodecError(ValueError):
    """Raised when topology codec bytes are malformed or non-canonical."""


@dataclass(slots=True)
class _Reader:
    data: bytes
    offset: int = 0

    def __post_init__(self) -> None:
        _require_bytes("data", self.data)

    def bytes(self, length: int) -> bytes:
        _require_uint("length", length, U32_MAX)
        end = self.offset + length
        if end > len(self.data):
            raise TopologyCodecError("topology codec bytes are truncated")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def u8(self) -> int:
        return self.bytes(U8_BYTES)[0]

    def u32(self) -> int:
        return int.from_bytes(self.bytes(U32_BYTES), "little")

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise TopologyCodecError("trailing topology codec bytes")


def compress_anchor_topology(parent_candidate_hashes: tuple[bytes, ...]) -> bytes:
    """Return canonical compressed-object bytes for anchor parent candidates."""

    commitments = _require_commitments(parent_candidate_hashes)
    raw = _serialize_commitments(commitments)
    factor_body = _factor_affine_int8(commitments, raw)
    if factor_body is None:
        return _compressed_object(CODEC_RAW, raw, raw)

    raw_object_len = COMPRESSED_OBJECT_HEADER_BYTES + len(raw)
    factor_object_len = COMPRESSED_OBJECT_HEADER_BYTES + len(factor_body)
    if factor_object_len >= raw_object_len:
        return _compressed_object(CODEC_RAW, raw, raw)
    return _compressed_object(CODEC_TOPOLOGY, raw, factor_body)


def decompress_anchor_topology(data: bytes) -> tuple[bytes, ...]:
    """Decode canonical anchor topology compressed-object bytes."""

    _require_bytes("data", data, max_len=MAX_TOPOLOGY_COMPRESSED_BYTES)
    if len(data) < COMPRESSED_OBJECT_HEADER_BYTES:
        raise TopologyCodecError("compressed object header is truncated")

    reader = _Reader(data)
    codec_id = int.from_bytes(reader.bytes(CODEC_ID_BYTES), "little")
    uncompressed_len = reader.u32()
    compressed_len = reader.u32()
    compressed = reader.bytes(compressed_len)
    reader.finish()

    if uncompressed_len > MAX_TOPOLOGY_RAW_BYTES:
        raise TopologyCodecError("uncompressed topology length exceeds maximum")
    if codec_id == CODEC_RAW:
        if compressed_len != uncompressed_len:
            raise TopologyCodecError("raw topology length mismatch")
        raw = compressed
    elif codec_id == CODEC_TOPOLOGY:
        raw = _decode_affine_int8_body(compressed)
        if len(raw) != uncompressed_len:
            raise TopologyCodecError("uncompressed topology length mismatch")
    else:
        raise TopologyCodecError("unsupported topology codec_id")

    commitments = _deserialize_commitments(raw)
    if compress_anchor_topology(commitments) != data:
        raise TopologyCodecError("topology codec bytes are non-canonical")
    return commitments


def _factor_affine_int8(commitments: tuple[bytes, ...], raw: bytes) -> bytes | None:
    if len(commitments) < 2:
        return None

    base = commitments[0]
    slope = bytes(
        (commitments[1][column] - base[column]) & 0xFF for column in range(HASH_LEN_BYTES)
    )
    for row_index, commitment in enumerate(commitments):
        expected = _affine_row(base, slope, row_index)
        if commitment != expected:
            return None

    return b"".join(
        (
            TOPOLOGY_CODEC_MAGIC,
            bytes((TOPOLOGY_AFFINE_INT8,)),
            len(commitments).to_bytes(U32_BYTES, "little"),
            hash_bytes(raw),
            _int8_factor_bytes(base),
            _int8_factor_bytes(slope),
        )
    )


def _decode_affine_int8_body(body: bytes) -> bytes:
    if len(body) != TOPOLOGY_FACTOR_BODY_BYTES:
        raise TopologyCodecError("topology factor body length is invalid")

    reader = _Reader(body)
    if reader.bytes(len(TOPOLOGY_CODEC_MAGIC)) != TOPOLOGY_CODEC_MAGIC:
        raise TopologyCodecError("topology factor magic is invalid")
    factor_kind = reader.u8()
    if factor_kind != TOPOLOGY_AFFINE_INT8:
        raise TopologyCodecError("topology factor kind is invalid")
    count = reader.u32()
    if count > MAX_TOPOLOGY_COMMITMENTS:
        raise TopologyCodecError("topology commitment count exceeds maximum")
    raw_hash = reader.bytes(HASH_LEN_BYTES)
    base = reader.bytes(HASH_LEN_BYTES)
    slope = reader.bytes(HASH_LEN_BYTES)
    reader.finish()

    commitments = tuple(_affine_row(base, slope, row_index) for row_index in range(count))
    raw = _serialize_commitments(commitments)
    if hash_bytes(raw) != raw_hash:
        raise TopologyCodecError("topology factor digest mismatch")
    return raw


def _affine_row(base: bytes, slope: bytes, row_index: int) -> bytes:
    return bytes(
        (_int8_value(base[column]) + (row_index * _int8_value(slope[column]))) & 0xFF
        for column in range(HASH_LEN_BYTES)
    )


def _int8_factor_bytes(values: bytes) -> bytes:
    if len(values) != HASH_LEN_BYTES:
        raise ValueError("INT8 topology factor has invalid width")
    return bytes(value & 0xFF for value in values)


def _int8_value(value: int) -> int:
    return value if value < 0x80 else value - 0x100


def _serialize_commitments(commitments: tuple[bytes, ...]) -> bytes:
    return len(commitments).to_bytes(U32_BYTES, "little") + b"".join(commitments)


def _deserialize_commitments(raw: bytes) -> tuple[bytes, ...]:
    _require_bytes("raw", raw, max_len=MAX_TOPOLOGY_RAW_BYTES)
    reader = _Reader(raw)
    count = reader.u32()
    if count > MAX_TOPOLOGY_COMMITMENTS:
        raise TopologyCodecError("topology commitment count exceeds maximum")
    expected_len = U32_BYTES + (count * HASH_LEN_BYTES)
    if len(raw) != expected_len:
        raise TopologyCodecError("topology raw length is invalid")
    commitments = tuple(reader.bytes(HASH_LEN_BYTES) for _ in range(count))
    reader.finish()
    return _require_commitments(commitments)


def _compressed_object(codec_id: int, raw: bytes, compressed: bytes) -> bytes:
    _require_uint("codec_id", codec_id, 0xFFFF)
    if len(raw) > MAX_TOPOLOGY_RAW_BYTES:
        raise TopologyCodecError("uncompressed topology length exceeds maximum")
    if len(compressed) > U32_MAX:
        raise TopologyCodecError("compressed topology length exceeds maximum")
    return b"".join(
        (
            codec_id.to_bytes(CODEC_ID_BYTES, "little"),
            len(raw).to_bytes(U32_BYTES, "little"),
            len(compressed).to_bytes(U32_BYTES, "little"),
            compressed,
        )
    )


def _require_commitments(commitments: tuple[bytes, ...]) -> tuple[bytes, ...]:
    if not isinstance(commitments, tuple):
        raise TypeError("parent_candidate_hashes must be a tuple")
    if len(commitments) > MAX_TOPOLOGY_COMMITMENTS:
        raise ValueError("too many topology commitments")
    checked = tuple(require_hash("topology_commitment", commitment) for commitment in commitments)
    if len(set(checked)) != len(checked):
        raise ValueError("duplicate topology commitments")
    return checked


def _require_bytes(name: str, value: bytes, *, max_len: int | None = None) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if max_len is not None and len(value) > max_len:
        raise ValueError(f"{name} exceeds max length")


def _require_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} outside uint range")


__all__ = [
    "CODEC_TOPOLOGY",
    "MAX_TOPOLOGY_COMMITMENTS",
    "MAX_TOPOLOGY_COMPRESSED_BYTES",
    "MAX_TOPOLOGY_RAW_BYTES",
    "TOPOLOGY_AFFINE_INT8",
    "TOPOLOGY_CODEC_COMPRESSION_PCT",
    "TOPOLOGY_CODEC_MAGIC",
    "TopologyCodecError",
    "compress_anchor_topology",
    "decompress_anchor_topology",
]
