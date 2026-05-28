"""Tests for deterministic anchor topology compression."""

from __future__ import annotations

import pytest

from tensorpow.codec.topology import (
    CODEC_RAW,
    CODEC_TOPOLOGY,
    COMPRESSED_OBJECT_HEADER_BYTES,
    TOPOLOGY_CODEC_COMPRESSION_PCT,
    TopologyCodecError,
    compress_anchor_topology,
    decompress_anchor_topology,
)
from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes


def test_topology_codec_round_trips_factorized_fixture_and_meets_ratio() -> None:
    commitments = _affine_commitments(96)
    raw = _raw_topology_bytes(commitments)

    encoded = compress_anchor_topology(commitments)
    raw_object_len = COMPRESSED_OBJECT_HEADER_BYTES + len(raw)
    compression_pct = (raw_object_len - len(encoded)) * 100 // raw_object_len

    assert encoded[:2] == CODEC_TOPOLOGY.to_bytes(2, "little")
    assert int.from_bytes(encoded[2:6], "little") == len(raw)
    assert int.from_bytes(encoded[6:10], "little") == len(encoded) - COMPRESSED_OBJECT_HEADER_BYTES
    assert decompress_anchor_topology(encoded) == commitments
    assert compress_anchor_topology(decompress_anchor_topology(encoded)) == encoded
    assert compression_pct >= TOPOLOGY_CODEC_COMPRESSION_PCT


def test_topology_codec_is_deterministic() -> None:
    commitments = _affine_commitments(64)

    first = compress_anchor_topology(commitments)
    second = compress_anchor_topology(commitments)
    third = compress_anchor_topology(decompress_anchor_topology(first))

    assert first == second == third


def test_topology_codec_falls_back_to_raw_when_factorization_is_not_beneficial() -> None:
    non_factorable = tuple(hash_bytes(seed.to_bytes(4, "little")) for seed in range(12))
    tiny_affine = _affine_commitments(3)

    non_factorable_encoded = compress_anchor_topology(non_factorable)
    tiny_affine_encoded = compress_anchor_topology(tiny_affine)

    assert non_factorable_encoded[:2] == CODEC_RAW.to_bytes(2, "little")
    assert non_factorable_encoded[COMPRESSED_OBJECT_HEADER_BYTES:] == _raw_topology_bytes(
        non_factorable
    )
    assert decompress_anchor_topology(non_factorable_encoded) == non_factorable
    assert tiny_affine_encoded[:2] == CODEC_RAW.to_bytes(2, "little")
    assert decompress_anchor_topology(tiny_affine_encoded) == tiny_affine


def test_topology_codec_rejects_invalid_commitments() -> None:
    commitments = _affine_commitments(2)

    with pytest.raises(TypeError, match="tuple"):
        compress_anchor_topology(list(commitments))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=f"{HASH_LEN_BYTES} bytes"):
        compress_anchor_topology((b"short",))
    with pytest.raises(ValueError, match="duplicate"):
        compress_anchor_topology((commitments[0], commitments[0]))


def test_topology_codec_rejects_malformed_and_corrupt_inputs() -> None:
    commitments = _affine_commitments(32)
    encoded = compress_anchor_topology(commitments)
    body_start = COMPRESSED_OBJECT_HEADER_BYTES
    compressed_len = len(encoded) - COMPRESSED_OBJECT_HEADER_BYTES

    malformed = [
        encoded[:5],
        encoded + b"\x00",
        _replace(encoded, 0, b"\x04\x00"),
        _replace(encoded, 2, (len(_raw_topology_bytes(commitments)) + 1).to_bytes(4, "little")),
        encoded[:6] + (compressed_len - 1).to_bytes(4, "little") + encoded[body_start:-1],
        _replace(encoded, body_start, b"X"),
        _replace(encoded, body_start + 4, b"\xff"),
        _replace(encoded, len(encoded) - 1, bytes((encoded[-1] ^ 0x01,))),
    ]

    for raw in malformed:
        with pytest.raises(TopologyCodecError):
            decompress_anchor_topology(raw)


def test_topology_codec_rejects_noncanonical_raw_when_factorization_wins() -> None:
    commitments = _affine_commitments(32)

    with pytest.raises(TopologyCodecError, match="non-canonical"):
        decompress_anchor_topology(_raw_object(commitments))


def _affine_commitments(count: int) -> tuple[bytes, ...]:
    base = bytes((17 + (column * 7)) & 0xFF for column in range(HASH_LEN_BYTES))
    slope = bytes((3 + (column * 5)) & 0xFF for column in range(HASH_LEN_BYTES))
    return tuple(
        bytes((base[column] + (row * slope[column])) & 0xFF for column in range(HASH_LEN_BYTES))
        for row in range(count)
    )


def _raw_topology_bytes(commitments: tuple[bytes, ...]) -> bytes:
    return len(commitments).to_bytes(4, "little") + b"".join(commitments)


def _raw_object(commitments: tuple[bytes, ...]) -> bytes:
    raw = _raw_topology_bytes(commitments)
    return b"".join(
        (
            CODEC_RAW.to_bytes(2, "little"),
            len(raw).to_bytes(4, "little"),
            len(raw).to_bytes(4, "little"),
            raw,
        )
    )


def _replace(data: bytes, offset: int, replacement: bytes) -> bytes:
    return data[:offset] + replacement + data[offset + len(replacement) :]
