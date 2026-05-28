"""Tests for deterministic PoW kernel helpers."""

from __future__ import annotations

import pytest
import torch

from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.pow.kernel import (
    ANCHOR_INITIAL_TARGET_LE,
    ANCHOR_WORK_MULTIPLIER,
    FRUIT_TARGET_LE,
    FRUIT_WORK_OPS,
    POW_MATRIX_DIM,
    POW_OPS_PER_MATMUL,
    PowInputError,
    canonical_output_bytes,
    matmul_int8,
    pow_digest,
    target_allows_digest,
)


def test_pow_difficulty_constants_match_spec_formula() -> None:
    fruit_target = ((1 << (8 * HASH_LEN_BYTES)) - 1) * POW_OPS_PER_MATMUL // FRUIT_WORK_OPS
    anchor_target = fruit_target // ANCHOR_WORK_MULTIPLIER

    assert fruit_target.to_bytes(HASH_LEN_BYTES, "little") == FRUIT_TARGET_LE
    assert anchor_target.to_bytes(HASH_LEN_BYTES, "little") == ANCHOR_INITIAL_TARGET_LE


def test_canonical_output_bytes_and_digest_match_reference_vector() -> None:
    output = torch.tensor([[1, -2], [300, -400]], dtype=torch.int32)

    assert canonical_output_bytes(output, expected_dim=2).hex() == (
        "01000000feffffff2c01000070feffff"
    )
    assert pow_digest(output, expected_dim=2).hex() == (
        "59ac76c5890db5a919f5802afaaa67d51a2fd5cecf2d82a8f5875c64d9a90222"
    )


def test_canonical_output_rejects_malformed_inputs() -> None:
    with pytest.raises(TypeError):
        canonical_output_bytes("not-a-tensor")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        canonical_output_bytes(torch.zeros((1, 1), dtype=torch.int32), expected_dim=0)
    with pytest.raises(PowInputError):
        canonical_output_bytes(torch.zeros((2, 2), dtype=torch.float32), expected_dim=2)
    with pytest.raises(PowInputError):
        canonical_output_bytes(torch.zeros((2, 3), dtype=torch.int32), expected_dim=2)


def test_target_comparison_is_unsigned_little_endian_and_inclusive() -> None:
    zero = bytes(HASH_LEN_BYTES)
    one = b"\x01" + bytes(HASH_LEN_BYTES - 1)
    two = b"\x02" + bytes(HASH_LEN_BYTES - 1)
    big_endian_one = bytes(HASH_LEN_BYTES - 1) + b"\x01"

    assert target_allows_digest(zero, zero)
    assert target_allows_digest(one, one)
    assert target_allows_digest(one, two)
    assert not target_allows_digest(two, one)
    assert not target_allows_digest(big_endian_one, one)


def test_target_comparison_rejects_malformed_bytes() -> None:
    with pytest.raises(TypeError):
        target_allows_digest("digest", bytes(HASH_LEN_BYTES))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        target_allows_digest(bytes(HASH_LEN_BYTES - 1), bytes(HASH_LEN_BYTES))
    with pytest.raises(ValueError):
        target_allows_digest(bytes(HASH_LEN_BYTES), bytes(HASH_LEN_BYTES + 1))


def test_matmul_int8_rejects_wrong_dtype_shape_and_layout() -> None:
    good = torch.zeros((POW_MATRIX_DIM, POW_MATRIX_DIM), dtype=torch.int8)

    with pytest.raises(PowInputError):
        matmul_int8(good.to(torch.float32), good, backend="cpu")
    with pytest.raises(PowInputError):
        matmul_int8(torch.zeros((POW_MATRIX_DIM - 1, POW_MATRIX_DIM), dtype=torch.int8), good)
    with pytest.raises(PowInputError):
        matmul_int8(good.t(), good)
    with pytest.raises(PowInputError):
        matmul_int8([[128] * POW_MATRIX_DIM for _ in range(POW_MATRIX_DIM)], good)
    with pytest.raises(ValueError):
        matmul_int8(good, good, backend="bogus")  # type: ignore[arg-type]


def test_matmul_int8_handles_signed_extreme_values() -> None:
    left = torch.full((POW_MATRIX_DIM, POW_MATRIX_DIM), 127, dtype=torch.int8)
    right = torch.full((POW_MATRIX_DIM, POW_MATRIX_DIM), -128, dtype=torch.int8)

    result = matmul_int8(left, right, backend="cpu")

    assert result.dtype == torch.int32
    assert result.shape == (POW_MATRIX_DIM, POW_MATRIX_DIM)
    assert int(result[0, 0]) == 127 * -128 * POW_MATRIX_DIM
    assert torch.equal(result[0, :16], torch.full((16,), 127 * -128 * POW_MATRIX_DIM))
