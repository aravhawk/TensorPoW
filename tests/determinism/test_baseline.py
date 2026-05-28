"""Baseline determinism checks for consensus-critical integer matmul."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest
from blake3 import blake3

MATRIX_SIZE = 32
REFERENCE_DIGEST = "66f13a2f3dd355caf9a44873231ff78e0148133921d1303ef883d7486890f660"


def _matrix_a() -> list[list[int]]:
    return [
        [((row * 17 + col * 31 + 13) % 256) - 128 for col in range(MATRIX_SIZE)]
        for row in range(MATRIX_SIZE)
    ]


def _matrix_b() -> list[list[int]]:
    return [
        [((row * 47 - col * 19 + 7) % 256) - 128 for col in range(MATRIX_SIZE)]
        for row in range(MATRIX_SIZE)
    ]


def _matmul_int8_to_int32(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> list[list[int]]:
    return [
        [
            sum(left[row][inner] * right[inner][col] for inner in range(MATRIX_SIZE))
            for col in range(MATRIX_SIZE)
        ]
        for row in range(MATRIX_SIZE)
    ]


def _canonical_int32_bytes(matrix: Sequence[Sequence[int]]) -> bytes:
    output = bytearray()
    for row in matrix:
        for value in row:
            output.extend(int(value).to_bytes(4, "little", signed=True))
    return bytes(output)


def _digest(matrix: Sequence[Sequence[int]]) -> str:
    return blake3(_canonical_int32_bytes(matrix)).hexdigest()


def _torch_backend_matrix(backend: str) -> list[list[int]]:
    torch = pytest.importorskip("torch")

    left = torch.tensor(_matrix_a(), dtype=torch.int8)
    right = torch.tensor(_matrix_b(), dtype=torch.int8)

    if backend == "cpu":
        result = left.to(torch.int32) @ right.to(torch.int32)
    elif backend == "cuda":
        if not torch.cuda.is_available():
            pytest.fail("TENSORPOW_DETERMINISM_BACKEND=cuda requested but CUDA is unavailable")
        if not hasattr(torch, "_int_mm"):
            pytest.fail("TENSORPOW_DETERMINISM_BACKEND=cuda requires torch._int_mm")
        result = torch._int_mm(left.cuda(), right.cuda()).cpu()
    elif backend == "mps":
        if not torch.backends.mps.is_available():
            pytest.fail("TENSORPOW_DETERMINISM_BACKEND=mps requested but MPS is unavailable")
        result = left.to(torch.int32) @ right.to(torch.int32)
    else:
        pytest.fail(f"unknown TENSORPOW_DETERMINISM_BACKEND={backend!r}")

    assert result.dtype == torch.int32
    return [[int(value) for value in row] for row in result.tolist()]


@pytest.mark.determinism
def test_pure_python_int8_matmul_reference_digest() -> None:
    result = _matmul_int8_to_int32(_matrix_a(), _matrix_b())
    assert _digest(result) == REFERENCE_DIGEST


@pytest.mark.determinism
def test_requested_torch_backend_matches_reference_digest() -> None:
    backend = os.environ.get("TENSORPOW_DETERMINISM_BACKEND", "cpu")
    result = _torch_backend_matrix(backend)
    assert _digest(result) == REFERENCE_DIGEST
