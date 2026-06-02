"""Deterministic INT8 proof-of-work matrix kernel."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from functools import cache
from typing import Final, Literal, cast

import torch

from tensorpow.crypto.hash import DOMAIN_POW_OUTPUT, HASH_LEN_BYTES, domain_hash

type Backend = Literal["auto", "cpu", "cuda", "mps"]

POW_MATRIX_DIM: Final[int] = 1024
POW_ACCUM_BYTES: Final[int] = 4
POW_ACCUM_BITS: Final[int] = 32
POW_MATRIX_INPUTS: Final[int] = 2
POW_MATRIX_BYTES: Final[int] = POW_MATRIX_DIM * POW_MATRIX_DIM
POW_OPS_PER_MAC: Final[int] = 2
POW_OPS_PER_MATMUL: Final[int] = POW_OPS_PER_MAC * POW_MATRIX_DIM**3
FRUIT_WORK_OPS: Final[int] = 6_000_000_000
ANCHOR_WORK_MULTIPLIER: Final[int] = 1000
TARGET_BYTES: Final[int] = HASH_LEN_BYTES
INT8_MIN_VALUE: Final[int] = -128
INT8_MAX_VALUE: Final[int] = 127

FRUIT_TARGET_LE: Final[bytes] = bytes.fromhex(
    "195719183ec6044d312767ab3e988ffe757563505b5c45760f1923cf803fa05b"
)
ANCHOR_INITIAL_TARGET_LE: Final[bytes] = bytes.fromhex(
    "c4e77a534b84e06c453fe6097f9953a30b5d8b9e968aa9cc16383daccc741700"
)
ANCHOR_MIN_TARGET_LE: Final[bytes] = bytes.fromhex(
    "0100000000000000000000000000000000000000000000000000000000000000"
)
ANCHOR_MAX_TARGET_LE: Final[bytes] = FRUIT_TARGET_LE

_DETERMINISM_CONFIGURED = False
_VALID_BACKENDS: Final[frozenset[str]] = frozenset({"auto", "cpu", "cuda", "mps"})


class PowInputError(ValueError):
    """Raised when a PoW input violates consensus shape or type rules."""


class PowBackendUnavailableError(RuntimeError):
    """Raised when a requested PoW backend cannot execute deterministically."""


def configure_torch_determinism() -> None:
    """Enable deterministic torch settings used by consensus-touching kernels."""

    global _DETERMINISM_CONFIGURED
    if _DETERMINISM_CONFIGURED:
        return

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _DETERMINISM_CONFIGURED = True


def matmul_int8(
    left: torch.Tensor | Sequence[Sequence[int]],
    right: torch.Tensor | Sequence[Sequence[int]],
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Return the canonical INT8 x INT8 -> INT32 PoW matrix product on CPU."""

    configure_torch_determinism()
    left_tensor = _require_int8_matrix("left", left, POW_MATRIX_DIM)
    right_tensor = _require_int8_matrix("right", right, POW_MATRIX_DIM)
    resolved_backend = resolve_backend(backend)

    if resolved_backend == "cuda":
        result = _int_mm(left_tensor.to("cuda"), right_tensor.to("cuda"))
        torch.cuda.synchronize()
        result = result.cpu()
    elif resolved_backend == "mps":
        result = _mps_int32_mm(left_tensor, right_tensor)
    else:
        result = _int_mm(left_tensor, right_tensor)

    if result.dtype != torch.int32:
        raise PowBackendUnavailableError("INT8 matmul backend did not produce int32 output")
    if tuple(result.shape) != (POW_MATRIX_DIM, POW_MATRIX_DIM):
        raise PowBackendUnavailableError("INT8 matmul backend returned an unexpected shape")
    return result.contiguous()


def resolve_backend(backend: Backend) -> Literal["cpu", "cuda", "mps"]:
    """Resolve a user backend name to the deterministic execution backend."""

    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"unknown PoW backend {backend!r}")

    if backend == "auto":
        if torch.cuda.is_available() and callable(getattr(torch, "_int_mm", None)):
            return "cuda"
        if _mps_int32_mm_available():
            return "mps"
        return "cpu"
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise PowBackendUnavailableError("CUDA backend requested but CUDA is unavailable")
        if not callable(getattr(torch, "_int_mm", None)):
            raise PowBackendUnavailableError("CUDA backend requires torch._int_mm")
        return "cuda"
    if backend == "mps":
        if not _mps_int32_mm_available():
            raise PowBackendUnavailableError(
                "MPS backend requested but deterministic MPS int32 matmul is unavailable"
            )
        return "mps"
    return "cpu"


def canonical_output_bytes(output: torch.Tensor, *, expected_dim: int = POW_MATRIX_DIM) -> bytes:
    """Serialize an INT32 output matrix as row-major little-endian bytes."""

    if not isinstance(expected_dim, int):
        raise TypeError("expected_dim must be int")
    if expected_dim <= 0:
        raise ValueError("expected_dim must be positive")
    if not isinstance(output, torch.Tensor):
        raise TypeError("output must be a torch.Tensor")
    if output.dtype != torch.int32:
        raise PowInputError("output must have dtype torch.int32")
    if output.device.type != "cpu":
        raise PowInputError("output must be on CPU before canonical serialization")
    if tuple(output.shape) != (expected_dim, expected_dim):
        raise PowInputError(f"output must have shape ({expected_dim}, {expected_dim})")

    contiguous = output.contiguous()
    expected_len = expected_dim * expected_dim * POW_ACCUM_BYTES
    if sys.byteorder == "little":
        raw = contiguous.numpy().tobytes(order="C")
        if len(raw) != expected_len:
            raise PowInputError("output storage length does not match matrix shape")
        return raw

    return b"".join(
        int(value).to_bytes(POW_ACCUM_BYTES, "little", signed=True)
        for value in contiguous.reshape(-1).tolist()
    )


def pow_digest(output: torch.Tensor, *, expected_dim: int = POW_MATRIX_DIM) -> bytes:
    """Hash canonical PoW output bytes with the PoW output domain."""

    return domain_hash(DOMAIN_POW_OUTPUT, canonical_output_bytes(output, expected_dim=expected_dim))


def target_allows_digest(digest: bytes, target: bytes) -> bool:
    """Return whether digest is less than or equal to target as little-endian uint256."""

    _require_bytes_len("digest", digest, TARGET_BYTES)
    _require_bytes_len("target", target, TARGET_BYTES)
    return int.from_bytes(digest, "little") <= int.from_bytes(target, "little")


def _require_int8_matrix(
    name: str,
    value: torch.Tensor | Sequence[Sequence[int]],
    expected_dim: int,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
        if not tensor.dtype.is_floating_point and tensor.numel() > 0:
            min_value = int(tensor.min().item())
            max_value = int(tensor.max().item())
            if min_value < INT8_MIN_VALUE or max_value > INT8_MAX_VALUE:
                raise PowInputError(f"{name} values must fit signed int8")
            tensor = tensor.to(torch.int8)

    if tensor.dtype != torch.int8:
        raise PowInputError(f"{name} must have dtype torch.int8")
    if tuple(tensor.shape) != (expected_dim, expected_dim):
        raise PowInputError(f"{name} must have shape ({expected_dim}, {expected_dim})")
    if tensor.device.type != "cpu":
        raise PowInputError(f"{name} must be a CPU tensor before backend transfer")
    if not tensor.is_contiguous():
        raise PowInputError(f"{name} must be contiguous")
    return tensor


def _int_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    int_mm = getattr(torch, "_int_mm", None)
    if callable(int_mm):
        return cast(torch.Tensor, int_mm(left, right))
    return left.to(torch.int32) @ right.to(torch.int32)


def _mps_int32_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    result = left.to("mps").to(torch.int32) @ right.to("mps").to(torch.int32)
    torch.mps.synchronize()
    return result.cpu()


def _mps_probe_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(POW_MATRIX_BYTES, dtype=torch.int32).reshape(
        POW_MATRIX_DIM,
        POW_MATRIX_DIM,
    )
    left = ((values * 73 + 19) % 256 - 128).to(torch.int8).contiguous()
    right = ((values * 37 + 91) % 256 - 128).to(torch.int8).contiguous()
    left[:16, :] = INT8_MIN_VALUE
    right[:, :16] = INT8_MIN_VALUE
    return left, right


@cache
def _mps_int32_mm_available() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        left, right = _mps_probe_matrices()
        reference = _int_mm(left, right)
        result = _mps_int32_mm(left, right)
    except RuntimeError:
        return False
    return (
        result.dtype == torch.int32
        and tuple(result.shape) == tuple(reference.shape)
        and torch.equal(result, reference)
    )


def _require_bytes_len(name: str, value: bytes, expected_len: int) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != expected_len:
        raise ValueError(f"{name} must be {expected_len} bytes")
