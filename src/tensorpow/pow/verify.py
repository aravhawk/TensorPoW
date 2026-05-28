"""Single-proof TensorPoW verification."""

from __future__ import annotations

from tensorpow.pow.challenge import PowHeader, build_challenge_matrices
from tensorpow.pow.kernel import Backend, matmul_int8, pow_digest, target_allows_digest


def pow_digest_for_header(header: PowHeader, *, backend: Backend = "auto") -> bytes:
    """Compute the canonical PoW digest for a header."""

    left, right = build_challenge_matrices(header)
    return pow_digest(matmul_int8(left, right, backend=backend))


def verify_pow(header: PowHeader, target: bytes, *, backend: Backend = "auto") -> bool:
    """Return whether the header's nonce satisfies target."""

    return target_allows_digest(pow_digest_for_header(header, backend=backend), target)
