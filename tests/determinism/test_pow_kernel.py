"""Consensus PoW kernel determinism vectors."""

from __future__ import annotations

import os
from typing import cast

import pytest

from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.challenge import FORMAT_EPOCH, FruitPowHeader, build_challenge_matrices
from tensorpow.pow.kernel import (
    Backend,
    PowBackendUnavailableError,
    matmul_int8,
    pow_digest,
    resolve_backend,
)

REFERENCE_POW_DIGEST = "f1aa4b92aceca26dbd45402dd4b996f9ff76d6c31db66df675bba44649bbbed3"


def _reference_header() -> FruitPowHeader:
    return FruitPowHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        effective_parent_hashes=(bytes.fromhex("11" * 32), bytes.fromhex("22" * 32)),
        latest_anchor=bytes.fromhex("33" * 32),
        tx_merkle_root=bytes.fromhex("44" * 32),
        timestamp_ms=1_779_841_408_123,
        shard_id=(3 << 16) | 5,
        nonce=42,
    )


@pytest.mark.determinism
def test_pow_kernel_reference_digest_matches_requested_backend() -> None:
    backend = cast(Backend, os.environ.get("TENSORPOW_DETERMINISM_BACKEND", "cpu"))
    left, right = build_challenge_matrices(_reference_header())
    resolved_backend = resolve_backend(backend)

    if backend != "auto":
        assert resolved_backend == backend

    assert pow_digest(matmul_int8(left, right, backend=backend)).hex() == REFERENCE_POW_DIGEST


@pytest.mark.determinism
@pytest.mark.parametrize("backend", ("cuda", "mps"))
def test_available_accelerator_backend_matches_cpu_digest(backend: Backend) -> None:
    left, right = build_challenge_matrices(_reference_header())
    try:
        resolved_backend = resolve_backend(backend)
    except PowBackendUnavailableError:
        pytest.skip(f"{backend} backend is unavailable on this host")

    assert resolved_backend == backend
    cpu_digest = pow_digest(matmul_int8(left, right, backend="cpu"))
    accelerator_digest = pow_digest(matmul_int8(left, right, backend=backend))
    assert accelerator_digest == cpu_digest
