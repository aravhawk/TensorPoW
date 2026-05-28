"""Consensus PoW kernel determinism vectors."""

from __future__ import annotations

import os

import pytest

from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.challenge import FORMAT_EPOCH, FruitPowHeader, build_challenge_matrices
from tensorpow.pow.kernel import matmul_int8, pow_digest

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
    backend = os.environ.get("TENSORPOW_DETERMINISM_BACKEND", "cpu")
    left, right = build_challenge_matrices(_reference_header())

    assert pow_digest(matmul_int8(left, right, backend=backend)).hex() == REFERENCE_POW_DIGEST
