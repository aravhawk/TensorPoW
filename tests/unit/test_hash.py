"""Tests for TensorPoW BLAKE3 helpers."""

from __future__ import annotations

import pytest

import tensorpow.crypto.hash as hash_module
from tensorpow.crypto.hash import (
    HASH_LEN_BYTES,
    derive_key,
    domain_hash,
    hash_bytes,
    keyed_hash,
)


def test_hash_bytes_matches_blake3_reference_vectors() -> None:
    assert (
        hash_bytes(b"").hex() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )
    assert (
        hash_bytes(b"abc").hex()
        == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
    )


def test_keyed_hash_and_derive_key_are_stable() -> None:
    zero_key = bytes(HASH_LEN_BYTES)
    assert (
        keyed_hash(zero_key, b"message").hex()
        == "4643f65493d8140c22900b2e0cfa25e65b536a7b73d534c4dd229375611c473e"
    )
    assert (
        derive_key("test", b"input").hex()
        == "4a73c372641e118a2cfa4c1ac6cb853c6b3cd842deb05081695019acf987721a"
    )


def test_hash_helpers_validate_inputs() -> None:
    with pytest.raises(TypeError):
        hash_bytes("abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        keyed_hash(b"short", b"data")
    with pytest.raises(ValueError):
        domain_hash(256, b"data")
    with pytest.raises(ValueError):
        derive_key("", b"material")


def test_crypto_hash_does_not_expose_sparse_merkle_tree_trap() -> None:
    assert not hasattr(hash_module, "MerkleTree")
    assert not hasattr(hash_module, "MerkleProof")
