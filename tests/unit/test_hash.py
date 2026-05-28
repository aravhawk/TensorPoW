"""Tests for TensorPoW BLAKE3 helpers."""

from __future__ import annotations

import pytest

from tensorpow.crypto.hash import (
    HASH_LEN_BYTES,
    MerkleProof,
    MerkleTree,
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


def test_merkle_tree_empty_single_and_mutation_roots() -> None:
    tree = MerkleTree()
    empty_root = tree.root()
    assert len(empty_root) == HASH_LEN_BYTES

    key = hash_bytes(b"outpoint-0")
    value = hash_bytes(b"utxo-0")
    tree.set(key, value)
    single_root = tree.root()
    assert single_root != empty_root

    tree.delete(key)
    assert tree.root() == empty_root


def test_merkle_tree_power_of_two_and_non_power_of_two_roots() -> None:
    power_items = {
        hash_bytes(f"key-{i}".encode()): hash_bytes(f"value-{i}".encode()) for i in range(4)
    }
    non_power_items = {
        hash_bytes(f"key-{i}".encode()): hash_bytes(f"value-{i}".encode()) for i in range(5)
    }
    assert MerkleTree(power_items).root() == MerkleTree(dict(reversed(power_items.items()))).root()
    assert MerkleTree(power_items).root() != MerkleTree(non_power_items).root()


def test_merkle_tree_inclusion_proof_verifies_and_detects_tampering() -> None:
    items = {hash_bytes(f"key-{i}".encode()): hash_bytes(f"value-{i}".encode()) for i in range(5)}
    tree = MerkleTree(items)
    key, value = next(iter(items.items()))
    proof = tree.inclusion_proof(key)

    assert proof.value == value
    assert MerkleTree.verify_proof(proof, tree.root())

    bad_proof = MerkleProof(key=proof.key, value=hash_bytes(b"wrong"), siblings=proof.siblings)
    assert not MerkleTree.verify_proof(bad_proof, tree.root())


def test_merkle_tree_rejects_malformed_keys_and_missing_proofs() -> None:
    tree = MerkleTree()
    with pytest.raises(ValueError):
        tree.set(b"short", hash_bytes(b"value"))
    with pytest.raises(KeyError):
        tree.inclusion_proof(hash_bytes(b"missing"))
