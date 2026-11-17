"""Tests for PoW challenge preimages and matrix generation."""

from __future__ import annotations

import pytest
import torch

from tensorpow.crypto.hash import DOMAIN_POW_CHALLENGE_ANCHOR, DOMAIN_POW_CHALLENGE_FRUIT
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.challenge import (
    FORMAT_EPOCH,
    GENESIS_PARENT_HASH,
    U64_MAX,
    AnchorPowHeader,
    FruitPowHeader,
    anchor_pow_preimage,
    build_challenge,
    build_challenge_matrices,
    fruit_pow_preimage,
    with_nonce,
)


def test_fruit_preimage_and_small_challenge_match_reference_vector() -> None:
    header = FruitPowHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        effective_parent_hashes=(bytes(range(32)),),
        latest_anchor=GENESIS_PARENT_HASH,
        tx_merkle_root=bytes(reversed(range(32))),
        timestamp_ms=123456789,
        shard_id=7,
        nonce=9,
    )

    assert fruit_pow_preimage(header).hex() == (
        "00000001000100000102030405060708090a0b0c0d0e0f101112131415161718191a"
        "1b1c1d1e1f0000000000000000000000000000000000000000000000000000000000"
        "00000000001f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403"
        "02010015cd5b0700000000070000000900000000000000"
    )
    left, right = build_challenge_matrices(header, matrix_dim=4)

    assert left.dtype == torch.int8
    assert right.dtype == torch.int8
    assert left.tolist() == [
        [112, 97, -38, 80],
        [-123, -33, -94, 5],
        [-47, 56, 11, 66],
        [67, 36, 92, 81],
    ]
    assert right.tolist() == [
        [114, -75, -66, 54],
        [76, -104, 37, -85],
        [-105, -94, -46, 73],
        [-126, -9, -103, -104],
    ]


def test_anchor_preimage_and_domain_separation_match_reference_vector() -> None:
    header = AnchorPowHeader(
        version=FORMAT_EPOCH,
        parent_anchor=bytes(range(32)),
        fruit_set_root=bytes([1]) * 32,
        parent_candidate_root=bytes([2]) * 32,
        shard_tree_state_root=bytes([3]) * 32,
        fee_floor_set_root=bytes([4]) * 32,
        anchor_reward_root=bytes([5]) * 32,
        timestamp_ms=987654321,
        nonce=11,
        utxo_state_root=bytes([6]) * 32,
    )

    assert anchor_pow_preimage(header).hex() == (
        "010000000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d"
        "1e1f01010101010101010101010101010101010101010101010101010101010101"
        "010202020202020202020202020202020202020202020202020202020202020202"
        "030303030303030303030303030303030303030303030303030303030303030304"
        "040404040404040404040404040404040404040404040404040404040404040505"
        "050505050505050505050505050505050505050505050505050505050505060606"
        "0606060606060606060606060606060606060606060606060606060606b168de3a"
        "000000000b00000000000000"
    )
    left, _right = build_challenge_matrices(header, matrix_dim=4)

    assert left.tolist() == [
        [10, 112, 112, -84],
        [15, -111, -54, 4],
        [-75, 25, -109, -125],
        [-64, 115, 18, -102],
    ]


def test_build_challenge_compatibility_helper_preserves_domain_separation() -> None:
    parent = bytes([7]) * 32
    root = bytes([8]) * 32

    fruit_matrix = build_challenge([parent], root, 1, 2, DOMAIN_POW_CHALLENGE_FRUIT, matrix_dim=8)
    anchor_matrix = build_challenge([parent], root, 1, 2, DOMAIN_POW_CHALLENGE_ANCHOR, matrix_dim=8)

    assert fruit_matrix.shape == (8, 8)
    assert anchor_matrix.shape == (8, 8)
    assert not torch.equal(fruit_matrix, anchor_matrix)


def test_challenge_inputs_reject_malformed_headers() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        FruitPowHeader(
            version=FORMAT_EPOCH,
            sig_type_supported=SIG_TYPE_ED25519_BIT,
            effective_parent_hashes=(bytes(32), bytes(32)),
            latest_anchor=GENESIS_PARENT_HASH,
            tx_merkle_root=GENESIS_PARENT_HASH,
            timestamp_ms=0,
            shard_id=0,
            nonce=0,
        )
    with pytest.raises(ValueError, match="ED25519"):
        FruitPowHeader(
            version=FORMAT_EPOCH,
            sig_type_supported=0,
            effective_parent_hashes=(bytes([1]) * 32,),
            latest_anchor=GENESIS_PARENT_HASH,
            tx_merkle_root=GENESIS_PARENT_HASH,
            timestamp_ms=0,
            shard_id=0,
            nonce=0,
        )
    with pytest.raises(ValueError):
        AnchorPowHeader(
            version=FORMAT_EPOCH,
            parent_anchor=b"short",
            fruit_set_root=GENESIS_PARENT_HASH,
            parent_candidate_root=GENESIS_PARENT_HASH,
            shard_tree_state_root=GENESIS_PARENT_HASH,
            fee_floor_set_root=GENESIS_PARENT_HASH,
            anchor_reward_root=GENESIS_PARENT_HASH,
            timestamp_ms=0,
            nonce=0,
        )
    with pytest.raises(ValueError):
        build_challenge([bytes(32), bytes([1]) * 32], bytes(32), 0, 0, DOMAIN_POW_CHALLENGE_ANCHOR)
    with pytest.raises(ValueError):
        build_challenge([bytes(32)], bytes(32), 0, 0, 255)
    with pytest.raises(ValueError):
        build_challenge([bytes(32)], bytes(32), 0, 0, DOMAIN_POW_CHALLENGE_FRUIT, matrix_dim=0)


def test_with_nonce_validates_uint64_bounds() -> None:
    header = FruitPowHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        effective_parent_hashes=(bytes([1]) * 32,),
        latest_anchor=GENESIS_PARENT_HASH,
        tx_merkle_root=GENESIS_PARENT_HASH,
        timestamp_ms=0,
        shard_id=0,
        nonce=0,
    )

    assert with_nonce(header, U64_MAX).nonce == U64_MAX
    with pytest.raises(ValueError):
        with_nonce(header, U64_MAX + 1)
