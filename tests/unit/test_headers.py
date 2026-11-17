"""Tests for canonical fruit and anchor headers."""

from __future__ import annotations

import pytest

from tensorpow.chain.blocks import (
    FeeFloorEntry,
    anchor_reward_root,
    fee_floor_set_root,
    fruit_set_root,
    parent_candidate_root,
    tx_merkle_root,
)
from tensorpow.chain.headers import (
    PARENT_BITMAP_MAX_BYTES,
    AnchorHeader,
    FruitHeader,
    HeaderDecodeError,
)
from tensorpow.crypto.hash import DOMAIN_SHARD_TREE, domain_hash
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH

FRUIT_HEADER_BYTES = (
    "0000010011111111111111111111111111111111111111111111111111111111"
    "1111111101000322222222222222222222222222222222222222222222222222"
    "2222222222222228d84a0537516761310f7989f39006c38a307e45b9a9f8ae34"
    "fd7b5a724bf06215cd5b0700000000010002000700000000000000"
)
FRUIT_HEADER_HASH = "72db89383c3fa8d73d37ba472c9b4996548cf4c9703e52cef408a0ae4cbb343c"
ANCHOR_HEADER_BYTES = (
    "0000000000000000000000000000000000000000000000000000000000000000"
    "0000e12431afcbb5661559652f696524ef69e8229b8d9bd5353cd6f54f09503b"
    "401495ce9275d68aa0d587917a28afd6cfbfee66ac46216617758f535cc67caa"
    "3b03d42272bdeb68b253cda0020ca5bcfb073519dca0ee1bf27b0f9a6e269b51"
    "fd4a21a30fb886833fc05c8db364caa104a5751eb899498016058ad41ed7f01f"
    "8a6016f4ab7a3f6c9dccc72d514a15b68b37fd8fa1c80874424b7e4909c8ea5b"
    "cc1c050505050505050505050505050505050505050505050505050505050505"
    "05058ed73e0d000000000800000000000000"
)
ANCHOR_HEADER_HASH = "d0767e1393567082a2a6a44eda30bc933c332648a2370c4f98520ffb7895408c"


def _fruit_header() -> FruitHeader:
    return FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=bytes.fromhex("11" * 32),
        parent_bitmap=b"\x03",
        latest_anchor=bytes.fromhex("22" * 32),
        tx_merkle_root=tx_merkle_root((b"coinbase", b"tx-1")),
        timestamp_ms=123456789,
        shard_id=(2 << 16) | 1,
        nonce=7,
    )


def _anchor_header() -> AnchorHeader:
    covered = (bytes.fromhex("01" * 32), bytes.fromhex("02" * 32))
    candidates = (bytes.fromhex("03" * 32), bytes.fromhex("04" * 32))
    fees = (FeeFloorEntry(0, 5), FeeFloorEntry(1 << 16, 9))
    shard_tree = b"shard-tree"
    return AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=GENESIS_PARENT_HASH,
        fruit_set_root=fruit_set_root(covered),
        parent_candidate_root=parent_candidate_root(candidates),
        shard_tree_state_root=domain_hash(DOMAIN_SHARD_TREE, shard_tree),
        fee_floor_set_root=fee_floor_set_root(fees),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=222222222,
        nonce=8,
        utxo_state_root=bytes.fromhex("05" * 32),
    )


def test_fruit_header_serializes_hashes_and_round_trips_reference_vector() -> None:
    header = _fruit_header()

    assert header.serialize().hex() == FRUIT_HEADER_BYTES
    assert header.header_hash().hex() == FRUIT_HEADER_HASH
    assert FruitHeader.deserialize(header.serialize()) == header


def test_anchor_header_serializes_hashes_and_round_trips_reference_vector() -> None:
    header = _anchor_header()

    assert header.serialize().hex() == ANCHOR_HEADER_BYTES
    assert header.header_hash().hex() == ANCHOR_HEADER_HASH
    assert AnchorHeader.deserialize(header.serialize()) == header


def test_header_deserializers_reject_trailing_and_truncated_bytes() -> None:
    with pytest.raises(HeaderDecodeError, match="trailing"):
        FruitHeader.deserialize(bytes.fromhex(FRUIT_HEADER_BYTES) + b"\x00")
    with pytest.raises(HeaderDecodeError, match="truncated"):
        AnchorHeader.deserialize(bytes.fromhex(ANCHOR_HEADER_BYTES)[:-1])


def test_fruit_header_rejects_malformed_fields() -> None:
    with pytest.raises(ValueError, match="FORMAT_EPOCH"):
        _fruit_header().__class__(
            version=1,
            sig_type_supported=SIG_TYPE_ED25519_BIT,
            parent_selected=bytes(32),
            parent_bitmap=b"",
            latest_anchor=bytes(32),
            tx_merkle_root=bytes(32),
            timestamp_ms=0,
            shard_id=0,
            nonce=0,
        )
    with pytest.raises(ValueError, match="Ed25519"):
        FruitHeader(
            version=FORMAT_EPOCH,
            sig_type_supported=SIG_TYPE_ED25519_BIT | 0x02,
            parent_selected=bytes(32),
            parent_bitmap=b"",
            latest_anchor=bytes(32),
            tx_merkle_root=bytes(32),
            timestamp_ms=0,
            shard_id=0,
            nonce=0,
        )
    with pytest.raises(ValueError, match="max length"):
        FruitHeader(
            version=FORMAT_EPOCH,
            sig_type_supported=SIG_TYPE_ED25519_BIT,
            parent_selected=bytes(32),
            parent_bitmap=bytes(PARENT_BITMAP_MAX_BYTES + 1),
            latest_anchor=bytes(32),
            tx_merkle_root=bytes(32),
            timestamp_ms=0,
            shard_id=0,
            nonce=0,
        )
    with pytest.raises(ValueError, match="path"):
        FruitHeader(
            version=FORMAT_EPOCH,
            sig_type_supported=SIG_TYPE_ED25519_BIT,
            parent_selected=bytes(32),
            parent_bitmap=b"",
            latest_anchor=bytes(32),
            tx_merkle_root=bytes(32),
            timestamp_ms=0,
            shard_id=1,
            nonce=0,
        )


def test_effective_parent_hashes_validate_bitmap_bounds_and_duplicates() -> None:
    header = _fruit_header()
    candidates = (bytes.fromhex("33" * 32), bytes.fromhex("44" * 32))

    assert header.effective_parent_hashes(candidates) == (
        bytes.fromhex("11" * 32),
        bytes.fromhex("33" * 32),
        bytes.fromhex("44" * 32),
    )
    with pytest.raises(ValueError, match="beyond"):
        header.effective_parent_hashes((bytes.fromhex("33" * 32),))
    duplicate_header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=bytes.fromhex("33" * 32),
        parent_bitmap=b"\x01",
        latest_anchor=bytes(32),
        tx_merkle_root=bytes(32),
        timestamp_ms=0,
        shard_id=0,
        nonce=0,
    )
    with pytest.raises(ValueError, match="duplicates"):
        duplicate_header.effective_parent_hashes((bytes.fromhex("33" * 32),))


def test_headers_convert_to_pow_headers() -> None:
    fruit_pow = _fruit_header().to_pow_header((bytes.fromhex("33" * 32), bytes.fromhex("44" * 32)))
    anchor_pow = _anchor_header().to_pow_header()

    assert fruit_pow.nonce == 7
    assert len(fruit_pow.effective_parent_hashes) == 3
    assert anchor_pow.nonce == 8
