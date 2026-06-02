"""Tests for canonical fruit and anchor body structures."""

from __future__ import annotations

import pytest

from tensorpow.chain.blocks import (
    MAX_FRUIT_PAYLOAD_BYTES,
    Anchor,
    BlockDecodeError,
    FeeFloorEntry,
    Fruit,
    anchor_reward_root,
    fee_floor_set_root,
    fruit_set_root,
    parent_candidate_root,
    tx_merkle_root,
)
from tensorpow.chain.headers import AnchorHeader, FruitHeader
from tensorpow.crypto.hash import DOMAIN_SHARD_TREE, HASH_LEN_BYTES, domain_hash
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.mempool import ROOT_SHARD_ID, ShardTree
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH
from tensorpow.state.utxo import TEMPLATE_PKH
from tensorpow.tx.transaction import Output


def _fruit(transactions: tuple[bytes, ...] = (b"coinbase", b"tx-1")) -> Fruit:
    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=bytes.fromhex("11" * 32),
        parent_bitmap=b"",
        latest_anchor=bytes.fromhex("22" * 32),
        tx_merkle_root=tx_merkle_root(transactions),
        timestamp_ms=1,
        shard_id=0,
        nonce=2,
    )
    return Fruit(header=header, transactions=transactions)


def _anchor() -> Anchor:
    covered = (bytes.fromhex("01" * 32), bytes.fromhex("02" * 32))
    candidates = (bytes.fromhex("03" * 32), bytes.fromhex("04" * 32))
    shard_tree = b"shard-tree"
    fees = (FeeFloorEntry(0, 5), FeeFloorEntry(1 << 16, 9))
    header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=GENESIS_PARENT_HASH,
        fruit_set_root=fruit_set_root(covered),
        parent_candidate_root=parent_candidate_root(candidates),
        shard_tree_state_root=domain_hash(DOMAIN_SHARD_TREE, shard_tree),
        fee_floor_set_root=fee_floor_set_root(fees),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=2,
        nonce=3,
    )
    return Anchor(
        header=header,
        covered_fruit_hashes=covered,
        parent_candidate_hashes=candidates,
        shard_tree_bytes=shard_tree,
        fee_floor_entries=fees,
    )


def test_fruit_body_serializes_round_trips_and_rejects_bad_roots() -> None:
    fruit = _fruit()

    assert Fruit.deserialize(fruit.serialize()) == fruit
    assert fruit.tx_merkle_root() == fruit.header.tx_merkle_root
    assert fruit.block_hash() == fruit.header.header_hash()

    bad_header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=bytes(32),
        parent_bitmap=b"",
        latest_anchor=bytes(32),
        tx_merkle_root=bytes([9]) * 32,
        timestamp_ms=0,
        shard_id=0,
        nonce=0,
    )
    with pytest.raises(ValueError, match="tx_merkle_root"):
        Fruit(header=bad_header, transactions=(b"coinbase",))


def test_fruit_body_rejects_empty_oversized_trailing_and_truncated_payloads() -> None:
    with pytest.raises(ValueError, match="coinbase"):
        _fruit(())
    with pytest.raises(ValueError, match="payload"):
        _fruit((b"x" * (MAX_FRUIT_PAYLOAD_BYTES + 1),))

    serialized = _fruit().serialize()
    with pytest.raises(BlockDecodeError, match="trailing"):
        Fruit.deserialize(serialized + b"\x00")
    with pytest.raises(BlockDecodeError, match="truncated"):
        Fruit.deserialize(serialized[:-1])


def test_anchor_body_serializes_round_trips_and_checks_roots() -> None:
    anchor = _anchor()

    assert Anchor.deserialize(anchor.serialize()) == anchor
    assert anchor.fruit_set_root() == anchor.header.fruit_set_root
    assert anchor.parent_candidate_root() == anchor.header.parent_candidate_root
    assert anchor.shard_tree_state_root() == anchor.header.shard_tree_state_root
    assert anchor.fee_floor_set_root() == anchor.header.fee_floor_set_root
    assert anchor.anchor_reward_root() == anchor.header.anchor_reward_root
    assert anchor.block_hash() == anchor.header.header_hash()


def test_anchor_reward_outputs_are_header_committed() -> None:
    anchor = _anchor()
    reward_output = Output(1, TEMPLATE_PKH, payload=bytes.fromhex("05" * 32))

    with pytest.raises(ValueError, match="anchor_reward_root"):
        Anchor(
            header=anchor.header,
            covered_fruit_hashes=anchor.covered_fruit_hashes,
            parent_candidate_hashes=anchor.parent_candidate_hashes,
            shard_tree_bytes=anchor.shard_tree_bytes,
            fee_floor_entries=anchor.fee_floor_entries,
            anchor_reward_outputs=(reward_output,),
        )


def test_anchor_body_rejects_unsorted_duplicates_bad_roots_and_empty_nongenesis() -> None:
    anchor = _anchor()
    with pytest.raises(ValueError, match="ascending"):
        Anchor(
            header=anchor.header,
            covered_fruit_hashes=tuple(reversed(anchor.covered_fruit_hashes)),
            parent_candidate_hashes=anchor.parent_candidate_hashes,
            shard_tree_bytes=anchor.shard_tree_bytes,
            fee_floor_entries=anchor.fee_floor_entries,
        )
    with pytest.raises(ValueError, match="duplicates"):
        Anchor(
            header=anchor.header,
            covered_fruit_hashes=(anchor.covered_fruit_hashes[0], anchor.covered_fruit_hashes[0]),
            parent_candidate_hashes=anchor.parent_candidate_hashes,
            shard_tree_bytes=anchor.shard_tree_bytes,
            fee_floor_entries=anchor.fee_floor_entries,
        )
    bad_header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=GENESIS_PARENT_HASH,
        fruit_set_root=bytes([9]) * 32,
        parent_candidate_root=anchor.header.parent_candidate_root,
        shard_tree_state_root=anchor.header.shard_tree_state_root,
        fee_floor_set_root=anchor.header.fee_floor_set_root,
        anchor_reward_root=anchor.header.anchor_reward_root,
        timestamp_ms=0,
        nonce=0,
    )
    with pytest.raises(ValueError, match="fruit_set_root"):
        Anchor(
            header=bad_header,
            covered_fruit_hashes=anchor.covered_fruit_hashes,
            parent_candidate_hashes=anchor.parent_candidate_hashes,
            shard_tree_bytes=anchor.shard_tree_bytes,
            fee_floor_entries=anchor.fee_floor_entries,
        )

    empty_header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=GENESIS_PARENT_HASH,
        fruit_set_root=fruit_set_root(()),
        parent_candidate_root=parent_candidate_root(()),
        shard_tree_state_root=domain_hash(DOMAIN_SHARD_TREE, b""),
        fee_floor_set_root=fee_floor_set_root(()),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=0,
        nonce=0,
    )
    with pytest.raises(ValueError, match="non-genesis"):
        Anchor(
            header=empty_header,
            covered_fruit_hashes=(),
            parent_candidate_hashes=(),
            shard_tree_bytes=b"",
            fee_floor_entries=(),
        )

    with pytest.raises(ValueError, match="canonical root shard tree"):
        Anchor(
            header=empty_header,
            covered_fruit_hashes=(),
            parent_candidate_hashes=(),
            shard_tree_bytes=b"",
            fee_floor_entries=(),
            genesis_commitment=bytes([1]) * HASH_LEN_BYTES,
        )

    canonical_tree = ShardTree()
    canonical_fees = (FeeFloorEntry(ROOT_SHARD_ID, 0),)
    genesis_header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=GENESIS_PARENT_HASH,
        fruit_set_root=fruit_set_root(()),
        parent_candidate_root=parent_candidate_root(()),
        shard_tree_state_root=canonical_tree.state_root(),
        fee_floor_set_root=fee_floor_set_root(canonical_fees),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=0,
        nonce=0,
    )
    genesis = Anchor(
        header=genesis_header,
        covered_fruit_hashes=(),
        parent_candidate_hashes=(),
        shard_tree_bytes=canonical_tree.serialize(),
        fee_floor_entries=canonical_fees,
        genesis_commitment=bytes([1]) * HASH_LEN_BYTES,
    )
    assert Anchor.deserialize(genesis.serialize()) == genesis
    assert genesis.block_hash() != genesis.header.header_hash()

    bad_fee_header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=GENESIS_PARENT_HASH,
        fruit_set_root=fruit_set_root(()),
        parent_candidate_root=parent_candidate_root(()),
        shard_tree_state_root=canonical_tree.state_root(),
        fee_floor_set_root=fee_floor_set_root((FeeFloorEntry(ROOT_SHARD_ID, 1),)),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=0,
        nonce=0,
    )
    with pytest.raises(ValueError, match="zero root-shard fee floor"):
        Anchor(
            header=bad_fee_header,
            covered_fruit_hashes=(),
            parent_candidate_hashes=(),
            shard_tree_bytes=canonical_tree.serialize(),
            fee_floor_entries=(FeeFloorEntry(ROOT_SHARD_ID, 1),),
            genesis_commitment=bytes([1]) * HASH_LEN_BYTES,
        )

    with pytest.raises(ValueError, match="must not cover fruits"):
        Anchor(
            header=anchor.header,
            covered_fruit_hashes=anchor.covered_fruit_hashes,
            parent_candidate_hashes=anchor.parent_candidate_hashes,
            shard_tree_bytes=anchor.shard_tree_bytes,
            fee_floor_entries=anchor.fee_floor_entries,
            genesis_commitment=bytes([1]) * HASH_LEN_BYTES,
        )
    bad_parent_header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=bytes([9]) * HASH_LEN_BYTES,
        fruit_set_root=fruit_set_root(()),
        parent_candidate_root=parent_candidate_root(()),
        shard_tree_state_root=genesis_header.shard_tree_state_root,
        fee_floor_set_root=genesis_header.fee_floor_set_root,
        anchor_reward_root=genesis_header.anchor_reward_root,
        timestamp_ms=0,
        nonce=0,
    )
    with pytest.raises(ValueError, match="GENESIS_PARENT_HASH"):
        Anchor(
            header=bad_parent_header,
            covered_fruit_hashes=(),
            parent_candidate_hashes=(),
            shard_tree_bytes=canonical_tree.serialize(),
            fee_floor_entries=canonical_fees,
            genesis_commitment=bytes([1]) * HASH_LEN_BYTES,
        )


def test_fee_floor_entries_validate_sorting_and_shard_ids() -> None:
    anchor = _anchor()
    with pytest.raises(ValueError, match="sorted"):
        Anchor(
            header=anchor.header,
            covered_fruit_hashes=anchor.covered_fruit_hashes,
            parent_candidate_hashes=anchor.parent_candidate_hashes,
            shard_tree_bytes=anchor.shard_tree_bytes,
            fee_floor_entries=tuple(reversed(anchor.fee_floor_entries)),
        )
    with pytest.raises(ValueError, match="path"):
        FeeFloorEntry(1, 1)
