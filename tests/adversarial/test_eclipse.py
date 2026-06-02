"""Adversarial regression for eclipse-isolated node recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorpow.chain.blocks import Anchor
from tensorpow.net import MSG_TYPE_TX, WireDecodeError, decode_wire_message, encode_wire_message
from tensorpow.node import TensorPowConfig, TensorPowNode
from tests.adversarial._helpers import anchor, coinbase_tx, fruit, genesis_anchor


def test_eclipse_packets_are_rejected_and_honest_sync_restores_state(tmp_path: Path) -> None:
    genesis = genesis_anchor()
    honest = _node(tmp_path / "honest", genesis=genesis)
    victim = _node(tmp_path / "victim", genesis=genesis)
    try:
        honest_fruit = fruit(
            (coinbase_tx(21, amount=50).to_bytes(),),
            nonce=21,
            timestamp_ms=2,
            latest_anchor=genesis.block_hash(),
        )
        honest_anchor = anchor(
            covered_fruit_hashes=(honest_fruit.block_hash(),),
            parent_anchor=genesis.block_hash(),
            fee_floor_matoms_per_kb=7,
            timestamp_ms=3,
            nonce=22,
        )
        assert honest.process_anchor(genesis)
        assert honest.process_fruit(honest_fruit)
        assert honest.process_anchor(honest_anchor)

        malformed_tx_message = encode_wire_message(MSG_TYPE_TX, b"not-a-transaction")
        decoded = decode_wire_message(malformed_tx_message)
        tx_result, add_result = victim.process_raw_tx(decoded.payload)
        bad_anchor = anchor(
            shard_tree_bytes=b"\x02\x00\x00\x00",
            covered_fruit_hashes=(honest_fruit.block_hash(),),
            parent_anchor=genesis.block_hash(),
            timestamp_ms=3,
            nonce=23,
        )

        assert decoded.message_type == MSG_TYPE_TX
        assert tx_result.reason == "malformed_tx"
        assert add_result is None
        assert victim.status()["blocks"] == 0
        ordinary = _node(tmp_path / "ordinary", genesis=genesis)
        try:
            assert ordinary.process_anchor(genesis)
            assert ordinary.process_fruit(honest_fruit)
            assert ordinary.process_anchor(bad_anchor).reason == "bad_shard_tree"
        finally:
            ordinary.close()

        with pytest.raises(WireDecodeError, match="checksum"):
            decode_wire_message(malformed_tx_message[:-1] + bytes([malformed_tx_message[-1] ^ 1]))

        victim.sync_from(honest)

        assert victim.status()["blocks"] == honest.status()["blocks"]
        assert victim.get_block(honest_fruit.block_hash()) == honest_fruit.serialize()
        assert victim.get_block(honest_anchor.block_hash()) == honest_anchor.serialize()
    finally:
        honest.close()
        victim.close()


def _node(data_dir: Path, *, genesis: Anchor) -> TensorPowNode:
    return TensorPowNode(
        TensorPowConfig(data_dir=data_dir, expected_genesis_hash=genesis.block_hash()),
        pow_verifier=lambda _header, _target, _backend: True,
    )
