"""Tests for TensorPoW libp2p integration and wire messages."""

from __future__ import annotations

import pytest
import trio

from tensorpow.mempool.shard_tree import encode_shard_id
from tensorpow.net import (
    MSG_TYPE_FRUIT,
    MSG_TYPE_TX,
    TOPIC_FRUITS,
    LibP2PNode,
    LibP2PNodeError,
    NodeIdentity,
    WireDecodeError,
    decode_wire_message,
    encode_wire_message,
    topic_for_shard_txs,
)


def test_wire_message_round_trip_and_topic_names() -> None:
    payload = b"fruit-bytes"
    encoded = encode_wire_message(MSG_TYPE_FRUIT, payload)
    decoded = decode_wire_message(encoded)

    assert decoded.message_type == MSG_TYPE_FRUIT
    assert decoded.payload == payload
    assert topic_for_shard_txs(encode_shard_id(3, 5)) == "tensorpow/txs/00030005/main"


def test_wire_message_rejects_malformed_inputs() -> None:
    encoded = encode_wire_message(MSG_TYPE_TX, b"tx")

    with pytest.raises(WireDecodeError, match="truncated"):
        decode_wire_message(encoded[:5])
    with pytest.raises(WireDecodeError, match="magic"):
        decode_wire_message(b"BAD!" + encoded[4:])
    with pytest.raises(ValueError, match="unknown"):
        encode_wire_message(0xFFFF, b"")
    unknown_type = bytearray(encoded)
    unknown_type[4:6] = (0xFFFF).to_bytes(2, "little")
    with pytest.raises(WireDecodeError, match="message_type"):
        decode_wire_message(bytes(unknown_type))
    bad_length = bytearray(encoded)
    bad_length[6:10] = (999).to_bytes(4, "little")
    with pytest.raises(WireDecodeError, match="length"):
        decode_wire_message(bytes(bad_length))
    with pytest.raises(WireDecodeError, match="checksum"):
        decode_wire_message(encoded[:-1] + bytes([encoded[-1] ^ 1]))


def test_node_identity_peer_id_is_persistent() -> None:
    identity = NodeIdentity.generate()

    assert identity.peer_id() == NodeIdentity(identity.private_key_bytes).peer_id()
    assert (
        identity.public_key_bytes() == NodeIdentity(identity.private_key_bytes).public_key_bytes()
    )
    with pytest.raises(ValueError, match="32 bytes"):
        NodeIdentity(b"short")


def test_two_libp2p_nodes_connect_find_peer_and_gossipsub_roundtrip() -> None:
    trio.run(_libp2p_roundtrip)


async def _libp2p_roundtrip() -> None:
    identity = NodeIdentity.generate()
    async with LibP2PNode() as publisher, LibP2PNode(identity=identity) as subscriber:
        await publisher.connect(subscriber.peer_info())
        found = await publisher.find_peer(subscriber.peer_info().peer_id)
        assert found is not None
        assert str(found.peer_id) == subscriber.peer_id

        await subscriber.subscribe(TOPIC_FRUITS)
        await trio.sleep(0.5)
        payload = encode_wire_message(MSG_TYPE_FRUIT, b"hello")
        await publisher.publish(TOPIC_FRUITS, payload)
        received = await subscriber.next_message(TOPIC_FRUITS, timeout_seconds=5)
        assert decode_wire_message(received).payload == b"hello"

    async with LibP2PNode(identity=identity) as restarted:
        assert restarted.peer_id == identity.peer_id()
        with pytest.raises(LibP2PNodeError, match="subscribed"):
            await restarted.next_message(TOPIC_FRUITS, timeout_seconds=0.01)
