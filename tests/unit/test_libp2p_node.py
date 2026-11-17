"""Tests for TensorPoW libp2p integration and wire messages."""

from __future__ import annotations

import pytest
import trio

from tensorpow.net import (
    MSG_TYPE_FRUIT,
    MSG_TYPE_TX,
    TOPIC_FRUITS,
    WIRE_MAX_PAYLOAD_BYTES,
    LibP2PNode,
    LibP2PNodeError,
    NodeIdentity,
    WireDecodeError,
    decode_wire_message,
    encode_wire_message,
    message_type_for_topic,
    topic_for_shard_txs,
)
from tensorpow.net.libp2p_node import (
    MAX_FRUIT_WIRE_PAYLOAD_BYTES,
    MAX_TX_WIRE_PAYLOAD_BYTES,
    TOPIC_CHAIN_MAINNET,
    TOPIC_CHAIN_TESTNET,
    topic_anchors,
    topic_chain,
    topic_fruits,
    validate_gossip_payload,
)


def test_wire_message_round_trip_and_topic_names() -> None:
    payload = b"fruit-bytes"
    encoded = encode_wire_message(MSG_TYPE_FRUIT, payload)
    decoded = decode_wire_message(encoded)

    assert decoded.message_type == MSG_TYPE_FRUIT
    assert decoded.payload == payload
    assert topic_for_shard_txs(0x00030005) == "tensorpow/txs/00030005/tensorpow-mainnet"
    assert message_type_for_topic(TOPIC_FRUITS) == MSG_TYPE_FRUIT
    assert message_type_for_topic(topic_for_shard_txs(0x00030005)) == MSG_TYPE_TX


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


def test_publish_wraps_payload_in_topic_wire_message() -> None:
    trio.run(_publish_wraps_payload_in_topic_wire_message)


async def _publish_wraps_payload_in_topic_wire_message() -> None:
    node = LibP2PNode()
    pubsub = _FakePubSub()
    node._pubsub = pubsub

    await node.publish(TOPIC_FRUITS, b"hello")

    assert pubsub.topic == TOPIC_FRUITS
    decoded = decode_wire_message(pubsub.payload)
    assert decoded.message_type == MSG_TYPE_FRUIT
    assert decoded.payload == b"hello"


def test_next_message_decodes_wire_envelope_and_rejects_topic_mismatch() -> None:
    trio.run(_next_message_decodes_wire_envelope_and_rejects_topic_mismatch)


async def _next_message_decodes_wire_envelope_and_rejects_topic_mismatch() -> None:
    node = LibP2PNode()
    node._subscriptions[TOPIC_FRUITS] = _FakeSubscription(
        TOPIC_FRUITS,
        encode_wire_message(MSG_TYPE_FRUIT, b"hello"),
    )

    assert await node.next_message(TOPIC_FRUITS, timeout_seconds=1) == b"hello"

    node._subscriptions[TOPIC_FRUITS] = _FakeSubscription(
        TOPIC_FRUITS,
        encode_wire_message(MSG_TYPE_TX, b"wrong-topic"),
    )
    with pytest.raises(LibP2PNodeError, match="message_type"):
        await node.next_message(TOPIC_FRUITS, timeout_seconds=1)


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
        await publisher.publish(TOPIC_FRUITS, b"hello")
        received = await subscriber.next_message(TOPIC_FRUITS, timeout_seconds=5)
        assert received == b"hello"

    async with LibP2PNode(identity=identity) as restarted:
        assert restarted.peer_id == identity.peer_id()
        with pytest.raises(LibP2PNodeError, match="subscribed"):
            await restarted.next_message(TOPIC_FRUITS, timeout_seconds=0.01)


class _FakePubSub:
    def __init__(self) -> None:
        self.topic = ""
        self.payload = b""
        self.validators: dict[str, object] = {}
        self.blacklisted: set[str] = set()

    async def publish(self, topic: str, payload: bytes) -> None:
        self.topic = topic
        self.payload = payload

    def set_topic_validator(self, topic: str, validator: object, _is_async: bool) -> None:
        self.validators[topic] = validator

    async def subscribe(self, topic: str) -> _FakeSubscription:
        return _FakeSubscription(topic, b"")

    def is_peer_blacklisted(self, peer_id: str) -> bool:
        return peer_id in self.blacklisted

    def add_to_blacklist(self, peer_id: str) -> None:
        self.blacklisted.add(peer_id)


class _FakeSubscription:
    def __init__(self, topic: str, payload: bytes) -> None:
        self._message = _FakeMessage(topic, payload)

    async def get(self) -> _FakeMessage:
        return self._message


class _FakeMessage:
    def __init__(self, topic_or_payload: str | bytes, payload: bytes | None = None) -> None:
        if payload is None:
            assert isinstance(topic_or_payload, bytes)
            self.topicIDs: tuple[str, ...] = ()
            self.data = topic_or_payload
        else:
            assert isinstance(topic_or_payload, str)
            self.topicIDs = (topic_or_payload,)
            self.data = payload


def test_topics_are_chain_scoped_and_legacy_topics_are_rejected() -> None:
    assert topic_fruits(TOPIC_CHAIN_TESTNET) != topic_fruits(TOPIC_CHAIN_MAINNET)
    assert topic_anchors(TOPIC_CHAIN_TESTNET).endswith(TOPIC_CHAIN_TESTNET)
    assert topic_chain(topic_for_shard_txs(0x00030005, chain_id=TOPIC_CHAIN_TESTNET)) == (
        TOPIC_CHAIN_TESTNET
    )
    for legacy in ("tensorpow/fruits/main", "tensorpow/txs/00000000/main", "tensorpow/x/y"):
        with pytest.raises(ValueError):
            message_type_for_topic(legacy)
    with pytest.raises(ValueError):
        topic_fruits("someothernet")


def test_per_type_payload_caps_are_close_to_real_maxima() -> None:
    assert WIRE_MAX_PAYLOAD_BYTES < 16_777_216
    with pytest.raises(ValueError, match="maximum size"):
        encode_wire_message(MSG_TYPE_TX, bytes(MAX_TX_WIRE_PAYLOAD_BYTES + 1))
    oversized = bytearray(encode_wire_message(MSG_TYPE_TX, b"tx"))
    oversized[6:10] = (MAX_TX_WIRE_PAYLOAD_BYTES + 1).to_bytes(4, "little")
    with pytest.raises(WireDecodeError, match="maximum size"):
        decode_wire_message(bytes(oversized))
    assert encode_wire_message(MSG_TYPE_FRUIT, bytes(MAX_FRUIT_WIRE_PAYLOAD_BYTES))


def test_gossip_validator_checks_envelope_type_size_and_consensus_hook() -> None:
    seen: list[tuple[str, int, bytes]] = []

    def precheck(peer: str, message_type: int, payload: bytes) -> bool:
        seen.append((peer, message_type, payload))
        return payload == b"good"

    topic = topic_fruits()
    good = encode_wire_message(MSG_TYPE_FRUIT, b"good")
    assert validate_gossip_payload(topic, good, peer_id="p", payload_validator=precheck)
    assert not validate_gossip_payload(
        topic, encode_wire_message(MSG_TYPE_FRUIT, b"bad"), payload_validator=precheck
    )
    assert not validate_gossip_payload(topic, encode_wire_message(MSG_TYPE_TX, b"good"))
    assert not validate_gossip_payload(topic, good[:-1] + bytes([good[-1] ^ 1]))
    assert not validate_gossip_payload(topic, b"\x00" * (MAX_FRUIT_WIRE_PAYLOAD_BYTES + 64))
    assert seen[0] == ("p", MSG_TYPE_FRUIT, b"good")


def test_invalid_gossip_strikes_and_blacklists_the_forwarding_peer() -> None:
    node = LibP2PNode(chain_id=TOPIC_CHAIN_TESTNET)
    pubsub = _FakePubSub()
    node._pubsub = pubsub
    topic = node.topic_fruits()
    validate = node._make_validator(topic)
    junk = _FakeMessage(encode_wire_message(MSG_TYPE_TX, b"wrong type"))

    results = [validate("attacker", junk) for _ in range(6)]

    assert not any(results)
    assert pubsub.blacklisted == {"attacker"}
    assert validate("honest", _FakeMessage(encode_wire_message(MSG_TYPE_FRUIT, b"ok")))
    assert not validate("attacker", _FakeMessage(encode_wire_message(MSG_TYPE_FRUIT, b"ok")))


def test_subscribe_rejects_other_chain_topics_and_registers_validator() -> None:
    trio.run(_subscribe_rejects_other_chain_topics_and_registers_validator)


async def _subscribe_rejects_other_chain_topics_and_registers_validator() -> None:
    node = LibP2PNode(chain_id=TOPIC_CHAIN_TESTNET)
    pubsub = _FakePubSub()
    node._pubsub = pubsub
    with pytest.raises(LibP2PNodeError, match="chain"):
        await node.subscribe(topic_fruits(TOPIC_CHAIN_MAINNET))
    await node.subscribe(node.topic_fruits())
    assert node.topic_fruits() in pubsub.validators


def test_started_host_uses_strict_signing_client_dht_and_connection_limits() -> None:
    trio.run(_started_host_uses_hardened_defaults)


async def _started_host_uses_hardened_defaults() -> None:
    from libp2p.kad_dht.kad_dht import DHTMode

    async with LibP2PNode(max_connections=32) as node:
        assert node._pubsub is not None
        assert node._pubsub.strict_signing is True
        assert node._dht is not None
        assert node._dht.mode == DHTMode.CLIENT
        assert node._host is not None
        assert node._host.get_network().connection_config.max_connections == 32
