"""libp2p host integration and TensorPoW wire messages."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Final

from tensorpow.crypto.hash import hash_bytes
from tensorpow.crypto.signatures import ED25519_PRIVATE_KEY_BYTES
from tensorpow.mempool.shard_tree import ShardId, require_shard_id

WIRE_MAGIC: Final[bytes] = bytes.fromhex("54504f57")
WIRE_MAGIC_BYTES: Final[int] = 4
WIRE_CHECKSUM_BYTES: Final[int] = 4
WIRE_MAX_PAYLOAD_BYTES: Final[int] = 16_777_216

MSG_TYPE_FRUIT: Final[int] = 0x0001
MSG_TYPE_ANCHOR: Final[int] = 0x0002
MSG_TYPE_TX: Final[int] = 0x0003
MSG_TYPE_GRAPHENE_SKETCH: Final[int] = 0x0004
MSG_TYPE_ERLAY_SKETCH: Final[int] = 0x0005
MSG_TYPE_DAS_REQUEST: Final[int] = 0x0006
MSG_TYPE_DAS_RESPONSE: Final[int] = 0x0007
KNOWN_MESSAGE_TYPES: Final[frozenset[int]] = frozenset(
    (
        MSG_TYPE_FRUIT,
        MSG_TYPE_ANCHOR,
        MSG_TYPE_TX,
        MSG_TYPE_GRAPHENE_SKETCH,
        MSG_TYPE_ERLAY_SKETCH,
        MSG_TYPE_DAS_REQUEST,
        MSG_TYPE_DAS_RESPONSE,
    )
)

TOPIC_FRUITS: Final[str] = "tensorpow/fruits/main"
TOPIC_ANCHORS: Final[str] = "tensorpow/anchors/main"
TOPIC_TXS_PREFIX: Final[str] = "tensorpow/txs/"
TOPIC_TXS_SUFFIX: Final[str] = "/main"

DEFAULT_GOSSIPSUB_DEGREE: Final[int] = 6
DEFAULT_GOSSIPSUB_DEGREE_LOW: Final[int] = 4
DEFAULT_GOSSIPSUB_DEGREE_HIGH: Final[int] = 12


class WireDecodeError(ValueError):
    """Raised when a TensorPoW wire message is malformed."""


class LibP2PNodeError(RuntimeError):
    """Raised when libp2p node lifecycle operations are invalid."""


@dataclass(frozen=True, slots=True)
class WireMessage:
    """Decoded TensorPoW wire message."""

    message_type: int
    payload: bytes

    def __post_init__(self) -> None:
        _require_message_type(self.message_type)
        _require_payload(self.payload)


@dataclass(frozen=True, slots=True)
class NodeIdentity:
    """Persistable Ed25519 identity for a libp2p host."""

    private_key_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.private_key_bytes, bytes):
            raise TypeError("private_key_bytes must be bytes")
        if len(self.private_key_bytes) != ED25519_PRIVATE_KEY_BYTES:
            raise ValueError(f"private_key_bytes must be {ED25519_PRIVATE_KEY_BYTES} bytes")

    @classmethod
    def generate(cls) -> NodeIdentity:
        """Generate a new libp2p Ed25519 identity."""

        create_new_key_pair = _libp2p_ed25519_module().create_new_key_pair
        return cls(create_new_key_pair().private_key.to_bytes())

    def key_pair(self) -> Any:
        """Return the py-libp2p keypair for this identity."""

        ed25519 = _libp2p_ed25519_module()
        keys = _libp2p_keys_module()
        private_key = ed25519.Ed25519PrivateKey.from_bytes(self.private_key_bytes)
        return keys.KeyPair(private_key, private_key.get_public_key())

    def public_key_bytes(self) -> bytes:
        """Return raw public-key bytes."""

        public_key_bytes = self.key_pair().public_key.to_bytes()
        if not isinstance(public_key_bytes, bytes):
            raise TypeError("libp2p public key bytes must be bytes")
        return public_key_bytes

    def peer_id(self) -> str:
        """Return the libp2p peer id derived from this identity."""

        libp2p = _libp2p_module()
        host = libp2p.new_host(key_pair=self.key_pair(), muxer_preference="YAMUX")
        return str(host.get_id())


class LibP2PNode:
    """Small managed py-libp2p GossipSub host used by TensorPoW nodes."""

    def __init__(
        self,
        *,
        identity: NodeIdentity | None = None,
        listen_addrs: tuple[str, ...] = ("/ip4/127.0.0.1/tcp/0",),
        enable_quic: bool = False,
        enable_kademlia: bool = True,
    ) -> None:
        if not isinstance(listen_addrs, tuple):
            raise TypeError("listen_addrs must be a tuple")
        if not listen_addrs:
            raise ValueError("listen_addrs must be nonempty")
        for addr in listen_addrs:
            if not isinstance(addr, str):
                raise TypeError("listen_addrs entries must be str")
        self.identity = NodeIdentity.generate() if identity is None else identity
        if not isinstance(self.identity, NodeIdentity):
            raise TypeError("identity must be NodeIdentity")
        self.listen_addrs = listen_addrs
        self.enable_quic = _require_bool("enable_quic", enable_quic)
        self.enable_kademlia = _require_bool("enable_kademlia", enable_kademlia)

        self._host: Any | None = None
        self._pubsub: Any | None = None
        self._dht: Any | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._subscriptions: dict[str, Any] = {}

    async def __aenter__(self) -> LibP2PNode:
        await self.start()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Start the libp2p host, GossipSub service, and Kademlia service."""

        if self._host is not None:
            raise LibP2PNodeError("node is already started")

        libp2p = _libp2p_module()
        multiaddr = _multiaddr_module()
        gossipsub = _gossipsub_module()
        pubsub_module = _pubsub_module()
        services = _async_service_module()

        listen_multiaddrs = [multiaddr.Multiaddr(addr) for addr in self.listen_addrs]
        host = libp2p.new_host(
            key_pair=self.identity.key_pair(),
            listen_addrs=listen_multiaddrs,
            muxer_preference="YAMUX",
            enable_quic=self.enable_quic,
        )
        router = gossipsub.GossipSub(
            [gossipsub.PROTOCOL_ID_V12],
            DEFAULT_GOSSIPSUB_DEGREE,
            DEFAULT_GOSSIPSUB_DEGREE_LOW,
            DEFAULT_GOSSIPSUB_DEGREE_HIGH,
            heartbeat_interval=1,
        )
        pubsub = pubsub_module.Pubsub(host, router, strict_signing=False)

        exit_stack = AsyncExitStack()
        await exit_stack.enter_async_context(host.run(listen_multiaddrs))
        await exit_stack.enter_async_context(services.background_trio_service(pubsub))
        if self.enable_kademlia:
            kad_module = _kad_dht_module()
            dht = kad_module.KadDHT(host, kad_module.DHTMode.SERVER)
            await exit_stack.enter_async_context(services.background_trio_service(dht))
            self._dht = dht

        self._host = host
        self._pubsub = pubsub
        self._exit_stack = exit_stack

    async def stop(self) -> None:
        """Stop all libp2p services."""

        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._subscriptions.clear()
        self._host = None
        self._pubsub = None
        self._dht = None
        self._exit_stack = None

    @property
    def peer_id(self) -> str:
        """Return this node's peer id string."""

        return str(self._require_host().get_id())

    def addrs(self) -> tuple[Any, ...]:
        """Return active listen addresses including the peer id."""

        return tuple(self._require_host().get_addrs())

    def peer_info(self) -> Any:
        """Return py-libp2p PeerInfo for this node."""

        return _peerinfo_module().PeerInfo(self._require_host().get_id(), self.addrs())

    async def connect(self, peer_info: Any) -> None:
        """Connect to a peer and record it in the local DHT routing table."""

        await self._require_host().connect(peer_info)
        if self._dht is not None:
            await self._dht.add_peer(peer_info.peer_id)

    async def find_peer(self, peer_id: Any) -> Any | None:
        """Find a peer through the local Kademlia routing state."""

        if self._dht is None:
            raise LibP2PNodeError("Kademlia is disabled")
        return await self._dht.find_peer(peer_id)

    async def subscribe(self, topic: str) -> None:
        """Subscribe to a GossipSub topic."""

        _require_topic(topic)
        self._subscriptions[topic] = await self._require_pubsub().subscribe(topic)

    async def publish(self, topic: str, payload: bytes) -> None:
        """Publish payload bytes inside the topic's TensorPoW wire envelope."""

        _require_topic(topic)
        wire_payload = encode_wire_message(message_type_for_topic(topic), payload)
        await self._require_pubsub().publish(topic, wire_payload)

    async def next_message(self, topic: str, *, timeout_seconds: float = 5.0) -> bytes:
        """Return the next decoded payload for a subscribed topic."""

        _require_topic(topic)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        subscription = self._subscriptions.get(topic)
        if subscription is None:
            raise LibP2PNodeError("topic is not subscribed")
        trio = _trio_module()
        with trio.fail_after(timeout_seconds):
            message = await subscription.get()
        if topic not in message.topicIDs:
            raise LibP2PNodeError("received message for unexpected topic")
        try:
            wire_message = decode_wire_message(bytes(message.data))
        except WireDecodeError as error:
            raise LibP2PNodeError("received malformed wire message") from error
        expected_message_type = message_type_for_topic(topic)
        if wire_message.message_type != expected_message_type:
            raise LibP2PNodeError("received wire message_type for unexpected topic")
        return wire_message.payload

    def _require_host(self) -> Any:
        if self._host is None:
            raise LibP2PNodeError("node is not started")
        return self._host

    def _require_pubsub(self) -> Any:
        if self._pubsub is None:
            raise LibP2PNodeError("node is not started")
        return self._pubsub


def topic_for_shard_txs(shard_id: ShardId) -> str:
    """Return the canonical tx GossipSub topic for a shard."""

    shard_id = require_shard_id(shard_id)
    return f"{TOPIC_TXS_PREFIX}{shard_id:08x}{TOPIC_TXS_SUFFIX}"


def message_type_for_topic(topic: str) -> int:
    """Return the expected wire message type for a canonical GossipSub topic."""

    _require_topic(topic)
    if topic == TOPIC_FRUITS:
        return MSG_TYPE_FRUIT
    if topic == TOPIC_ANCHORS:
        return MSG_TYPE_ANCHOR
    if _is_shard_tx_topic(topic):
        return MSG_TYPE_TX
    raise ValueError("topic has no wire message type")


def encode_wire_message(message_type: int, payload: bytes) -> bytes:
    """Encode one TensorPoW wire message."""

    _require_message_type(message_type)
    _require_payload(payload)
    return b"".join(
        (
            WIRE_MAGIC,
            message_type.to_bytes(2, "little"),
            len(payload).to_bytes(4, "little"),
            payload,
            _wire_checksum(payload),
        )
    )


def decode_wire_message(data: bytes) -> WireMessage:
    """Decode and validate one TensorPoW wire message."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    header_len = WIRE_MAGIC_BYTES + 2 + 4
    min_len = header_len + WIRE_CHECKSUM_BYTES
    if len(data) < min_len:
        raise WireDecodeError("wire message is truncated")
    if data[:WIRE_MAGIC_BYTES] != WIRE_MAGIC:
        raise WireDecodeError("wire magic is invalid")
    message_type = int.from_bytes(data[WIRE_MAGIC_BYTES : WIRE_MAGIC_BYTES + 2], "little")
    try:
        _require_message_type(message_type)
    except (TypeError, ValueError) as exc:
        raise WireDecodeError("wire message_type is unknown") from exc
    payload_len_offset = WIRE_MAGIC_BYTES + 2
    payload_len = int.from_bytes(data[payload_len_offset : payload_len_offset + 4], "little")
    if payload_len > WIRE_MAX_PAYLOAD_BYTES:
        raise WireDecodeError("wire payload exceeds maximum size")
    expected_len = header_len + payload_len + WIRE_CHECKSUM_BYTES
    if len(data) != expected_len:
        raise WireDecodeError("wire payload length mismatch")
    payload = data[header_len : header_len + payload_len]
    checksum = data[header_len + payload_len :]
    if checksum != _wire_checksum(payload):
        raise WireDecodeError("wire checksum mismatch")
    return WireMessage(message_type=message_type, payload=payload)


def _wire_checksum(payload: bytes) -> bytes:
    return hash_bytes(payload)[:WIRE_CHECKSUM_BYTES]


def _require_message_type(message_type: int) -> None:
    if not isinstance(message_type, int) or isinstance(message_type, bool):
        raise TypeError("message_type must be int")
    if message_type not in KNOWN_MESSAGE_TYPES:
        raise ValueError("message_type is unknown")


def _require_payload(payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) > WIRE_MAX_PAYLOAD_BYTES:
        raise ValueError("payload exceeds WIRE_MAX_PAYLOAD_BYTES")


def _require_topic(topic: str) -> None:
    if not isinstance(topic, str):
        raise TypeError("topic must be str")
    if topic == "":
        raise ValueError("topic must not be empty")


def _is_shard_tx_topic(topic: str) -> bool:
    if not topic.startswith(TOPIC_TXS_PREFIX) or not topic.endswith(TOPIC_TXS_SUFFIX):
        return False
    encoded = topic[len(TOPIC_TXS_PREFIX) : -len(TOPIC_TXS_SUFFIX)]
    if len(encoded) != 8:
        return False
    try:
        int(encoded, 16)
    except ValueError:
        return False
    return encoded.lower() == encoded


def _require_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _libp2p_module() -> Any:
    import libp2p

    return libp2p


def _libp2p_ed25519_module() -> Any:
    from libp2p.crypto import ed25519

    return ed25519


def _libp2p_keys_module() -> Any:
    from libp2p.crypto import keys

    return keys


def _multiaddr_module() -> Any:
    import multiaddr  # type: ignore[import-untyped]

    return multiaddr


def _gossipsub_module() -> Any:
    from libp2p.pubsub import gossipsub

    return gossipsub


def _pubsub_module() -> Any:
    from libp2p.pubsub import pubsub

    return pubsub


def _async_service_module() -> Any:
    from libp2p.tools import async_service

    return async_service


def _kad_dht_module() -> Any:
    from libp2p.kad_dht import kad_dht

    return kad_dht


def _peerinfo_module() -> Any:
    from libp2p.peer import peerinfo

    return peerinfo


def _trio_module() -> Any:
    import trio

    return trio


__all__ = [
    "KNOWN_MESSAGE_TYPES",
    "MSG_TYPE_ANCHOR",
    "MSG_TYPE_DAS_REQUEST",
    "MSG_TYPE_DAS_RESPONSE",
    "MSG_TYPE_ERLAY_SKETCH",
    "MSG_TYPE_FRUIT",
    "MSG_TYPE_GRAPHENE_SKETCH",
    "MSG_TYPE_TX",
    "TOPIC_ANCHORS",
    "TOPIC_FRUITS",
    "TOPIC_TXS_PREFIX",
    "TOPIC_TXS_SUFFIX",
    "WIRE_CHECKSUM_BYTES",
    "WIRE_MAGIC",
    "WIRE_MAGIC_BYTES",
    "WIRE_MAX_PAYLOAD_BYTES",
    "LibP2PNode",
    "LibP2PNodeError",
    "NodeIdentity",
    "WireDecodeError",
    "WireMessage",
    "decode_wire_message",
    "encode_wire_message",
    "message_type_for_topic",
    "topic_for_shard_txs",
]
