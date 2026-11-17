"""libp2p host integration and TensorPoW wire messages."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Final

from tensorpow.crypto.hash import hash_bytes
from tensorpow.crypto.signatures import ED25519_PRIVATE_KEY_BYTES
from tensorpow.mempool.shard_tree import ShardId, require_shard_id
from tensorpow.net.admission import AdmissionConfig, PeerAdmission

WIRE_MAGIC: Final[bytes] = bytes.fromhex("54504f57")
WIRE_MAGIC_BYTES: Final[int] = 4
WIRE_CHECKSUM_BYTES: Final[int] = 4
WIRE_MAX_PAYLOAD_BYTES: Final[int] = 4_194_304

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

# Per-type payload ceilings, close to the largest canonical object of each type.
MAX_FRUIT_WIRE_PAYLOAD_BYTES: Final[int] = 16_384
MAX_TX_WIRE_PAYLOAD_BYTES: Final[int] = 8_192
MAX_ANCHOR_WIRE_PAYLOAD_BYTES: Final[int] = 2_097_152
MAX_PAYLOAD_BYTES_BY_TYPE: Final[dict[int, int]] = {
    MSG_TYPE_FRUIT: MAX_FRUIT_WIRE_PAYLOAD_BYTES,
    MSG_TYPE_ANCHOR: MAX_ANCHOR_WIRE_PAYLOAD_BYTES,
    MSG_TYPE_TX: MAX_TX_WIRE_PAYLOAD_BYTES,
    MSG_TYPE_GRAPHENE_SKETCH: WIRE_MAX_PAYLOAD_BYTES,
    MSG_TYPE_ERLAY_SKETCH: WIRE_MAX_PAYLOAD_BYTES,
    MSG_TYPE_DAS_REQUEST: WIRE_MAX_PAYLOAD_BYTES,
    MSG_TYPE_DAS_RESPONSE: WIRE_MAX_PAYLOAD_BYTES,
}
WIRE_ENVELOPE_OVERHEAD_BYTES: Final[int] = WIRE_MAGIC_BYTES + 2 + 4 + WIRE_CHECKSUM_BYTES
MAX_GOSSIP_RPC_BYTES: Final[int] = MAX_ANCHOR_WIRE_PAYLOAD_BYTES + 65_536

# Topics are scoped by chain id so mainnet and testnet gossip can never mix.
TOPIC_CHAIN_MAINNET: Final[str] = "tensorpow-mainnet"
TOPIC_CHAIN_TESTNET: Final[str] = "tensorpow-testnet"
TOPIC_CHAINS: Final[frozenset[str]] = frozenset((TOPIC_CHAIN_MAINNET, TOPIC_CHAIN_TESTNET))
TOPIC_FRUITS_PREFIX: Final[str] = "tensorpow/fruits/"
TOPIC_ANCHORS_PREFIX: Final[str] = "tensorpow/anchors/"
TOPIC_TXS_PREFIX: Final[str] = "tensorpow/txs/"
TOPIC_FRUITS: Final[str] = TOPIC_FRUITS_PREFIX + TOPIC_CHAIN_MAINNET
TOPIC_ANCHORS: Final[str] = TOPIC_ANCHORS_PREFIX + TOPIC_CHAIN_MAINNET
TOPIC_TXS_SUFFIX: Final[str] = "/" + TOPIC_CHAIN_MAINNET

DEFAULT_GOSSIPSUB_DEGREE: Final[int] = 6
DEFAULT_GOSSIPSUB_DEGREE_LOW: Final[int] = 4
DEFAULT_GOSSIPSUB_DEGREE_HIGH: Final[int] = 12
DEFAULT_MAX_CONNECTIONS: Final[int] = 128
DEFAULT_MAX_CONNECTIONS_PER_PEER: Final[int] = 2

type PayloadValidator = Callable[[str, int, bytes], bool]


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
        _require_payload(self.payload, self.message_type)


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
        chain_id: str = TOPIC_CHAIN_MAINNET,
        dht_server: bool = False,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_connections_per_peer: int = DEFAULT_MAX_CONNECTIONS_PER_PEER,
        payload_validator: PayloadValidator | None = None,
        admission: PeerAdmission | None = None,
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
        self.chain_id = _require_chain(chain_id)
        self.dht_server = _require_bool("dht_server", dht_server)
        self.max_connections = _require_positive_int("max_connections", max_connections)
        self.max_connections_per_peer = _require_positive_int(
            "max_connections_per_peer", max_connections_per_peer
        )
        self.payload_validator = payload_validator
        self.admission = (
            PeerAdmission(AdmissionConfig(malformed_strike_limit=4))
            if admission is None
            else admission
        )

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
        connection_config = _network_config_module().ConnectionConfig(
            max_connections=self.max_connections,
            max_connections_per_peer=self.max_connections_per_peer,
            min_connections=min(8, self.max_connections),
            low_watermark=min(16, self.max_connections),
            high_watermark=max(min(16, self.max_connections), self.max_connections * 3 // 4),
        )
        host = libp2p.new_host(
            key_pair=self.identity.key_pair(),
            listen_addrs=listen_multiaddrs,
            muxer_preference="YAMUX",
            enable_quic=self.enable_quic,
            connection_config=connection_config,
        )
        router = gossipsub.GossipSub(
            [gossipsub.PROTOCOL_ID_V12],
            DEFAULT_GOSSIPSUB_DEGREE,
            DEFAULT_GOSSIPSUB_DEGREE_LOW,
            DEFAULT_GOSSIPSUB_DEGREE_HIGH,
            heartbeat_interval=1,
        )
        pubsub = pubsub_module.Pubsub(
            host,
            router,
            strict_signing=True,
            max_inbound_rpc_size=MAX_GOSSIP_RPC_BYTES,
        )

        exit_stack = AsyncExitStack()
        await exit_stack.enter_async_context(host.run(listen_multiaddrs))
        await exit_stack.enter_async_context(services.background_trio_service(pubsub))
        if self.enable_kademlia:
            kad_module = _kad_dht_module()
            mode = kad_module.DHTMode.SERVER if self.dht_server else kad_module.DHTMode.CLIENT
            dht = kad_module.KadDHT(host, mode)
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

    def topic_fruits(self) -> str:
        return topic_fruits(self.chain_id)

    def topic_anchors(self) -> str:
        return topic_anchors(self.chain_id)

    def topic_txs(self, shard_id: ShardId) -> str:
        return topic_for_shard_txs(shard_id, chain_id=self.chain_id)

    async def subscribe(self, topic: str) -> None:
        """Subscribe to a GossipSub topic of this node's chain with a gossip validator."""

        _require_topic(topic)
        if topic_chain(topic) != self.chain_id:
            raise LibP2PNodeError("topic does not belong to this node's chain")
        pubsub = self._require_pubsub()
        pubsub.set_topic_validator(topic, self._make_validator(topic), False)
        self._subscriptions[topic] = await pubsub.subscribe(topic)

    def _make_validator(self, topic: str) -> Callable[[Any, Any], bool]:
        expected_type = message_type_for_topic(topic)

        def validate(peer_id: Any, message: Any) -> bool:
            peer = str(peer_id)
            if self.admission.is_banned(peer):
                self._blacklist(peer_id)
                return False
            if validate_gossip_payload(
                topic,
                bytes(message.data),
                expected_type=expected_type,
                peer_id=peer,
                payload_validator=self.payload_validator,
            ):
                return True
            self.admission.strike(peer)
            if self.admission.is_banned(peer):
                self._blacklist(peer_id)
            return False

        return validate

    def _blacklist(self, peer_id: Any) -> None:
        pubsub = self._pubsub
        if pubsub is not None and not pubsub.is_peer_blacklisted(peer_id):
            pubsub.add_to_blacklist(peer_id)

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


def topic_fruits(chain_id: str = TOPIC_CHAIN_MAINNET) -> str:
    """Return the fruit GossipSub topic for ``chain_id``."""

    return TOPIC_FRUITS_PREFIX + _require_chain(chain_id)


def topic_anchors(chain_id: str = TOPIC_CHAIN_MAINNET) -> str:
    """Return the anchor GossipSub topic for ``chain_id``."""

    return TOPIC_ANCHORS_PREFIX + _require_chain(chain_id)


def topic_for_shard_txs(shard_id: ShardId, *, chain_id: str = TOPIC_CHAIN_MAINNET) -> str:
    """Return the canonical tx GossipSub topic for a shard on ``chain_id``."""

    shard_id = require_shard_id(shard_id)
    return f"{TOPIC_TXS_PREFIX}{shard_id:08x}/{_require_chain(chain_id)}"


def topic_chain(topic: str) -> str:
    """Return the chain id a canonical topic belongs to."""

    _require_topic(topic)
    for prefix in (TOPIC_FRUITS_PREFIX, TOPIC_ANCHORS_PREFIX):
        if topic.startswith(prefix) and topic[len(prefix) :] in TOPIC_CHAINS:
            return topic[len(prefix) :]
    parsed = _parse_shard_tx_topic(topic)
    if parsed is not None:
        return parsed
    raise ValueError("topic is not a canonical TensorPoW topic")


def message_type_for_topic(topic: str) -> int:
    """Return the expected wire message type for a canonical GossipSub topic."""

    _require_topic(topic)
    if topic.startswith(TOPIC_FRUITS_PREFIX) and topic[len(TOPIC_FRUITS_PREFIX) :] in TOPIC_CHAINS:
        return MSG_TYPE_FRUIT
    if (
        topic.startswith(TOPIC_ANCHORS_PREFIX)
        and topic[len(TOPIC_ANCHORS_PREFIX) :] in TOPIC_CHAINS
    ):
        return MSG_TYPE_ANCHOR
    if _parse_shard_tx_topic(topic) is not None:
        return MSG_TYPE_TX
    raise ValueError("topic has no wire message type")


def validate_gossip_payload(
    topic: str,
    data: bytes,
    *,
    expected_type: int | None = None,
    peer_id: str = "",
    payload_validator: PayloadValidator | None = None,
) -> bool:
    """Return whether a gossip message may be accepted and forwarded.

    Checks the wire envelope, the topic/message-type binding and the per-type size
    ceiling, then runs the optional consensus pre-check supplied by the node.
    """

    try:
        message_type = message_type_for_topic(topic) if expected_type is None else expected_type
        if len(data) > MAX_PAYLOAD_BYTES_BY_TYPE[message_type] + WIRE_ENVELOPE_OVERHEAD_BYTES:
            return False
        wire_message = decode_wire_message(data)
    except (TypeError, ValueError, KeyError):
        return False
    if wire_message.message_type != message_type:
        return False
    if payload_validator is None:
        return True
    try:
        return bool(payload_validator(peer_id, message_type, wire_message.payload))
    except Exception:
        return False


def encode_wire_message(message_type: int, payload: bytes) -> bytes:
    """Encode one TensorPoW wire message."""

    _require_message_type(message_type)
    _require_payload(payload, message_type)
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
    if payload_len > MAX_PAYLOAD_BYTES_BY_TYPE[message_type]:
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


def _require_payload(payload: bytes, message_type: int | None = None) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    limit = (
        WIRE_MAX_PAYLOAD_BYTES
        if message_type is None
        else MAX_PAYLOAD_BYTES_BY_TYPE.get(message_type, WIRE_MAX_PAYLOAD_BYTES)
    )
    if len(payload) > limit:
        raise ValueError("payload exceeds the maximum size for its message type")


def _require_topic(topic: str) -> None:
    if not isinstance(topic, str):
        raise TypeError("topic must be str")
    if topic == "":
        raise ValueError("topic must not be empty")


def _parse_shard_tx_topic(topic: str) -> str | None:
    if not topic.startswith(TOPIC_TXS_PREFIX):
        return None
    encoded, separator, chain_id = topic[len(TOPIC_TXS_PREFIX) :].partition("/")
    if separator != "/" or chain_id not in TOPIC_CHAINS or len(encoded) != 8:
        return None
    if encoded.lower() != encoded:
        return None
    try:
        require_shard_id(int(encoded, 16))
    except ValueError:
        return None
    return chain_id


def _require_chain(chain_id: str) -> str:
    if chain_id not in TOPIC_CHAINS:
        raise ValueError("chain_id must be a TensorPoW mainnet or testnet chain id")
    return chain_id


def _require_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


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


def _network_config_module() -> Any:
    from libp2p.network import config

    return config


def _gossipsub_module() -> Any:
    from libp2p.pubsub import gossipsub

    return gossipsub


def _pubsub_module() -> Any:
    from libp2p.pubsub import pubsub

    return pubsub


def _async_service_module() -> Any:
    from libp2p.tools import anyio_service

    return anyio_service


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
    "MAX_ANCHOR_WIRE_PAYLOAD_BYTES",
    "MAX_FRUIT_WIRE_PAYLOAD_BYTES",
    "MAX_PAYLOAD_BYTES_BY_TYPE",
    "MAX_TX_WIRE_PAYLOAD_BYTES",
    "MSG_TYPE_ANCHOR",
    "MSG_TYPE_DAS_REQUEST",
    "MSG_TYPE_DAS_RESPONSE",
    "MSG_TYPE_ERLAY_SKETCH",
    "MSG_TYPE_FRUIT",
    "MSG_TYPE_GRAPHENE_SKETCH",
    "MSG_TYPE_TX",
    "TOPIC_ANCHORS",
    "TOPIC_CHAIN_MAINNET",
    "TOPIC_CHAIN_TESTNET",
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
    "topic_anchors",
    "topic_chain",
    "topic_for_shard_txs",
    "topic_fruits",
    "validate_gossip_payload",
]
