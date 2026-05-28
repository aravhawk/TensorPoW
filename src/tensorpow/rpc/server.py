"""JSON-RPC 2.0 HTTP server and TensorPoW method bindings."""

from __future__ import annotations

import json
import socket
from base64 import b64encode
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha1
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from json import JSONDecodeError
from math import isfinite
from secrets import token_hex
from threading import Lock
from typing import Final, Protocol, cast, runtime_checkable
from urllib.parse import parse_qs, urlparse

from tensorpow.chain.blocks import Anchor, Fruit
from tensorpow.consensus.finality import (
    FinalityTier,
    finality_tier_from_depths,
    satisfied_finality_tiers,
)
from tensorpow.crypto.address import AddressDecodeError, address_to_pubkey_hash
from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.mempool import Mempool, MempoolEntry, ShardTree, require_shard_id
from tensorpow.state import UTXO, UTXOSet
from tensorpow.tx import Transaction, TxDecodeError

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]
type JsonParams = dict[str, object]

JSONRPC_VERSION: Final[str] = "2.0"

PARSE_ERROR: Final[int] = -32700
INVALID_REQUEST: Final[int] = -32600
METHOD_NOT_FOUND: Final[int] = -32601
INVALID_PARAMS: Final[int] = -32602
INTERNAL_ERROR: Final[int] = -32603

MAX_HTTP_BODY_BYTES: Final[int] = 1_048_576
DEFAULT_EVENT_DRAIN_LIMIT: Final[int] = 100
MAX_EVENT_DRAIN_LIMIT: Final[int] = 1_000
WEBSOCKET_GUID: Final[str] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

WS_OPCODE_CONTINUATION: Final[int] = 0x0
WS_OPCODE_TEXT: Final[int] = 0x1
WS_OPCODE_BINARY: Final[int] = 0x2
WS_OPCODE_CLOSE: Final[int] = 0x8
WS_OPCODE_PING: Final[int] = 0x9
WS_OPCODE_PONG: Final[int] = 0xA
WS_CLOSE_PROTOCOL_ERROR: Final[int] = 1002

_PARAMS_OMITTED: Final[object] = object()


class JsonRpcError(Exception):
    """Error that is safe to return as a JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: JsonValue | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True, slots=True)
class Subscription:
    """A registered topic subscription."""

    subscription_id: str
    topic: str


@dataclass(frozen=True, slots=True)
class _WebSocketFrame:
    opcode: int
    payload: bytes


class SubscriptionHub:
    """Small in-process topic hub used by RPC subscribe and HTTP event draining."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscriptions: dict[str, Subscription] = {}
        self._events: dict[str, deque[JsonObject]] = {}

    def subscribe(self, topic: str) -> Subscription:
        """Create a subscription for ``topic`` and return its identifier."""

        topic = _require_topic(topic)
        subscription = Subscription(subscription_id=token_hex(16), topic=topic)
        with self._lock:
            self._subscriptions[subscription.subscription_id] = subscription
            self._events[subscription.subscription_id] = deque()
        return subscription

    def publish(self, topic: str, payload: JsonValue) -> int:
        """Append ``payload`` to every subscription for ``topic``."""

        topic = _require_topic(topic)
        event: JsonObject = {"topic": topic, "payload": payload}
        delivered = 0
        with self._lock:
            for subscription_id, subscription in self._subscriptions.items():
                if subscription.topic != topic:
                    continue
                self._events[subscription_id].append(dict(event))
                delivered += 1
        return delivered

    def drain(self, subscription_id: str, *, limit: int = DEFAULT_EVENT_DRAIN_LIMIT) -> JsonObject:
        """Return and remove up to ``limit`` pending events for one subscription."""

        subscription_id = _require_nonempty_string("subscription_id", subscription_id)
        limit = _require_event_limit(limit)
        with self._lock:
            subscription = self._subscriptions.get(subscription_id)
            if subscription is None:
                raise JsonRpcError(INVALID_PARAMS, "Unknown subscription")
            queue = self._events[subscription_id]
            events: list[JsonValue] = []
            while queue and len(events) < limit:
                events.append(queue.popleft())
        return {
            "subscription_id": subscription.subscription_id,
            "topic": subscription.topic,
            "events": events,
        }


@runtime_checkable
class RpcBackend(Protocol):
    """Backend interface consumed by the JSON-RPC method layer."""

    def getblock(self, block_hash: bytes) -> JsonObject:
        """Return a block lookup response."""

    def gettx(self, tx_id: bytes) -> JsonObject:
        """Return a transaction lookup response."""

    def sendrawtx(self, raw_tx: bytes) -> JsonObject:
        """Validate and relay a raw transaction."""

    def getbalance(self, owner_pubkey_hash: bytes, address: str) -> JsonObject:
        """Return the spendable balance for ``address``."""

    def getutxos(self, owner_pubkey_hash: bytes, address: str) -> JsonObject:
        """Return UTXOs for ``address``."""

    def getmempool(self, shard_id: int | None) -> JsonObject:
        """Return mempool entries."""

    def getshardtree(self) -> JsonObject:
        """Return the current shard tree."""

    def getfinality(self, block_hash: bytes) -> JsonObject:
        """Return local finality information for a block hash."""


class InMemoryRpcBackend:
    """Standalone backend useful for tests and lightweight embedding."""

    def __init__(
        self,
        *,
        mempool: Mempool | None = None,
        shard_tree: ShardTree | None = None,
        utxo_set: UTXOSet | None = None,
    ) -> None:
        self.shard_tree = ShardTree() if shard_tree is None else shard_tree
        self.utxo_set = UTXOSet() if utxo_set is None else utxo_set
        self.mempool = (
            Mempool(shard_tree=self.shard_tree, utxo_view=self.utxo_set)
            if mempool is None
            else mempool
        )
        self._blocks: dict[bytes, bytes] = {}
        self._finality: dict[bytes, tuple[int, int, bool]] = {}

    def put_block(self, block_hash: bytes, block_bytes: bytes) -> None:
        """Store canonical block bytes for ``getblock``."""

        _require_hash_bytes("block_hash", block_hash)
        self._blocks[block_hash] = _require_bytes("block_bytes", block_bytes)

    def set_finality(
        self,
        block_hash: bytes,
        *,
        blue_depth: int,
        anchor_depth: int,
        seen: bool = True,
    ) -> None:
        """Set local finality depths for ``block_hash``."""

        _require_hash_bytes("block_hash", block_hash)
        blue_depth = _require_nonnegative_int("blue_depth", blue_depth)
        anchor_depth = _require_nonnegative_int("anchor_depth", anchor_depth)
        if not isinstance(seen, bool):
            raise TypeError("seen must be bool")
        self._finality[block_hash] = (blue_depth, anchor_depth, seen)

    def getblock(self, block_hash: bytes) -> JsonObject:
        raw_block = self._blocks.get(block_hash)
        return _block_lookup_json(block_hash, raw_block)

    def gettx(self, tx_id: bytes) -> JsonObject:
        entry = self.mempool.get_entry(tx_id)
        if entry is not None:
            return _tx_lookup_json(tx_id, entry.tx, in_mempool=True, entry=entry)
        return _tx_lookup_json(tx_id, None, in_mempool=False)

    def sendrawtx(self, raw_tx: bytes) -> JsonObject:
        try:
            tx = Transaction.from_bytes(raw_tx)
        except (TypeError, ValueError, TxDecodeError):
            return _sendrawtx_response(False, reason="malformed_tx")
        if not tx.inputs:
            return _sendrawtx_response(False, reason="coinbase_not_relayable", tx_id=tx.tx_id())
        result = self.mempool.add_tx(tx, utxo_view=self.utxo_set)
        return _sendrawtx_response(
            result.accepted,
            reason=result.reason,
            tx_id=result.tx_id,
            shard_id=result.shard_id,
            fee_matoms=result.fee_matoms,
            fee_rate_matoms_per_kb=result.fee_rate_matoms_per_kb,
        )

    def getbalance(self, owner_pubkey_hash: bytes, address: str) -> JsonObject:
        utxos = _utxos_for_owner(self.utxo_set, owner_pubkey_hash)
        return {
            "address": address,
            "balance_matoms": sum(utxo.amount_matoms for utxo in utxos),
            "utxo_count": len(utxos),
        }

    def getutxos(self, owner_pubkey_hash: bytes, address: str) -> JsonObject:
        utxos = _utxos_for_owner(self.utxo_set, owner_pubkey_hash)
        return {
            "address": address,
            "utxos": [_utxo_json(utxo) for utxo in utxos],
        }

    def getmempool(self, shard_id: int | None) -> JsonObject:
        entries = _mempool_entries(self.mempool, shard_id)
        return {
            "count": len(entries),
            "transactions": [_mempool_entry_json(entry) for entry in entries],
        }

    def getshardtree(self) -> JsonObject:
        return _shard_tree_json(self.shard_tree)

    def getfinality(self, block_hash: bytes) -> JsonObject:
        depths = self._finality.get(block_hash)
        if depths is not None:
            blue_depth, anchor_depth, seen = depths
            return _finality_json(
                block_hash,
                seen=seen,
                blue_depth=blue_depth,
                anchor_depth=anchor_depth,
            )
        return _finality_json(block_hash, seen=block_hash in self._blocks)


class NodeRpcAdapter:
    """Structural adapter for ``tensorpow.node.node.TensorPowNode`` and similar objects."""

    def __init__(self, node: object) -> None:
        self._node = node

    def getblock(self, block_hash: bytes) -> JsonObject:
        raw_block = self._node_block_bytes(block_hash)
        return _block_lookup_json(block_hash, raw_block)

    def gettx(self, tx_id: bytes) -> JsonObject:
        tx = _coerce_transaction(_call_optional(self._node, ("gettx", "get_tx"), tx_id))
        if tx is not None:
            return _tx_lookup_json(tx_id, tx, in_mempool=_mempool_contains(self._node, tx_id))
        return _tx_lookup_json(tx_id, None, in_mempool=False)

    def sendrawtx(self, raw_tx: bytes) -> JsonObject:
        processed = _call_optional(self._node, ("sendrawtx", "process_raw_tx"), raw_tx)
        if processed is not None:
            return _node_sendrawtx_response(processed)

        try:
            tx = Transaction.from_bytes(raw_tx)
        except (TypeError, ValueError, TxDecodeError):
            return _sendrawtx_response(False, reason="malformed_tx")
        if not tx.inputs:
            return _sendrawtx_response(False, reason="coinbase_not_relayable", tx_id=tx.tx_id())
        mempool = _node_mempool(self._node)
        result = mempool.add_tx(tx, utxo_view=_node_utxo_set(self._node))
        return _sendrawtx_response(
            result.accepted,
            reason=result.reason,
            tx_id=result.tx_id,
            shard_id=result.shard_id,
            fee_matoms=result.fee_matoms,
            fee_rate_matoms_per_kb=result.fee_rate_matoms_per_kb,
        )

    def getbalance(self, owner_pubkey_hash: bytes, address: str) -> JsonObject:
        utxos = _utxos_for_owner(_node_utxo_set(self._node), owner_pubkey_hash)
        return {
            "address": address,
            "balance_matoms": sum(utxo.amount_matoms for utxo in utxos),
            "utxo_count": len(utxos),
        }

    def getutxos(self, owner_pubkey_hash: bytes, address: str) -> JsonObject:
        utxos = _utxos_for_owner(_node_utxo_set(self._node), owner_pubkey_hash)
        return {
            "address": address,
            "utxos": [_utxo_json(utxo) for utxo in utxos],
        }

    def getmempool(self, shard_id: int | None) -> JsonObject:
        entries = _mempool_entries(_node_mempool(self._node), shard_id)
        return {
            "count": len(entries),
            "transactions": [_mempool_entry_json(entry) for entry in entries],
        }

    def getshardtree(self) -> JsonObject:
        return _shard_tree_json(_node_shard_tree(self._node))

    def getfinality(self, block_hash: bytes) -> JsonObject:
        explicit = _call_optional(self._node, ("getfinality", "get_finality"), block_hash)
        if explicit is not None:
            return _coerce_finality_result(block_hash, explicit)
        return _finality_json(block_hash, seen=self._node_block_bytes(block_hash) is not None)

    def _node_block_bytes(self, block_hash: bytes) -> bytes | None:
        block = _call_optional(self._node, ("getblock", "get_block"), block_hash)
        raw = _coerce_block_bytes(block)
        if raw is not None:
            return raw

        store = getattr(self._node, "store", None)
        if store is None:
            return None
        return _coerce_block_bytes(_call_optional(store, ("get_body_bytes",), block_hash))


class JsonRpcServer:
    """JSON-RPC method dispatcher with optional stdlib HTTP integration."""

    def __init__(
        self,
        backend: object | None = None,
        *,
        subscription_hub: SubscriptionHub | None = None,
    ) -> None:
        if backend is None:
            self.backend: RpcBackend = InMemoryRpcBackend()
        elif isinstance(backend, RpcBackend):
            self.backend = backend
        else:
            self.backend = NodeRpcAdapter(backend)
        self.subscriptions = SubscriptionHub() if subscription_hub is None else subscription_hub

    def handle_json(self, payload: bytes | str) -> bytes | None:
        """Handle a serialized JSON-RPC message and return response bytes."""

        try:
            text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        except UnicodeDecodeError:
            response = _error_response(None, PARSE_ERROR, "Parse error")
            return _json_response_bytes(response)

        try:
            message = cast(object, json.loads(text))
        except JSONDecodeError:
            response = _error_response(None, PARSE_ERROR, "Parse error")
            return _json_response_bytes(response)

        rpc_response = self.handle_message(message)
        if rpc_response is None:
            return None
        return _json_response_bytes(rpc_response)

    def handle_message(self, message: object) -> JsonValue | None:
        """Handle a decoded JSON-RPC message."""

        if isinstance(message, list):
            if not message:
                return _error_response(None, INVALID_REQUEST, "Invalid Request")
            responses: list[JsonValue] = []
            for item in message:
                response = self._handle_request(item)
                if response is not None:
                    responses.append(response)
            return responses if responses else None
        return self._handle_request(message)

    def publish(self, topic: str, payload: JsonValue) -> int:
        """Publish a subscription event to local subscribers."""

        return self.subscriptions.publish(topic, payload)

    def drain_subscription(
        self,
        subscription_id: str,
        *,
        limit: int = DEFAULT_EVENT_DRAIN_LIMIT,
    ) -> JsonObject:
        """Return queued events for a subscription."""

        return self.subscriptions.drain(subscription_id, limit=limit)

    def openrpc_document(self) -> JsonObject:
        """Return the OpenRPC document served by ``GET /openrpc.json``."""

        return openrpc_document()

    def _handle_request(self, request: object) -> JsonObject | None:
        request_id = _request_id_or_none(request)
        if not isinstance(request, dict):
            return _error_response(None, INVALID_REQUEST, "Invalid Request")

        if "id" in request and not _is_valid_request_id(request["id"]):
            return _error_response(None, INVALID_REQUEST, "Invalid Request")

        is_notification = "id" not in request
        if request.get("jsonrpc") != JSONRPC_VERSION or not isinstance(request.get("method"), str):
            return _error_response(request_id, INVALID_REQUEST, "Invalid Request")

        raw_params = request.get("params", _PARAMS_OMITTED)
        if raw_params is not _PARAMS_OMITTED and not isinstance(raw_params, dict | list):
            return _error_response(request_id, INVALID_REQUEST, "Invalid Request")

        method = cast(str, request["method"])
        try:
            result = self._dispatch(method, raw_params)
        except JsonRpcError as error:
            if is_notification:
                return None
            return _error_response(request_id, error.code, error.message, error.data)
        except Exception as error:
            if is_notification:
                return None
            return _error_response(request_id, INTERNAL_ERROR, "Internal error", str(error))

        if is_notification:
            return None
        return {"jsonrpc": JSONRPC_VERSION, "result": result, "id": request_id}

    def _dispatch(self, method: str, raw_params: object) -> JsonValue:
        if method == "getblock":
            params = _params(raw_params, required=("block_hash",))
            return self.backend.getblock(_parse_hash_param(params, "block_hash"))
        if method == "gettx":
            params = _params(raw_params, required=("txid",))
            return self.backend.gettx(_parse_hash_param(params, "txid"))
        if method == "sendrawtx":
            params = _params(raw_params, required=("rawtx",))
            return self.backend.sendrawtx(_parse_hex_param(params, "rawtx"))
        if method == "getbalance":
            address, owner_pubkey_hash = _address_param(raw_params)
            return self.backend.getbalance(owner_pubkey_hash, address)
        if method == "getutxos":
            address, owner_pubkey_hash = _address_param(raw_params)
            return self.backend.getutxos(owner_pubkey_hash, address)
        if method == "getmempool":
            params = _params(raw_params, optional=("shard_id",))
            shard_id = _optional_shard_id(params.get("shard_id"))
            return self.backend.getmempool(shard_id)
        if method == "getshardtree":
            _params(raw_params)
            return self.backend.getshardtree()
        if method == "getfinality":
            params = _params(raw_params, required=("block_hash",))
            return self.backend.getfinality(_parse_hash_param(params, "block_hash"))
        if method == "subscribe":
            params = _params(raw_params, required=("topic",))
            topic = _string_param(params, "topic")
            subscription = self.subscriptions.subscribe(topic)
            return {
                "subscription_id": subscription.subscription_id,
                "topic": subscription.topic,
                "transport": "http-poll",
                "events_path": f"/subscriptions/{subscription.subscription_id}/events",
            }
        raise JsonRpcError(METHOD_NOT_FOUND, "Method not found")


class RpcHttpServer(ThreadingHTTPServer):
    """Threading HTTP server carrying a ``JsonRpcServer`` instance."""

    def __init__(
        self,
        server_address: tuple[str, int],
        rpc_server: JsonRpcServer,
        *,
        max_body_bytes: int = MAX_HTTP_BODY_BYTES,
    ) -> None:
        super().__init__(server_address, _RpcHttpHandler)
        self.rpc_server = rpc_server
        self.max_body_bytes = _require_positive_int("max_body_bytes", max_body_bytes)


class _RpcHttpHandler(BaseHTTPRequestHandler):
    server_version = "TensorPoWRPC"
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        """Serve JSON-RPC requests on ``/`` and ``/rpc``."""

        if self.path not in ("/", "/rpc"):
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._send_json({"error": "content length required"}, status=HTTPStatus.LENGTH_REQUIRED)
            return
        try:
            length = int(content_length)
        except ValueError:
            self._send_json({"error": "invalid content length"}, status=HTTPStatus.BAD_REQUEST)
            return
        rpc_http_server = cast(RpcHttpServer, self.server)
        if length < 0 or length > rpc_http_server.max_body_bytes:
            self._send_json(
                {"error": "request body too large"},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        response = rpc_http_server.rpc_server.handle_json(self.rfile.read(length))
        if response is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send_bytes(response, content_type="application/json")

    def do_GET(self) -> None:
        """Serve OpenRPC and subscription event drain endpoints."""

        parsed = urlparse(self.path)
        rpc_http_server = cast(RpcHttpServer, self.server)
        if parsed.path == "/ws" or (
            parsed.path == "/rpc" and self.headers.get("Upgrade", "").lower() == "websocket"
        ):
            self._handle_websocket(rpc_http_server)
            return

        if parsed.path == "/openrpc.json":
            self._send_json(rpc_http_server.rpc_server.openrpc_document())
            return

        prefix = "/subscriptions/"
        suffix = "/events"
        if parsed.path.startswith(prefix) and parsed.path.endswith(suffix):
            subscription_id = parsed.path[len(prefix) : -len(suffix)]
            limit = _limit_from_query(parsed.query)
            try:
                payload = rpc_http_server.rpc_server.drain_subscription(
                    subscription_id,
                    limit=limit,
                )
            except JsonRpcError as error:
                self._send_json(
                    {"error": {"code": error.code, "message": error.message}},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_json(payload)
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_websocket(self, rpc_http_server: RpcHttpServer) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        version = self.headers.get("Sec-WebSocket-Version")
        upgrade = self.headers.get("Upgrade", "").lower()
        connection = self.headers.get("Connection", "").lower()
        if not key or version != "13" or upgrade != "websocket" or "upgrade" not in connection:
            self._send_json({"error": "invalid websocket upgrade"}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", _websocket_accept(key))
        self.end_headers()
        self.close_connection = True

        while True:
            try:
                frame = _read_websocket_frame(
                    self.connection,
                    max_payload_bytes=rpc_http_server.max_body_bytes,
                )
            except JsonRpcError:
                _write_websocket_frame(
                    self.connection,
                    WS_OPCODE_CLOSE,
                    _websocket_close_payload(WS_CLOSE_PROTOCOL_ERROR, "protocol error"),
                )
                return
            if frame is None:
                return
            if frame.opcode == WS_OPCODE_CLOSE:
                _write_websocket_frame(self.connection, WS_OPCODE_CLOSE, frame.payload[:125])
                return
            if frame.opcode == WS_OPCODE_PING:
                _write_websocket_frame(self.connection, WS_OPCODE_PONG, frame.payload)
                continue
            if frame.opcode == WS_OPCODE_PONG:
                continue
            if frame.opcode not in (WS_OPCODE_TEXT, WS_OPCODE_BINARY):
                _write_websocket_frame(
                    self.connection,
                    WS_OPCODE_CLOSE,
                    _websocket_close_payload(WS_CLOSE_PROTOCOL_ERROR, "unsupported opcode"),
                )
                return

            response = rpc_http_server.rpc_server.handle_json(frame.payload)
            if response is not None:
                _write_websocket_frame(self.connection, WS_OPCODE_TEXT, response)

    def log_message(self, format_: str, *args: object) -> None:
        """Silence stdlib request logging; callers can wrap the server if needed."""

    def _send_json(
        self,
        payload: JsonValue,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self._send_bytes(
            _json_response_bytes(payload),
            status=status,
            content_type="application/json",
        )

    def _send_bytes(
        self,
        payload: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def create_http_server(
    backend: object | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 28332,
    rpc_server: JsonRpcServer | None = None,
    max_body_bytes: int = MAX_HTTP_BODY_BYTES,
) -> RpcHttpServer:
    """Create a stdlib HTTP JSON-RPC server without starting its loop."""

    if rpc_server is None:
        rpc_server = JsonRpcServer(backend)
    return RpcHttpServer((host, port), rpc_server, max_body_bytes=max_body_bytes)


def openrpc_document() -> JsonObject:
    """Return the TensorPoW OpenRPC method document."""

    return {
        "openrpc": "1.4.0",
        "info": {
            "title": "TensorPoW JSON-RPC API",
            "version": "0.1.0",
            "description": "HTTP and WebSocket JSON-RPC interface for TensorPoW nodes.",
        },
        "servers": [
            {"name": "local-http", "url": "http://127.0.0.1:28332/rpc"},
            {"name": "local-websocket", "url": "ws://127.0.0.1:28332/ws"},
        ],
        "methods": [
            _openrpc_method(
                "getblock",
                "Return canonical block bytes by hash.",
                (("block_hash", _hex_schema(HASH_LEN_BYTES), True),),
                _object_schema(),
            ),
            _openrpc_method(
                "gettx",
                "Return a transaction by id when known locally.",
                (("txid", _hex_schema(HASH_LEN_BYTES), True),),
                _object_schema(),
            ),
            _openrpc_method(
                "sendrawtx",
                "Validate and relay a canonical transaction.",
                (("rawtx", _hex_schema(), True),),
                _object_schema(),
            ),
            _openrpc_method(
                "getbalance",
                "Return spendable balance for an address.",
                (("address", {"type": "string"}, True),),
                _object_schema(),
            ),
            _openrpc_method(
                "getutxos",
                "Return spendable UTXOs for an address.",
                (("address", {"type": "string"}, True),),
                _object_schema(),
            ),
            _openrpc_method(
                "getmempool",
                "Return local mempool entries, optionally filtered by shard.",
                (("shard_id", {"type": "integer", "minimum": 0}, False),),
                _object_schema(),
            ),
            _openrpc_method(
                "getshardtree",
                "Return the current shard tree commitment and leaves.",
                (),
                _object_schema(),
            ),
            _openrpc_method(
                "getfinality",
                "Return local finality status for a block hash.",
                (("block_hash", _hex_schema(HASH_LEN_BYTES), True),),
                _object_schema(),
            ),
            _openrpc_method(
                "subscribe",
                "Create a local topic subscription and return an event-drain endpoint.",
                (("topic", {"type": "string", "minLength": 1}, True),),
                _object_schema(),
            ),
        ],
    }


def _params(
    raw_params: object,
    *,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
) -> JsonParams:
    names = (*required, *optional)
    if raw_params is _PARAMS_OMITTED:
        params: JsonParams = {}
    elif isinstance(raw_params, dict):
        params = dict(cast(Mapping[str, object], raw_params))
    elif isinstance(raw_params, list):
        if len(raw_params) > len(names):
            raise JsonRpcError(INVALID_PARAMS, "Invalid params", "too many positional params")
        params = {name: raw_params[index] for index, name in enumerate(names[: len(raw_params)])}
    else:
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request")

    extra = sorted(set(params) - set(names))
    if extra:
        unexpected: list[JsonValue] = list(extra)
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", {"unexpected": unexpected})

    missing = [name for name in required if name not in params]
    if missing:
        missing_params: list[JsonValue] = list(missing)
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", {"missing": missing_params})
    return params


def _address_param(raw_params: object) -> tuple[str, bytes]:
    params = _params(raw_params, required=("address",))
    address = _string_param(params, "address")
    try:
        return address, address_to_pubkey_hash(address)
    except (AddressDecodeError, TypeError, ValueError) as error:
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", str(error)) from error


def _parse_hash_param(params: JsonParams, name: str) -> bytes:
    value = _string_param(params, name)
    decoded = _parse_hex(value, name)
    if len(decoded) != HASH_LEN_BYTES:
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", f"{name} must be 32-byte hex")
    return decoded


def _parse_hex_param(params: JsonParams, name: str) -> bytes:
    return _parse_hex(_string_param(params, name), name)


def _parse_hex(value: str, name: str) -> bytes:
    if value.strip() != value or value.lower() != value:
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", f"{name} must be canonical hex")
    if len(value) % 2 != 0:
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", f"{name} must be even-length hex")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", f"{name} must be hex") from error
    if decoded.hex() != value:
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", f"{name} must be canonical hex")
    return decoded


def _string_param(params: JsonParams, name: str) -> str:
    value = params[name]
    if not isinstance(value, str) or value == "":
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", f"{name} must be a non-empty string")
    return value


def _optional_shard_id(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", "shard_id must be an integer")
    try:
        return require_shard_id(value)
    except (TypeError, ValueError) as error:
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", str(error)) from error


def _request_id_or_none(request: object) -> JsonScalar:
    if not isinstance(request, dict) or "id" not in request:
        return None
    request_id = request["id"]
    return cast(JsonScalar, request_id) if _is_valid_request_id(request_id) else None


def _is_valid_request_id(value: object) -> bool:
    return (
        value is None
        or isinstance(value, str)
        or (isinstance(value, int) and not isinstance(value, bool))
    )


def _error_response(
    request_id: JsonScalar,
    code: int,
    message: str,
    data: JsonValue | None = None,
) -> JsonObject:
    error: JsonObject = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "error": error, "id": request_id}


def _json_response_bytes(payload: JsonValue) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _block_lookup_json(block_hash: bytes, raw_block: bytes | None) -> JsonObject:
    return {
        "block_hash": block_hash.hex(),
        "found": raw_block is not None,
        "raw": None if raw_block is None else raw_block.hex(),
    }


def _tx_lookup_json(
    tx_id: bytes,
    tx: Transaction | None,
    *,
    in_mempool: bool,
    entry: MempoolEntry | None = None,
) -> JsonObject:
    if tx is None:
        return {"txid": tx_id.hex(), "found": False}
    result = _tx_json(tx)
    result["found"] = True
    result["in_mempool"] = in_mempool
    if entry is not None:
        result.update(_mempool_metadata_json(entry))
    return result


def _sendrawtx_response(
    accepted: bool,
    *,
    reason: str | None,
    tx_id: bytes | None = None,
    shard_id: int | None = None,
    fee_matoms: int | None = None,
    fee_rate_matoms_per_kb: int | None = None,
) -> JsonObject:
    result: JsonObject = {"accepted": accepted}
    if tx_id is not None:
        result["txid"] = tx_id.hex()
    if shard_id is not None:
        result["shard_id"] = shard_id
    if fee_matoms is not None:
        result["fee_matoms"] = fee_matoms
    if fee_rate_matoms_per_kb is not None:
        result["fee_rate_matoms_per_kb"] = fee_rate_matoms_per_kb
    if reason is not None:
        result["reason"] = reason
    return result


def _node_sendrawtx_response(processed: object) -> JsonObject:
    if not isinstance(processed, tuple) or not processed:
        return _coerce_sendrawtx_object(processed)

    node_result = processed[0]
    mempool_result = processed[1] if len(processed) > 1 else None
    accepted = bool(getattr(node_result, "accepted", False))
    reason = _optional_str(getattr(node_result, "reason", None))
    tx_id = _optional_hash(getattr(node_result, "object_hash", None))

    if mempool_result is not None:
        tx_id = tx_id or _optional_hash(getattr(mempool_result, "tx_id", None))
        shard_id = _optional_int(getattr(mempool_result, "shard_id", None))
        fee_matoms = _optional_int(getattr(mempool_result, "fee_matoms", None))
        fee_rate = _optional_int(getattr(mempool_result, "fee_rate_matoms_per_kb", None))
        reason = reason or _optional_str(getattr(mempool_result, "reason", None))
    else:
        shard_id = None
        fee_matoms = None
        fee_rate = None

    return _sendrawtx_response(
        accepted,
        reason=reason,
        tx_id=tx_id,
        shard_id=shard_id,
        fee_matoms=fee_matoms,
        fee_rate_matoms_per_kb=fee_rate,
    )


def _coerce_sendrawtx_object(value: object) -> JsonObject:
    if isinstance(value, dict):
        normalized = _json_object(value)
        if "accepted" in normalized:
            return normalized
    raise JsonRpcError(INTERNAL_ERROR, "Internal error", "sendrawtx backend returned invalid data")


def _tx_json(tx: Transaction) -> JsonObject:
    inputs: list[JsonValue] = []
    for input_ in tx.inputs:
        inputs.append(
            {
                "previous_outpoint": {
                    "txid": input_.previous_outpoint.tx_id.hex(),
                    "output_index": input_.previous_outpoint.output_index,
                },
                "sequence": input_.sequence,
                "witness": input_.witness.hex(),
            }
        )

    outputs: list[JsonValue] = []
    for index, output in enumerate(tx.outputs):
        outputs.append(
            {
                "index": index,
                "amount_matoms": output.amount_matoms,
                "template_id": output.template_id,
                "locktime_ms": output.locktime_ms,
                "lockheight": output.lockheight,
                "payload": output.payload.hex(),
            }
        )

    return {
        "txid": tx.tx_id().hex(),
        "raw": tx.to_bytes().hex(),
        "version": tx.version,
        "sig_type": tx.sig_type,
        "locktime_ms": tx.locktime_ms,
        "lockheight": tx.lockheight,
        "inputs": inputs,
        "outputs": outputs,
    }


def _mempool_entry_json(entry: MempoolEntry) -> JsonObject:
    tx_json = _tx_json(entry.tx)
    tx_json.update(_mempool_metadata_json(entry))
    return tx_json


def _mempool_metadata_json(entry: MempoolEntry) -> JsonObject:
    return {
        "shard_id": entry.shard_id,
        "tx_size_bytes": entry.tx_size_bytes,
        "fee_matoms": entry.fee_matoms,
        "fee_rate_matoms_per_kb": entry.fee_rate_matoms_per_kb,
        "is_coinbase": entry.is_coinbase,
    }


def _utxo_json(utxo: UTXO) -> JsonObject:
    return {
        "outpoint": {
            "txid": utxo.outpoint.tx_id.hex(),
            "output_index": utxo.outpoint.output_index,
        },
        "amount_matoms": utxo.amount_matoms,
        "template_id": utxo.template_id,
        "owner_pubkey_hash": utxo.owner_pubkey_hash.hex(),
        "locktime_ms": utxo.locktime_ms,
        "lockheight": utxo.lockheight,
        "payload": utxo.payload.hex(),
    }


def _shard_tree_json(shard_tree: ShardTree) -> JsonObject:
    leaves: list[JsonValue] = []
    for shard_id in shard_tree.leaf_shard_ids:
        leaves.append(shard_id)
    return {
        "leaf_shard_ids": leaves,
        "state_root": shard_tree.state_root().hex(),
        "serialized": shard_tree.serialize().hex(),
    }


def _finality_json(
    block_hash: bytes,
    *,
    seen: bool,
    blue_depth: int = 0,
    anchor_depth: int = 0,
) -> JsonObject:
    tier = finality_tier_from_depths(blue_depth, anchor_depth, seen=seen)
    satisfied = cast(
        list[JsonValue],
        sorted(
            finality.value
            for finality in satisfied_finality_tiers(blue_depth, anchor_depth, seen=seen)
        ),
    )
    return {
        "block_hash": block_hash.hex(),
        "seen": seen,
        "tier": tier.value,
        "satisfied_tiers": satisfied,
        "blue_depth": blue_depth,
        "anchor_depth": anchor_depth,
    }


def _coerce_finality_result(block_hash: bytes, value: object) -> JsonObject:
    if isinstance(value, FinalityTier):
        return _finality_json(block_hash, seen=value is not FinalityTier.NONE)
    if isinstance(value, dict):
        normalized = _json_object(value)
        if "block_hash" not in normalized:
            normalized["block_hash"] = block_hash.hex()
        return normalized
    if isinstance(value, tuple) and len(value) in (2, 3):
        blue_depth = _require_nonnegative_int("blue_depth", value[0])
        anchor_depth = _require_nonnegative_int("anchor_depth", value[1])
        seen = True if len(value) == 2 else value[2]
        if not isinstance(seen, bool):
            raise JsonRpcError(INTERNAL_ERROR, "Internal error", "finality seen flag must be bool")
        return _finality_json(
            block_hash,
            seen=seen,
            blue_depth=blue_depth,
            anchor_depth=anchor_depth,
        )
    raise JsonRpcError(INTERNAL_ERROR, "Internal error", "finality backend returned invalid data")


def _utxos_for_owner(utxo_set: UTXOSet, owner_pubkey_hash: bytes) -> tuple[UTXO, ...]:
    _require_hash_bytes("owner_pubkey_hash", owner_pubkey_hash)
    return tuple(utxo for utxo in utxo_set.utxos() if utxo.owner_pubkey_hash == owner_pubkey_hash)


def _mempool_entries(mempool: Mempool, shard_id: int | None) -> tuple[MempoolEntry, ...]:
    entries = mempool.entries()
    if shard_id is None:
        return entries
    return tuple(entry for entry in entries if entry.shard_id == shard_id)


def _mempool_contains(node: object, tx_id: bytes) -> bool:
    mempool = getattr(node, "mempool", None)
    if not isinstance(mempool, Mempool):
        return False
    return mempool.contains(tx_id)


def _node_mempool(node: object) -> Mempool:
    mempool = getattr(node, "mempool", None)
    if not isinstance(mempool, Mempool):
        raise JsonRpcError(INTERNAL_ERROR, "Internal error", "backend has no mempool")
    return mempool


def _node_utxo_set(node: object) -> UTXOSet:
    utxo_set = getattr(node, "utxo_set", None)
    if not isinstance(utxo_set, UTXOSet):
        raise JsonRpcError(INTERNAL_ERROR, "Internal error", "backend has no utxo_set")
    return utxo_set


def _node_shard_tree(node: object) -> ShardTree:
    shard_tree = getattr(node, "shard_tree", None)
    if not isinstance(shard_tree, ShardTree):
        raise JsonRpcError(INTERNAL_ERROR, "Internal error", "backend has no shard_tree")
    return shard_tree


def _call_optional(target: object, names: tuple[str, ...], *args: object) -> object | None:
    for name in names:
        candidate = getattr(target, name, None)
        if callable(candidate):
            return cast(Callable[..., object], candidate)(*args)
    return None


def _coerce_block_bytes(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, Fruit | Anchor):
        return value.serialize()
    raise JsonRpcError(INTERNAL_ERROR, "Internal error", "block backend returned invalid data")


def _coerce_transaction(value: object) -> Transaction | None:
    if value is None:
        return None
    if isinstance(value, Transaction):
        return value
    if isinstance(value, bytes):
        try:
            return Transaction.from_bytes(value)
        except (TypeError, ValueError, TxDecodeError) as error:
            raise JsonRpcError(
                INTERNAL_ERROR,
                "Internal error",
                "stored transaction is malformed",
            ) from error
    raise JsonRpcError(
        INTERNAL_ERROR,
        "Internal error",
        "transaction backend returned invalid data",
    )


def _json_object(value: Mapping[object, object]) -> JsonObject:
    normalized: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise JsonRpcError(INTERNAL_ERROR, "Internal error", "JSON object key must be string")
        normalized[key] = _json_value(item)
    return normalized


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise JsonRpcError(INTERNAL_ERROR, "Internal error", "JSON number must be finite")
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return _json_object(value)
    raise JsonRpcError(INTERNAL_ERROR, "Internal error", "value is not JSON serializable")


def _openrpc_method(
    name: str,
    description: str,
    params: tuple[tuple[str, JsonObject, bool], ...],
    result_schema: JsonObject,
) -> JsonObject:
    method_params: list[JsonValue] = []
    for param_name, schema, required in params:
        method_params.append({"name": param_name, "required": required, "schema": schema})
    return {
        "name": name,
        "description": description,
        "params": method_params,
        "result": {"name": "result", "schema": result_schema},
    }


def _hex_schema(byte_len: int | None = None) -> JsonObject:
    schema: JsonObject = {"type": "string", "pattern": "^[0-9a-fA-F]*$"}
    if byte_len is not None:
        schema["minLength"] = byte_len * 2
        schema["maxLength"] = byte_len * 2
    return schema


def _object_schema() -> JsonObject:
    return {"type": "object"}


def _limit_from_query(query: str) -> int:
    values = parse_qs(query).get("limit")
    if not values:
        return DEFAULT_EVENT_DRAIN_LIMIT
    try:
        return _require_event_limit(int(values[0]))
    except ValueError:
        return DEFAULT_EVENT_DRAIN_LIMIT


def _websocket_accept(key: str) -> str:
    digest = sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
    return b64encode(digest).decode("ascii")


def _read_websocket_frame(sock: socket.socket, *, max_payload_bytes: int) -> _WebSocketFrame | None:
    header = _read_exact(sock, 2)
    if header is None:
        return None
    first, second = header
    fin = bool(first & 0x80)
    rsv = first & 0x70
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if rsv or not fin or not masked or opcode == WS_OPCODE_CONTINUATION:
        raise JsonRpcError(INVALID_REQUEST, "Invalid WebSocket frame")
    if opcode in (WS_OPCODE_CLOSE, WS_OPCODE_PING, WS_OPCODE_PONG) and length > 125:
        raise JsonRpcError(INVALID_REQUEST, "Invalid WebSocket control frame")
    if length == 126:
        extended = _read_exact(sock, 2)
        if extended is None:
            return None
        length = int.from_bytes(extended, "big")
        if length < 126:
            raise JsonRpcError(INVALID_REQUEST, "Invalid WebSocket frame length")
    elif length == 127:
        extended = _read_exact(sock, 8)
        if extended is None:
            return None
        length = int.from_bytes(extended, "big")
        if length < 65536 or length >= 2**63:
            raise JsonRpcError(INVALID_REQUEST, "Invalid WebSocket frame length")
    if length > max_payload_bytes:
        raise JsonRpcError(INVALID_REQUEST, "WebSocket frame too large")
    mask = _read_exact(sock, 4)
    payload = _read_exact(sock, length)
    if mask is None or payload is None:
        return None
    unmasked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return _WebSocketFrame(opcode=opcode, payload=unmasked)


def _write_websocket_frame(sock: socket.socket, opcode: int, payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    first = 0x80 | opcode
    length = len(payload)
    if length <= 125:
        header = bytes((first, length))
    elif length <= 0xFFFF:
        header = bytes((first, 126)) + length.to_bytes(2, "big")
    else:
        header = bytes((first, 127)) + length.to_bytes(8, "big")
    sock.sendall(header + payload)


def _websocket_close_payload(code: int, reason: str = "") -> bytes:
    return code.to_bytes(2, "big") + reason.encode("utf-8")[:123]


def _read_exact(sock: socket.socket, length: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _require_topic(topic: str) -> str:
    return _require_nonempty_string("topic", topic)


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", f"{name} must be a non-empty string")
    return value


def _require_bytes(name: str, value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    return value


def _require_hash_bytes(name: str, value: object) -> bytes:
    value = _require_bytes(name, value)
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_positive_int(name: str, value: object) -> int:
    value = _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_event_limit(value: int) -> int:
    value = _require_positive_int("limit", value)
    return min(value, MAX_EVENT_DRAIN_LIMIT)


def _optional_hash(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes) and len(value) == HASH_LEN_BYTES:
        return value
    return None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "DEFAULT_EVENT_DRAIN_LIMIT",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "JSONRPC_VERSION",
    "MAX_HTTP_BODY_BYTES",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "InMemoryRpcBackend",
    "JsonObject",
    "JsonRpcError",
    "JsonRpcServer",
    "JsonValue",
    "NodeRpcAdapter",
    "RpcBackend",
    "RpcHttpServer",
    "Subscription",
    "SubscriptionHub",
    "create_http_server",
    "openrpc_document",
]
