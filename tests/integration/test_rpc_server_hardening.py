"""M-6: RPC rejects cross-origin / rebinding / unauthenticated clients and bounds lockouts."""

from __future__ import annotations

import http.client
import json
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Thread

import pytest

from tensorpow.rpc.server import (
    InMemoryRpcBackend,
    JsonRpcError,
    JsonRpcServer,
    RpcHttpServer,
    SubscriptionHub,
    create_http_server,
)
from tests.integration.test_rpc_server import _read_ws_frame, _recv_until, _send_ws_frame

TOKEN = "t" * 32
BODY = json.dumps({"jsonrpc": "2.0", "method": "getmempool", "id": 1}).encode("utf-8")


@contextmanager
def _server(**kwargs: object) -> Iterator[tuple[RpcHttpServer, int]]:
    httpd = create_http_server(
        rpc_server=JsonRpcServer(InMemoryRpcBackend()),
        port=0,
        **kwargs,  # type: ignore[arg-type]
    )
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, int(httpd.server_address[1])
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post_status(port: int, headers: dict[str, str]) -> int:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("POST", "/rpc", body=BODY, headers=headers)
    response = connection.getresponse()
    response.read()
    connection.close()
    return response.status


def _ws_status(port: int, *, extra_headers: str = "", path: str = "/ws") -> int:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(
            (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                f"{extra_headers}"
                "\r\n"
            ).encode("ascii")
        )
        response = _recv_until(sock, b"\r\n\r\n")
    return int(response.split(maxsplit=2)[1])


def test_browser_origin_is_refused_for_http_and_websocket() -> None:
    with _server() as (_httpd, port):
        json_headers = {"Content-Type": "application/json"}
        assert _post_status(port, json_headers) == 200
        assert _post_status(port, {**json_headers, "Origin": "https://evil.example"}) == 403
        assert _ws_status(port) == 101
        assert _ws_status(port, extra_headers="Origin: https://evil.example\r\n") == 403

    with _server(allowed_origins=("https://wallet.example",)) as (_httpd, port):
        allowed = {"Content-Type": "application/json", "Origin": "https://wallet.example"}
        assert _post_status(port, allowed) == 200


def test_post_requires_json_content_type() -> None:
    with _server() as (_httpd, port):
        assert _post_status(port, {"Content-Type": "text/plain"}) == 415
        assert _post_status(port, {}) == 415
        assert _post_status(port, {"Content-Type": "application/json; charset=utf-8"}) == 200


def test_dns_rebinding_host_is_refused() -> None:
    with _server() as (_httpd, port):
        rebinding = {"Content-Type": "application/json", "Host": f"evil.example:{port}"}
        assert _post_status(port, rebinding) == 403
        assert _post_status(port, {"Content-Type": "application/json", "Host": "localhost"}) == 200
    with _server(allowed_hosts=("node.internal",)) as (_httpd, port):
        named = {"Content-Type": "application/json", "Host": f"node.internal:{port}"}
        assert _post_status(port, named) == 200


def test_optional_bearer_token_is_enforced() -> None:
    with _server(auth_token=TOKEN) as (_httpd, port):
        base = {"Content-Type": "application/json"}
        assert _post_status(port, base) == 401
        assert _post_status(port, {**base, "Authorization": "Bearer wrong-token-value!!"}) == 401
        assert _post_status(port, {**base, "Authorization": f"Bearer {TOKEN}"}) == 200
        assert _ws_status(port) == 401
        assert _ws_status(port, path=f"/ws?token={TOKEN}") == 101
    with pytest.raises(ValueError, match="auth_token"):
        create_http_server(port=0, auth_token="short")


def test_one_client_cannot_take_every_connection_slot() -> None:
    httpd = create_http_server(
        rpc_server=JsonRpcServer(InMemoryRpcBackend()),
        port=0,
        max_connections_per_client=2,
    )
    try:
        assert httpd._acquire_client_slot("10.0.0.1")
        assert httpd._acquire_client_slot("10.0.0.1")
        left, right = socket.socketpair()
        httpd.process_request(left, ("10.0.0.1", 1234))
        assert left.fileno() == -1
        right.close()
        assert httpd._acquire_client_slot("10.0.0.2")
    finally:
        httpd.server_close()


def test_websocket_lifetime_is_bounded_even_with_keepalive_pings() -> None:
    with (
        _server(websocket_max_lifetime_seconds=0.3) as (_httpd, port),
        socket.create_connection(("127.0.0.1", port), timeout=5) as sock,
    ):
        sock.sendall(
            (
                "GET /ws HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode("ascii")
        )
        assert _recv_until(sock, b"\r\n\r\n").startswith(b"HTTP/1.1 101")
        closed = False
        for _ in range(20):
            _send_ws_frame(sock, b"keepalive", opcode=0x9)
            opcode, payload = _read_ws_frame(sock)
            if opcode == 0x8:
                assert int.from_bytes(payload[:2], "big") == 1001
                closed = True
                break
            time.sleep(0.05)
        assert closed


def test_websocket_slots_are_capped() -> None:
    with _server(max_websocket_connections=1) as (httpd, port):
        assert httpd._websocket_slots.acquire(blocking=False)
        try:
            assert _ws_status(port) == 503
        finally:
            httpd._websocket_slots.release()


def test_subscriptions_are_capped_per_client() -> None:
    hub = SubscriptionHub(max_subscriptions_per_owner=2)
    hub.subscribe("t", owner="a")
    hub.subscribe("t", owner="a")
    with pytest.raises(JsonRpcError, match="Per-client"):
        hub.subscribe("t", owner="a")
    hub.subscribe("t", owner="b")
