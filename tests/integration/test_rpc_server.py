"""HTTP integration tests for the TensorPoW JSON-RPC server."""

from __future__ import annotations

import http.client
import json
import os
import socket
from threading import Thread

from tensorpow.crypto.address import pubkey_to_address
from tensorpow.rpc.server import (
    PARSE_ERROR,
    InMemoryRpcBackend,
    JsonRpcServer,
    RpcHttpServer,
    create_http_server,
)


def test_http_json_rpc_covers_all_methods_and_malformed_requests() -> None:
    backend = InMemoryRpcBackend()
    rpc_server = JsonRpcServer(backend)
    httpd = create_http_server(rpc_server=rpc_server, port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = int(httpd.server_address[1])

    try:
        missing_hash = "00" * 32
        address = pubkey_to_address(bytes(range(32)))

        assert _rpc(port, "getblock", {"block_hash": missing_hash})["found"] is False
        assert _rpc(port, "gettx", {"txid": missing_hash})["found"] is False
        assert _rpc(port, "sendrawtx", {"rawtx": "00"})["reason"] == "malformed_tx"
        assert _rpc(port, "getbalance", {"address": address})["balance_matoms"] == 0
        assert _rpc(port, "getutxos", {"address": address})["utxos"] == []
        assert _rpc(port, "getmempool")["count"] == 0
        assert _rpc(port, "getshardtree")["leaf_shard_ids"] == [0]
        assert _rpc(port, "getfinality", {"block_hash": missing_hash})["tier"] == "None"

        subscription = _rpc(port, "subscribe", {"topic": "tensorpow/anchors/main"})
        assert rpc_server.publish("tensorpow/anchors/main", {"block_hash": missing_hash}) == 1
        events = _get(port, str(subscription["events_path"]))
        assert events["events"] == [
            {"topic": "tensorpow/anchors/main", "payload": {"block_hash": missing_hash}}
        ]

        openrpc = _get(port, "/openrpc.json")
        method_names = {method["name"] for method in openrpc["methods"]}
        assert {
            "getblock",
            "gettx",
            "sendrawtx",
            "getbalance",
            "getutxos",
            "getmempool",
            "getshardtree",
            "getfinality",
            "subscribe",
        } <= method_names

        status, malformed = _post(port, "{")
        assert status == 200
        assert malformed["error"]["code"] == PARSE_ERROR

        websocket_tree = _websocket_rpc(
            port,
            {"jsonrpc": "2.0", "method": "getshardtree", "id": "ws-tree"},
        )
        assert websocket_tree["id"] == "ws-tree"
        assert websocket_tree["result"]["leaf_shard_ids"] == [0]

        close_opcode, close_payload = _websocket_unmasked_protocol_error(port)
        assert close_opcode == 0x8
        assert int.from_bytes(close_payload[:2], "big") == 1002

        nonminimal_opcode, nonminimal_payload = _websocket_nonminimal_length_error(port)
        assert nonminimal_opcode == 0x8
        assert int.from_bytes(nonminimal_payload[:2], "big") == 1002
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_http_request_body_read_times_out() -> None:
    httpd, thread, port = _start_server(request_timeout_seconds=0.1)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.settimeout(5)
            sock.sendall(
                b"POST /rpc HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 20\r\n"
                b"\r\n"
                b"{"
            )
            response = _recv_until(sock, b"\r\n\r\n")

        assert response.startswith(b"HTTP/1.1 408")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_websocket_idle_read_deadline_closes_connection() -> None:
    httpd, thread, port = _start_server(request_timeout_seconds=0.1)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.settimeout(5)
            _websocket_handshake(sock, port)
            assert sock.recv(1) == b""
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_http_server_enforces_connection_cap() -> None:
    httpd = create_http_server(
        rpc_server=JsonRpcServer(InMemoryRpcBackend()),
        port=0,
        max_connections=1,
    )
    left, right = socket.socketpair()
    try:
        assert httpd._connection_slots.acquire(blocking=False)
        httpd.process_request(left, ("local", 0))
        assert left.fileno() == -1
    finally:
        httpd._connection_slots.release()
        right.close()
        httpd.server_close()


def _start_server(*, request_timeout_seconds: float) -> tuple[RpcHttpServer, Thread, int]:
    rpc_server = JsonRpcServer(InMemoryRpcBackend())
    httpd = create_http_server(
        rpc_server=rpc_server,
        port=0,
        request_timeout_seconds=request_timeout_seconds,
    )
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, int(httpd.server_address[1])


def _rpc(
    port: int,
    method: str,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    request: dict[str, object] = {"jsonrpc": "2.0", "method": method, "id": method}
    if params is not None:
        request["params"] = params
    status, response = _post(port, json.dumps(request))
    assert status == 200
    assert response["id"] == method
    return response["result"]  # type: ignore[return-value]


def _post(port: int, body: str) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(
        "POST",
        "/rpc",
        body=body.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, json.loads(payload)


def _get(port: int, path: str) -> dict[str, object]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    assert response.status == 200
    return json.loads(payload)  # type: ignore[no-any-return]


def _websocket_rpc(port: int, request: dict[str, object]) -> dict[str, object]:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        _websocket_handshake(sock, port)
        _send_ws_frame(sock, json.dumps(request).encode("utf-8"))
        opcode, payload = _read_ws_frame(sock)
        _send_ws_frame(sock, b"", opcode=0x8)
    assert opcode == 0x1
    response = json.loads(payload)
    assert isinstance(response, dict)
    return response  # type: ignore[no-any-return]


def _websocket_unmasked_protocol_error(port: int) -> tuple[int, bytes]:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        _websocket_handshake(sock, port)
        payload = json.dumps({"jsonrpc": "2.0", "method": "getshardtree", "id": 1}).encode("utf-8")
        sock.sendall(bytes((0x81, len(payload))) + payload)
        return _read_ws_frame(sock)


def _websocket_nonminimal_length_error(port: int) -> tuple[int, bytes]:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        _websocket_handshake(sock, port)
        payload = b"{}"
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        sock.sendall(bytes((0x81, 0x80 | 126)) + len(payload).to_bytes(2, "big") + mask + masked)
        return _read_ws_frame(sock)


def _websocket_handshake(sock: socket.socket, port: int) -> None:
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    request = (
        "GET /ws HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    response = _recv_until(sock, b"\r\n\r\n")
    assert response.startswith(b"HTTP/1.1 101")
    assert b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" in response


def _send_ws_frame(sock: socket.socket, payload: bytes, *, opcode: int = 0x1) -> None:
    mask = os.urandom(4)
    length = len(payload)
    if length <= 125:
        header = bytes((0x80 | opcode, 0x80 | length))
    elif length <= 0xFFFF:
        header = bytes((0x80 | opcode, 0x80 | 126)) + length.to_bytes(2, "big")
    else:
        header = bytes((0x80 | opcode, 0x80 | 127)) + length.to_bytes(8, "big")
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(header + mask + masked)


def _read_ws_frame(sock: socket.socket) -> tuple[int, bytes]:
    header = _recv_exact(sock, 2)
    first, second = header
    assert first & 0x80
    opcode = first & 0x0F
    assert not second & 0x80
    length = second & 0x7F
    if length == 126:
        length = int.from_bytes(_recv_exact(sock, 2), "big")
    elif length == 127:
        length = int.from_bytes(_recv_exact(sock, 8), "big")
    return opcode, _recv_exact(sock, length)


def _recv_until(sock: socket.socket, marker: bytes) -> bytes:
    chunks = bytearray()
    while marker not in chunks:
        chunk = sock.recv(1)
        assert chunk
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        assert chunk
        chunks.extend(chunk)
    return bytes(chunks)
