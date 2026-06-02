"""Unit tests for the TensorPoW JSON-RPC server."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from tensorpow.crypto.address import pubkey_to_address
from tensorpow.crypto.signatures import sign
from tensorpow.rpc.server import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    MAX_GETUTXOS_LIMIT,
    MAX_JSON_RPC_BATCH_SIZE,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    InMemoryRpcBackend,
    JsonRpcServer,
)
from tensorpow.state import TEMPLATE_PKH, UTXO, Outpoint, UTXOSet
from tensorpow.tx.script import pubkey_hash
from tensorpow.tx.transaction import FORMAT_EPOCH, Input, Output, Transaction

PUB1 = bytes.fromhex("343010a1aba8774dd1e6f4f0c3349bae6824908a1e64cd638dc2ed1bc625af1d")
PRIV1 = bytes.fromhex("cd4f7f79a2b8168f5cbeccb55d415492fd3504e52ed4fe7b02ea404fede9a40b")
ADDRESS1 = pubkey_to_address(PUB1)
PKH1 = pubkey_hash(PUB1)
ROOT_SHARD_ID = 0


def test_all_rpc_methods_return_project_shaped_results() -> None:
    utxo = _utxo(1, amount=2_000)
    backend = InMemoryRpcBackend(utxo_set=UTXOSet((utxo,)))
    server = JsonRpcServer(backend)
    block_hash = bytes([9]) * 32
    backend.put_block(block_hash, b"block-body")
    backend.set_finality(block_hash, blue_depth=20, anchor_depth=1)

    block = _rpc(server, "getblock", {"block_hash": block_hash.hex()})
    assert block["found"] is True
    assert block["raw"] == b"block-body".hex()

    tx = _signed_tx(utxo, fee=250)
    sent = _rpc(server, "sendrawtx", {"rawtx": tx.to_bytes().hex()})
    assert sent["accepted"] is True
    assert sent["txid"] == tx.tx_id().hex()
    assert sent["shard_id"] == ROOT_SHARD_ID

    tx_lookup = _rpc(server, "gettx", {"txid": tx.tx_id().hex()})
    assert tx_lookup["found"] is True
    assert tx_lookup["in_mempool"] is True
    assert tx_lookup["raw"] == tx.to_bytes().hex()

    balance = _rpc(server, "getbalance", {"address": ADDRESS1})
    assert balance == {"address": ADDRESS1, "balance_matoms": 2_000, "utxo_count": 1}

    utxos = _rpc(server, "getutxos", {"address": ADDRESS1})
    assert utxos["address"] == ADDRESS1
    assert utxos["utxos"] == [_utxo_json(utxo)]

    mempool = _rpc(server, "getmempool", {"shard_id": ROOT_SHARD_ID})
    assert mempool["count"] == 1
    assert mempool["transactions"][0]["txid"] == tx.tx_id().hex()

    shard_tree = _rpc(server, "getshardtree")
    assert shard_tree["leaf_shard_ids"] == [ROOT_SHARD_ID]
    assert isinstance(shard_tree["state_root"], str)

    finality = _rpc(server, "getfinality", {"block_hash": block_hash.hex()})
    assert finality["tier"] == "AnchorSecured"
    assert finality["blue_depth"] == 20
    assert finality["anchor_depth"] == 1

    subscription = _rpc(server, "subscribe", {"topic": "tensorpow/fruits/main"})
    assert subscription["topic"] == "tensorpow/fruits/main"
    assert subscription["transport"] == "http-poll"
    assert server.publish("tensorpow/fruits/main", {"block_hash": block_hash.hex()}) == 1
    drained = server.drain_subscription(str(subscription["subscription_id"]))
    assert drained["events"] == [
        {"topic": "tensorpow/fruits/main", "payload": {"block_hash": block_hash.hex()}}
    ]


def test_malformed_requests_return_clean_json_rpc_errors() -> None:
    server = JsonRpcServer(InMemoryRpcBackend())

    assert _error_code(_handle(server, "{")) == PARSE_ERROR
    assert _error_code(_handle(server, [])) == INVALID_REQUEST
    assert _error_code(_handle(server, {"jsonrpc": "2.0", "method": 1})) == INVALID_REQUEST
    assert (
        _error_code(_handle(server, {"jsonrpc": "2.0", "method": "missing", "id": 1}))
        == METHOD_NOT_FOUND
    )
    assert (
        _error_code(_handle(server, {"jsonrpc": "2.0", "method": "gettx", "id": 1}))
        == INVALID_PARAMS
    )
    assert (
        _error_code(
            _handle(
                server,
                {"jsonrpc": "2.0", "method": "gettx", "params": {"txid": "not-hex"}, "id": 1},
            )
        )
        == INVALID_PARAMS
    )
    assert (
        _error_code(
            _handle(
                server,
                {
                    "jsonrpc": "2.0",
                    "method": "gettx",
                    "params": {"txid": ("00" * 31) + " 0"},
                    "id": 1,
                },
            )
        )
        == INVALID_PARAMS
    )
    assert (
        _error_code(
            _handle(
                server,
                {
                    "jsonrpc": "2.0",
                    "method": "getshardtree",
                    "params": {"extra": True},
                    "id": 1,
                },
            )
        )
        == INVALID_PARAMS
    )
    assert (
        _error_code(
            _handle(server, {"jsonrpc": "2.0", "method": "getmempool", "params": "bad", "id": 1})
        )
        == INVALID_REQUEST
    )
    assert _handle(server, {"jsonrpc": "2.0", "method": "getshardtree"}) is None


def test_batches_omit_notifications_and_preserve_errors() -> None:
    server = JsonRpcServer(InMemoryRpcBackend())
    payload = [
        {"jsonrpc": "2.0", "method": "getshardtree", "id": "tree"},
        {"jsonrpc": "2.0", "method": "getshardtree"},
        7,
    ]

    response = _decode_response(_handle(server, payload))

    assert isinstance(response, list)
    assert len(response) == 2
    assert response[0]["id"] == "tree"
    assert response[1]["error"]["code"] == INVALID_REQUEST


def test_batch_size_is_bounded() -> None:
    server = JsonRpcServer(InMemoryRpcBackend())
    payload: list[dict[str, object]] = [
        {"jsonrpc": "2.0", "method": "getshardtree", "id": index}
        for index in range(MAX_JSON_RPC_BATCH_SIZE + 1)
    ]

    assert _error_code(_handle(server, payload)) == INVALID_REQUEST


def test_getutxos_uses_owner_index_and_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _utxo(1, amount=100)
    second = _utxo(3, amount=300)
    other = _utxo(2, amount=200, owner=bytes([4]) * 32)
    utxo_set = UTXOSet((second, other, first))
    backend = InMemoryRpcBackend(utxo_set=utxo_set)
    server = JsonRpcServer(backend)

    def fail_global_scan() -> tuple[UTXO, ...]:
        raise AssertionError("getbalance/getutxos must not scan all UTXOs")

    monkeypatch.setattr(utxo_set, "utxos", fail_global_scan)

    balance = _rpc(server, "getbalance", {"address": ADDRESS1})
    page = _rpc(server, "getutxos", {"address": ADDRESS1, "offset": 1, "limit": 1})

    assert balance == {"address": ADDRESS1, "balance_matoms": 400, "utxo_count": 2}
    assert page["address"] == ADDRESS1
    assert page["offset"] == 1
    assert page["limit"] == 1
    assert page["total"] == 2
    assert page["utxos"] == [_utxo_json(second)]


def test_getutxos_rejects_unbounded_limits() -> None:
    server = JsonRpcServer(InMemoryRpcBackend())

    assert (
        _error_code(
            _handle(
                server,
                {
                    "jsonrpc": "2.0",
                    "method": "getutxos",
                    "params": {"address": ADDRESS1, "limit": MAX_GETUTXOS_LIMIT + 1},
                    "id": 1,
                },
            )
        )
        == INVALID_PARAMS
    )


def test_structural_node_adapter_uses_node_methods() -> None:
    tx = Transaction.coinbase((Output(100, TEMPLATE_PKH, payload=PKH1),))
    node = _FakeNode(tx)
    server = JsonRpcServer(node)

    block = _rpc(server, "getblock", {"block_hash": node.block_hash.hex()})
    tx_lookup = _rpc(server, "gettx", {"txid": tx.tx_id().hex()})
    sent = _rpc(server, "sendrawtx", {"rawtx": tx.to_bytes().hex()})

    assert block["raw"] == node.block_bytes.hex()
    assert tx_lookup["raw"] == tx.to_bytes().hex()
    assert sent == {"accepted": True, "txid": tx.tx_id().hex()}


@dataclass(slots=True)
class _FakeNodeResult:
    accepted: bool
    reason: str | None = None
    object_hash: bytes | None = None


class _FakeNode:
    def __init__(self, tx: Transaction) -> None:
        self.tx = tx
        self.block_hash = bytes([7]) * 32
        self.block_bytes = b"node-block"

    def get_block(self, block_hash: bytes) -> bytes | None:
        return self.block_bytes if block_hash == self.block_hash else None

    def get_tx(self, tx_id: bytes) -> Transaction | None:
        return self.tx if tx_id == self.tx.tx_id() else None

    def process_raw_tx(self, data: bytes) -> tuple[_FakeNodeResult, None]:
        tx = Transaction.from_bytes(data)
        return _FakeNodeResult(True, object_hash=tx.tx_id()), None


def _rpc(
    server: JsonRpcServer,
    method: str,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    request: dict[str, object] = {"jsonrpc": "2.0", "method": method, "id": method}
    if params is not None:
        request["params"] = params
    response = _decode_response(_handle(server, request))
    assert isinstance(response, dict)
    assert response["id"] == method
    return response["result"]  # type: ignore[return-value]


def _decode_response(payload: bytes | None) -> object:
    assert payload is not None
    return json.loads(payload)


def _handle(server: JsonRpcServer, payload: object) -> bytes | None:
    if isinstance(payload, bytes | str):
        return server.handle_json(payload)
    return server.handle_json(json.dumps(payload))


def _error_code(payload: bytes | None) -> int:
    response = _decode_response(payload)
    assert isinstance(response, dict)
    error = response["error"]
    assert isinstance(error, dict)
    code = error["code"]
    assert isinstance(code, int)
    return code


def _outpoint(seed: int) -> Outpoint:
    return Outpoint(bytes([seed]) * 32, 0)


def _utxo(seed: int, *, amount: int, owner: bytes = PKH1) -> UTXO:
    return UTXO(
        outpoint=_outpoint(seed),
        amount_matoms=amount,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=owner,
    )


def _signed_tx(utxo: UTXO, *, fee: int) -> Transaction:
    output = Output(utxo.amount_matoms - fee, TEMPLATE_PKH, payload=PKH1)
    unsigned_input = Input(utxo.outpoint)
    unsigned_tx = Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=0,
        lockheight=0,
        inputs=(unsigned_input,),
        outputs=(output,),
    )
    signature = sign(unsigned_tx.sighash(0), PRIV1)
    return Transaction(
        version=unsigned_tx.version,
        sig_type=unsigned_tx.sig_type,
        locktime_ms=unsigned_tx.locktime_ms,
        lockheight=unsigned_tx.lockheight,
        inputs=(Input(utxo.outpoint, witness=signature + PUB1),),
        outputs=unsigned_tx.outputs,
    )


def _utxo_json(utxo: UTXO) -> dict[str, object]:
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
