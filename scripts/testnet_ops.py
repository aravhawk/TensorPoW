"""Public testnet genesis, faucet, explorer, and operator helpers."""

from __future__ import annotations

import argparse
import contextlib
import hmac
import http.client
import ipaddress
import json
import os
import re
import socket
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock, RLock
from typing import Any, Final, cast
from urllib.parse import parse_qs, urlparse

from blake3 import blake3

from tensorpow.crypto.address import address_to_pubkey_hash, validate_address
from tensorpow.genesis import (
    GENESIS_CHAIN_ID_TESTNET,
    GenesisInputs,
    artifact_from_json,
    build_genesis_artifact,
)
from tensorpow.launch_policy import PUBLIC_TESTNET_MIN_DAYS, PUBLIC_TESTNET_MIN_NODES
from tensorpow.node import TensorPowConfig
from tensorpow.tx import Transaction
from tensorpow.wallet import Wallet, load_utxos_json, load_wallet, utxos_to_json

DEFAULT_TESTNET_RPC_URL: Final[str] = "http://127.0.0.1:28332/rpc"
DEFAULT_FAUCET_PASSWORD_ENV: Final[str] = "TENSORPOW_TESTNET_FAUCET_PASSWORD"
MAX_TESTNET_HTTP_BODY_BYTES: Final[int] = 1_048_576
TESTNET_HTTP_TIMEOUT_SECONDS: Final[float] = 10.0
TESTNET_HTTP_MAX_CONNECTIONS: Final[int] = 32
FAUCET_ADDRESS_COOLDOWN_SECONDS: Final[float] = 24 * 60 * 60
FAUCET_IP_COOLDOWN_SECONDS: Final[float] = 60 * 60
FAUCET_MAX_DISPENSES_PER_HOUR: Final[int] = 120
FAUCET_MAX_TRACKED_KEYS: Final[int] = 100_000
POST_RATE_PER_SECOND: Final[float] = 1.0
POST_BURST: Final[int] = 10
DEFAULT_FAUCET_POW_BITS: Final[int] = 18
FAUCET_CHALLENGE_TTL_SECONDS: Final[float] = 300.0
FAUCET_CHALLENGE_DOMAIN: Final[bytes] = b"tensorpow-testnet-faucet-v1"
DNS_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class TestnetOpsError(ValueError):
    """Raised when public-testnet helper inputs are invalid."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public testnet helper CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return cast(int, args.handler(args))
    except (OSError, TypeError, ValueError, TestnetOpsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="testnet_ops", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    genesis = subparsers.add_parser("genesis", allow_abbrev=False)
    genesis.add_argument("--whitepaper-hash", required=True)
    genesis.add_argument("--bitcoin-block-hash", required=True)
    genesis.add_argument("--ethereum-block-hash", required=True)
    genesis.add_argument("--founder-pubkey-hash", required=True)
    genesis.add_argument("--out", type=Path, required=True)
    genesis.set_defaults(handler=_cmd_genesis)

    bootstrap = subparsers.add_parser("bootstrap-config", allow_abbrev=False)
    bootstrap.add_argument("--data-dir", type=Path, required=True)
    bootstrap.add_argument("--genesis", type=Path, required=True)
    bootstrap.add_argument("--listen-host", default="0.0.0.0")
    bootstrap.add_argument("--public-host")
    bootstrap.add_argument("--p2p-port", type=int, default=28333)
    bootstrap.add_argument("--rpc-host", default="127.0.0.1")
    bootstrap.add_argument("--rpc-port", type=int, default=28332)
    bootstrap.add_argument("--peer-key-path", type=Path)
    bootstrap.add_argument("--out", type=Path, required=True)
    bootstrap.set_defaults(handler=_cmd_bootstrap_config)

    faucet = subparsers.add_parser("faucet-tx", allow_abbrev=False)
    faucet.add_argument("--wallet", type=Path, required=True)
    faucet.add_argument(
        "--password-env",
        default=DEFAULT_FAUCET_PASSWORD_ENV,
        help="environment variable holding the faucet keystore password",
    )
    faucet.add_argument("--utxos", type=Path, required=True)
    faucet.add_argument("--to", required=True)
    faucet.add_argument("--amount", type=int, required=True)
    faucet.add_argument("--fee", type=int, default=0)
    faucet.set_defaults(handler=_cmd_faucet_tx)

    serve = subparsers.add_parser("serve", allow_abbrev=False)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=28380)
    serve.add_argument("--rpc-url", default=DEFAULT_TESTNET_RPC_URL)
    serve.add_argument("--faucet-wallet", type=Path)
    serve.add_argument("--faucet-password-env", default=DEFAULT_FAUCET_PASSWORD_ENV)
    serve.add_argument("--faucet-utxos", type=Path)
    serve.add_argument("--faucet-amount", type=int)
    serve.add_argument("--faucet-fee", type=int, default=0)
    serve.add_argument(
        "--faucet-pow-bits",
        type=int,
        default=DEFAULT_FAUCET_POW_BITS,
        help="leading zero bits required in the faucet client challenge (0 disables)",
    )
    serve.add_argument(
        "--faucet-address-cooldown",
        type=float,
        default=FAUCET_ADDRESS_COOLDOWN_SECONDS,
        help="seconds before the same address can be funded again",
    )
    serve.add_argument(
        "--faucet-ip-cooldown",
        type=float,
        default=FAUCET_IP_COOLDOWN_SECONDS,
        help="seconds before the same client IP can be funded again",
    )
    serve.add_argument(
        "--faucet-max-per-hour",
        type=int,
        default=FAUCET_MAX_DISPENSES_PER_HOUR,
        help="global cap on faucet payouts per rolling hour",
    )
    serve.set_defaults(handler=_cmd_serve)
    return parser


def _cmd_genesis(args: argparse.Namespace) -> int:
    inputs = GenesisInputs.create(
        chain_id=GENESIS_CHAIN_ID_TESTNET,
        whitepaper_hash=_hash_arg("whitepaper_hash", args.whitepaper_hash),
        bitcoin_block_hash=_hash_arg("bitcoin_block_hash", args.bitcoin_block_hash),
        ethereum_block_hash=_hash_arg("ethereum_block_hash", args.ethereum_block_hash),
        founder_pubkey_hash=_hash_arg("founder_pubkey_hash", args.founder_pubkey_hash),
    )
    artifact = build_genesis_artifact(inputs)
    payload = artifact.to_json()
    payload["public_testnet_gate"] = {
        "minimum_days": PUBLIC_TESTNET_MIN_DAYS,
        "minimum_unique_nodes": PUBLIC_TESTNET_MIN_NODES,
        "status": "not_satisfied_until_observed",
    }
    _write_json(Path(args.out), payload)
    return 0


def _cmd_bootstrap_config(args: argparse.Namespace) -> int:
    genesis_artifact = artifact_from_json(_read_json_object(Path(args.genesis)))
    if genesis_artifact.inputs.chain_id != GENESIS_CHAIN_ID_TESTNET:
        raise TestnetOpsError("bootstrap genesis must use tensorpow-testnet chain_id")
    public_host = _public_host(args.public_host, args.listen_host)
    config = TensorPowConfig(
        data_dir=args.data_dir,
        enable_network=True,
        listen_host=args.listen_host,
        p2p_tcp_port=args.p2p_port,
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        peer_key_path=args.peer_key_path,
        chain_id=GENESIS_CHAIN_ID_TESTNET,
        expected_genesis_hash=genesis_artifact.block_hash,
    )
    payload = {
        "chain_id": GENESIS_CHAIN_ID_TESTNET,
        "config_toml": config.to_toml(),
        "operator_requirements": {
            "minimum_days": PUBLIC_TESTNET_MIN_DAYS,
            "minimum_unique_nodes": PUBLIC_TESTNET_MIN_NODES,
        },
        "public_endpoints": {
            "p2p": _public_multiaddr(public_host, args.p2p_port),
            "rpc": f"http://{args.rpc_host}:{args.rpc_port}/rpc",
        },
        "testnet_genesis_hash": genesis_artifact.block_hash.hex(),
    }
    _write_json(Path(args.out), payload)
    return 0


def _cmd_faucet_tx(args: argparse.Namespace) -> int:
    password = os.environ.get(args.password_env)
    if not password:
        raise TestnetOpsError(f"{args.password_env} must be set to unlock the faucet wallet")
    wallet = load_wallet(args.wallet, password)
    tx = build_faucet_transaction(
        wallet=wallet,
        utxos_path=args.utxos,
        recipient_address=args.to,
        amount_matoms=args.amount,
        fee_matoms=args.fee,
    )
    print(json.dumps({"rawtx": tx.to_bytes().hex(), "txid": tx.tx_id().hex()}, sort_keys=True))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    server = create_testnet_http_server(
        host=args.host,
        port=args.port,
        rpc_url=args.rpc_url,
        faucet_state=_faucet_state_from_args(args),
        limiter=FaucetLimiter(
            address_cooldown_seconds=args.faucet_address_cooldown,
            ip_cooldown_seconds=args.faucet_ip_cooldown,
            max_dispenses_per_hour=args.faucet_max_per_hour,
        ),
        challenge=FaucetChallenge(args.faucet_pow_bits) if args.faucet_pow_bits > 0 else None,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def build_faucet_transaction(
    *,
    wallet: Wallet,
    utxos_path: Path,
    recipient_address: str,
    amount_matoms: int,
    fee_matoms: int,
) -> Transaction:
    """Build a faucet spend that can be submitted through ``sendrawtx``."""

    if not validate_address(recipient_address):
        raise TestnetOpsError("recipient address is invalid")
    return wallet.create_signed_transaction(
        utxos=load_utxos_json(utxos_path),
        recipient_address=recipient_address,
        amount_matoms=amount_matoms,
        fee_matoms=fee_matoms,
    )


class FaucetState:
    """Operator-configured faucet dispenser backed by a funded wallet and UTXO file."""

    def __init__(
        self,
        *,
        wallet: Wallet,
        utxos_path: Path,
        amount_matoms: int,
        fee_matoms: int,
    ) -> None:
        if not isinstance(wallet, Wallet):
            raise TypeError("wallet must be Wallet")
        self.wallet = wallet
        self.utxos_path = Path(utxos_path)
        self.amount_matoms = _positive_int("faucet_amount", amount_matoms)
        self.fee_matoms = _nonnegative_int("faucet_fee", fee_matoms)
        self._lock = RLock()

    def dispense(self, *, recipient_address: str, rpc_url: str) -> dict[str, object]:
        """Build, relay, and account for one faucet transaction."""

        with self._lock:
            tx = build_faucet_transaction(
                wallet=self.wallet,
                utxos_path=self.utxos_path,
                recipient_address=recipient_address,
                amount_matoms=self.amount_matoms,
                fee_matoms=self.fee_matoms,
            )
            tx_id = tx.tx_id().hex()
            rawtx = tx.to_bytes().hex()
            result = rpc_call(rpc_url, "sendrawtx", {"rawtx": rawtx})
            if result.get("txid") not in (None, tx_id):
                raise TestnetOpsError("RPC returned a txid that does not match faucet spend")
            accepted = result.get("accepted") is True
            if accepted:
                self._remove_spent_utxos(tx)
            return {
                "accepted": accepted,
                "address": recipient_address,
                "amount_matoms": self.amount_matoms,
                "fee_matoms": self.fee_matoms,
                "rawtx": rawtx,
                "sendrawtx": result,
                "txid": tx_id,
            }

    def _remove_spent_utxos(self, tx: Transaction) -> None:
        spent = {input_.previous_outpoint.to_bytes() for input_ in tx.inputs}
        current = load_utxos_json(self.utxos_path)
        remaining = tuple(utxo for utxo in current if utxo.outpoint.to_bytes() not in spent)
        if len(remaining) == len(current):
            raise TestnetOpsError("faucet spend did not consume any tracked UTXO")
        tmp_path = self.utxos_path.with_name(f"{self.utxos_path.name}.tmp")
        tmp_path.write_text(utxos_to_json(remaining) + "\n", encoding="utf-8")
        os.replace(tmp_path, self.utxos_path)


class FaucetLimiter:
    """Per-address and per-IP faucet cooldowns, a global hourly cap and POST throttling.

    All tables are bounded; the oldest entries are evicted first.
    """

    def __init__(
        self,
        *,
        address_cooldown_seconds: float = FAUCET_ADDRESS_COOLDOWN_SECONDS,
        ip_cooldown_seconds: float = FAUCET_IP_COOLDOWN_SECONDS,
        max_dispenses_per_hour: int = FAUCET_MAX_DISPENSES_PER_HOUR,
        post_rate_per_second: float = POST_RATE_PER_SECOND,
        post_burst: int = POST_BURST,
        max_tracked_keys: int = FAUCET_MAX_TRACKED_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if address_cooldown_seconds < 0 or ip_cooldown_seconds < 0:
            raise TestnetOpsError("faucet cooldowns must be non-negative")
        self.address_cooldown_seconds = address_cooldown_seconds
        self.ip_cooldown_seconds = ip_cooldown_seconds
        self.max_dispenses_per_hour = _positive_int("faucet_max_per_hour", max_dispenses_per_hour)
        self.post_rate_per_second = post_rate_per_second
        self.post_burst = _positive_int("post_burst", post_burst)
        self.max_tracked_keys = _positive_int("max_tracked_keys", max_tracked_keys)
        self._clock = clock
        self._lock = Lock()
        self._address_until: OrderedDict[str, float] = OrderedDict()
        self._ip_until: OrderedDict[str, float] = OrderedDict()
        self._recent: list[float] = []
        self._post_tokens: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def allow_post(self, client_ip: str) -> bool:
        """Charge one POST to ``client_ip``'s token bucket."""

        with self._lock:
            now = self._clock()
            tokens, updated = self._post_tokens.pop(client_ip, (float(self.post_burst), now))
            tokens = min(
                float(self.post_burst),
                tokens + max(0.0, now - updated) * self.post_rate_per_second,
            )
            allowed = tokens >= 1.0
            self._post_tokens[client_ip] = (tokens - 1.0 if allowed else tokens, now)
            self._trim(self._post_tokens)
            return allowed

    def reserve(self, *, address: str, client_ip: str) -> str | None:
        """Reserve one payout or return the reason it is refused."""

        with self._lock:
            now = self._clock()
            if self._address_until.get(address, 0.0) > now:
                return "address recently funded"
            if self._ip_until.get(client_ip, 0.0) > now:
                return "client recently funded"
            self._recent = [stamp for stamp in self._recent if stamp > now - 3600.0]
            if len(self._recent) >= self.max_dispenses_per_hour:
                return "faucet hourly limit reached"
            self._recent.append(now)
            self._address_until[address] = now + self.address_cooldown_seconds
            self._ip_until[client_ip] = now + self.ip_cooldown_seconds
            self._trim(self._address_until)
            self._trim(self._ip_until)
            return None

    def release(self, *, address: str, client_ip: str) -> None:
        """Undo a reservation whose payout was not accepted."""

        with self._lock:
            self._address_until.pop(address, None)
            self._ip_until.pop(client_ip, None)
            if self._recent:
                self._recent.pop()

    def _trim(self, table: OrderedDict[str, Any]) -> None:
        while len(table) > self.max_tracked_keys:
            table.popitem(last=False)


class FaucetChallenge:
    """Stateless, HMAC-signed hashcash challenges bound to the recipient address."""

    def __init__(
        self,
        bits: int = DEFAULT_FAUCET_POW_BITS,
        *,
        ttl_seconds: float = FAUCET_CHALLENGE_TTL_SECONDS,
        secret: bytes | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 1 <= bits <= 32:
            raise TestnetOpsError("faucet pow bits must be in [1, 32]")
        self.bits = bits
        self.ttl_seconds = ttl_seconds
        self._secret = os.urandom(32) if secret is None else secret
        self._clock = clock
        self._lock = Lock()
        self._used: OrderedDict[bytes, float] = OrderedDict()

    def issue(self) -> dict[str, object]:
        expires_at = int(self._clock() + self.ttl_seconds)
        body = expires_at.to_bytes(8, "big") + os.urandom(16)
        token = body + self._mac(body)
        return {"bits": self.bits, "challenge": token.hex(), "expires_at": expires_at}

    def verify(self, *, challenge: object, solution: object, address: str) -> str | None:
        if not isinstance(challenge, str) or not isinstance(solution, str):
            return "challenge and solution are required"
        try:
            token = bytes.fromhex(challenge)
            nonce = bytes.fromhex(solution)
        except ValueError:
            return "challenge and solution must be hex"
        if len(token) != 40 or not 1 <= len(nonce) <= 32:
            return "malformed challenge"
        body, mac = token[:24], token[24:]
        if not hmac.compare_digest(mac, self._mac(body)):
            return "invalid challenge"
        now = self._clock()
        if int.from_bytes(body[:8], "big") < now:
            return "challenge expired"
        digest = blake3(FAUCET_CHALLENGE_DOMAIN + token + address.encode() + nonce).digest()
        if _leading_zero_bits(digest) < self.bits:
            return "insufficient work"
        with self._lock:
            for used_token, expiry in list(self._used.items()):
                if expiry >= now:
                    break
                del self._used[used_token]
            if token in self._used:
                return "challenge already used"
            self._used[token] = float(int.from_bytes(body[:8], "big"))
        return None

    def _mac(self, body: bytes) -> bytes:
        return hmac.new(self._secret, body, sha256).digest()[:16]


def solve_faucet_challenge(challenge: str, bits: int, address: str) -> str:
    """Find a solution for a faucet challenge (client helper and tests)."""

    token = bytes.fromhex(challenge)
    prefix = FAUCET_CHALLENGE_DOMAIN + token + address.encode()
    counter = 0
    while True:
        nonce = counter.to_bytes(8, "big")
        if _leading_zero_bits(blake3(prefix + nonce).digest()) >= bits:
            return nonce.hex()
        counter += 1


def _leading_zero_bits(digest: bytes) -> int:
    value = int.from_bytes(digest, "big")
    return len(digest) * 8 - value.bit_length()


class TestnetHttpServer(ThreadingHTTPServer):
    """Minimal public testnet faucet/explorer HTTP surface."""

    def __init__(
        self,
        address: tuple[str, int],
        rpc_url: str,
        *,
        faucet_state: FaucetState | None = None,
        limiter: FaucetLimiter | None = None,
        challenge: FaucetChallenge | None = None,
        max_connections: int = TESTNET_HTTP_MAX_CONNECTIONS,
    ) -> None:
        super().__init__(address, _TestnetHttpHandler)
        self.rpc_url = _require_rpc_url(rpc_url)
        self.faucet_state = faucet_state
        self.limiter = FaucetLimiter() if limiter is None else limiter
        self.challenge = challenge
        self._connection_slots = BoundedSemaphore(_positive_int("max_connections", max_connections))

    def process_request(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        """Refuse new connections once every slot is busy instead of spawning threads."""

        if not self._connection_slots.acquire(blocking=False):
            if isinstance(request, socket.socket):
                with contextlib.suppress(OSError):
                    request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._connection_slots.release()
            raise

    def process_request_thread(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


def create_testnet_http_server(
    *,
    host: str = "127.0.0.1",
    port: int = 28380,
    rpc_url: str = DEFAULT_TESTNET_RPC_URL,
    faucet_state: FaucetState | None = None,
    limiter: FaucetLimiter | None = None,
    challenge: FaucetChallenge | None = None,
    max_connections: int = TESTNET_HTTP_MAX_CONNECTIONS,
) -> TestnetHttpServer:
    """Create the public testnet helper HTTP server without starting it."""

    return TestnetHttpServer(
        (host, port),
        rpc_url,
        faucet_state=faucet_state,
        limiter=limiter,
        challenge=challenge,
        max_connections=max_connections,
    )


class _TestnetHttpHandler(BaseHTTPRequestHandler):
    server_version = "TensorPoWTestnet"
    protocol_version = "HTTP/1.1"
    timeout = TESTNET_HTTP_TIMEOUT_SECONDS

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        server = cast(TestnetHttpServer, self.server)
        if parsed.path == "/faucet/challenge":
            if server.challenge is None:
                self._send_json({"bits": 0, "challenge": None})
                return
            self._send_json(server.challenge.issue())
            return
        if parsed.path == "/health":
            self._send_json({"chain_id": GENESIS_CHAIN_ID_TESTNET, "healthy": True})
            return
        if parsed.path == "/explorer/mempool":
            self._send_json(rpc_call(server.rpc_url, "getmempool"))
            return
        if parsed.path == "/explorer/block":
            block_hash = parse_qs(parsed.query).get("hash", [""])[0]
            if not _is_canonical_hash_hex(block_hash):
                self._send_json({"error": "invalid block hash"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(rpc_call(server.rpc_url, "getblock", {"block_hash": block_hash}))
            return
        if parsed.path == "/explorer/tx":
            tx_id = parse_qs(parsed.query).get("txid", [""])[0]
            if not _is_canonical_hash_hex(tx_id):
                self._send_json({"error": "invalid txid"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(rpc_call(server.rpc_url, "gettx", {"txid": tx_id}))
            return
        if parsed.path == "/explorer/finality":
            block_hash = parse_qs(parsed.query).get("hash", [""])[0]
            if not _is_canonical_hash_hex(block_hash):
                self._send_json({"error": "invalid block hash"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(rpc_call(server.rpc_url, "getfinality", {"block_hash": block_hash}))
            return
        if parsed.path == "/explorer/utxos":
            address = parse_qs(parsed.query).get("address", [""])[0]
            if not validate_address(address):
                self._send_json({"error": "invalid address"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(rpc_call(server.rpc_url, "getutxos", {"address": address}))
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        server = cast(TestnetHttpServer, self.server)
        if parsed.path != "/faucet":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        client_ip = str(self.client_address[0])
        if not server.limiter.allow_post(client_ip):
            self._send_json({"error": "rate limited"}, status=HTTPStatus.TOO_MANY_REQUESTS)
            return
        body = self._read_json_body()
        if body is None:
            return
        address = body.get("address")
        rawtx = body.get("rawtx")
        if isinstance(rawtx, str):
            self._send_json(rpc_call(server.rpc_url, "sendrawtx", {"rawtx": rawtx}))
            return
        if not isinstance(address, str) or not validate_address(address):
            self._send_json({"error": "invalid address"}, status=HTTPStatus.BAD_REQUEST)
            return
        if server.faucet_state is not None:
            if server.challenge is not None:
                problem = server.challenge.verify(
                    challenge=body.get("challenge"),
                    solution=body.get("solution"),
                    address=address,
                )
                if problem is not None:
                    self._send_json({"error": problem}, status=HTTPStatus.FORBIDDEN)
                    return
            refusal = server.limiter.reserve(address=address, client_ip=client_ip)
            if refusal is not None:
                self._send_json({"error": refusal}, status=HTTPStatus.TOO_MANY_REQUESTS)
                return
            try:
                result = server.faucet_state.dispense(
                    recipient_address=address,
                    rpc_url=server.rpc_url,
                )
            except (OSError, ValueError, TestnetOpsError) as exc:
                server.limiter.release(address=address, client_ip=client_ip)
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if result.get("accepted") is not True:
                server.limiter.release(address=address, client_ip=client_ip)
            self._send_json(result)
            return
        self._send_json(
            {
                "address": address,
                "owner_pubkey_hash": address_to_pubkey_hash(address).hex(),
                "ready": True,
                "submit_rawtx_with": "rawtx",
            }
        )

    def log_message(self, format_: str, *args: object) -> None:
        """Silence stdlib request logging for deterministic tests."""

    def _read_json_body(self) -> Mapping[str, object] | None:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._send_json(
                {"error": "content length required"},
                status=HTTPStatus.LENGTH_REQUIRED,
            )
            return None
        try:
            length = int(content_length)
        except ValueError:
            self._send_json({"error": "invalid content length"}, status=HTTPStatus.BAD_REQUEST)
            return None
        if length < 0 or length > MAX_TESTNET_HTTP_BODY_BYTES:
            self._send_json(
                {"error": "request body too large"},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except TimeoutError:
            self.close_connection = True
            self._send_json({"error": "request timeout"}, status=HTTPStatus.REQUEST_TIMEOUT)
            return None
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "invalid json"}, status=HTTPStatus.BAD_REQUEST)
            return None
        if not isinstance(payload, Mapping):
            self._send_json({"error": "json object required"}, status=HTTPStatus.BAD_REQUEST)
            return None
        return payload

    def _send_json(
        self,
        payload: Mapping[str, object],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def rpc_call(
    rpc_url: str,
    method: str,
    params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Call a TensorPoW JSON-RPC endpoint and return its result object."""

    parsed = urlparse(_require_rpc_url(rpc_url))
    if parsed.scheme != "http" or parsed.hostname is None:
        raise TestnetOpsError("only http RPC URLs are supported")
    body: dict[str, object] = {"jsonrpc": "2.0", "method": method, "id": method}
    if params is not None:
        body["params"] = dict(params)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
    try:
        connection.request(
            "POST",
            parsed.path or "/rpc",
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
    finally:
        connection.close()
    if response.status != HTTPStatus.OK:
        raise TestnetOpsError(f"RPC status {response.status}")
    if not isinstance(payload, dict) or "result" not in payload:
        raise TestnetOpsError("RPC response did not contain result")
    result = payload["result"]
    if not isinstance(result, dict):
        raise TestnetOpsError("RPC result must be an object")
    return result


def _faucet_state_from_args(args: argparse.Namespace) -> FaucetState | None:
    configured = (
        args.faucet_wallet is not None,
        args.faucet_utxos is not None,
        args.faucet_amount is not None,
    )
    if not any(configured):
        return None
    if not all(configured):
        raise TestnetOpsError(
            "--faucet-wallet, --faucet-utxos, and --faucet-amount must be supplied together"
        )
    password = os.environ.get(args.faucet_password_env)
    if not password:
        raise TestnetOpsError(f"{args.faucet_password_env} must be set for public faucet mode")
    return FaucetState(
        wallet=load_wallet(args.faucet_wallet, password),
        utxos_path=args.faucet_utxos,
        amount_matoms=args.faucet_amount,
        fee_matoms=args.faucet_fee,
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestnetOpsError(f"failed to read JSON object: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TestnetOpsError("JSON file must contain an object")
    return payload


def _hash_arg(name: str, value: str) -> bytes:
    if value == "" or value.strip() != value or value.lower() != value:
        raise TestnetOpsError(f"{name} must be canonical lowercase hex")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise TestnetOpsError(f"{name} must be hex") from exc
    if decoded.hex() != value:
        raise TestnetOpsError(f"{name} must be canonical lowercase hex")
    if len(decoded) != 32:
        raise TestnetOpsError(f"{name} must decode to 32 bytes")
    return decoded


def _is_canonical_hash_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        decoded = bytes.fromhex(value)
    except ValueError:
        return False
    return decoded.hex() == value


def _require_rpc_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise TestnetOpsError("rpc_url must be an http URL with a host")
    return value


def _positive_int(name: str, value: int) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise TestnetOpsError(f"{name} must be positive")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TestnetOpsError(f"{name} must be int")
    if value < 0:
        raise TestnetOpsError(f"{name} must be nonnegative")
    return value


def _public_host(public_host: str | None, listen_host: str) -> str:
    host = listen_host if public_host is None else public_host
    if not isinstance(host, str) or host == "" or host.strip() != host:
        raise TestnetOpsError("public_host must be a nonempty host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        if host in ("0.0.0.0", "::") or "/" in host:
            raise TestnetOpsError("public_host must be a public DNS name or IP address") from exc
        _require_public_dns_name(host)
        return host
    if not address.is_global:
        raise TestnetOpsError("public_host IP address must be globally routable")
    return host


def _require_public_dns_name(host: str) -> None:
    if host.lower() != host:
        raise TestnetOpsError("public_host DNS name must be lowercase")
    if host.endswith("."):
        raise TestnetOpsError("public_host DNS name must be canonical")
    if host in ("localhost",) or host.endswith(".localhost") or host.endswith(".local"):
        raise TestnetOpsError("public_host DNS name must be public")
    if "." not in host:
        raise TestnetOpsError("public_host DNS name must contain a public suffix")
    labels = host.rstrip(".").split(".")
    if any(DNS_LABEL_RE.fullmatch(label) is None for label in labels):
        raise TestnetOpsError("public_host DNS name is malformed")
    if _is_reserved_dns_name(host):
        raise TestnetOpsError("public_host DNS name must use a globally delegated public suffix")


def _is_reserved_dns_name(host: str) -> bool:
    labels = tuple(host.split("."))
    reserved_suffixes = (
        ("localhost",),
        ("local",),
        ("test",),
        ("invalid",),
        ("example",),
        ("onion",),
        ("example", "com"),
        ("example", "net"),
        ("example", "org"),
    )
    return any(labels[-len(suffix) :] == suffix for suffix in reserved_suffixes)


def _public_multiaddr(host: str, port: int) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return f"/dns4/{host}/tcp/{port}"
    protocol = "ip6" if address.version == 6 else "ip4"
    return f"/{protocol}/{host}/tcp/{port}"


if __name__ == "__main__":
    raise SystemExit(main())
