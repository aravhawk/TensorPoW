"""Full-node orchestration for TensorPoW."""

from __future__ import annotations

import asyncio
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Final

from tensorpow.chain.blocks import PARENT_CANDIDATE_MAX_COUNT, Anchor, BlockDecodeError, Fruit
from tensorpow.chain.headers import HeaderDecodeError
from tensorpow.consensus.anchor_daa import (
    ANCHOR_INITIAL_TARGET_LE,
    AnchorRecord,
    next_anchor_target,
)
from tensorpow.consensus.finality import (
    anchor_depth,
    blue_depth,
    finality_tier_from_depths,
    satisfied_finality_tiers,
)
from tensorpow.consensus.ghostdag import DYNAMIC_K_MIN, BlockDAG
from tensorpow.consensus.rewards import (
    coinbase_maturity_height,
    fruit_subsidy_assignments,
    interval_subsidy_matoms,
    reward_pools,
)
from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes
from tensorpow.genesis import GENESIS_CHAIN_ID_MAINNET, GENESIS_CHAIN_ID_TESTNET
from tensorpow.mempool import (
    BYTES_PER_KB,
    Mempool,
    MempoolAddResult,
    ShardTree,
    burned_fee_matoms,
)
from tensorpow.net import LibP2PNode, NodeIdentity
from tensorpow.pow.challenge import GENESIS_PARENT_HASH, PowHeader
from tensorpow.pow.kernel import FRUIT_TARGET_LE, Backend
from tensorpow.pow.verify import verify_pow
from tensorpow.state.utxo import MAX_SUPPLY_MATOMS, TEMPLATE_PKH, UTXO, Outpoint, UTXOSet
from tensorpow.storage import (
    COLUMN_BODIES,
    COLUMN_DAG,
    COLUMN_FEE_FLOORS,
    COLUMN_HEADERS,
    COLUMN_MEMPOOL,
    COLUMN_SHARD_TREE,
    COLUMN_UTXO,
    STORAGE_COLUMNS,
    BatchDelete,
    BatchPut,
    RocksDBStore,
    StorageBatch,
    atomic_state_batch,
)
from tensorpow.tx.script import ScriptError, check_locks, verify_utxo_spend
from tensorpow.tx.transaction import Output, Transaction, TxDecodeError

DEFAULT_CONFIG_NAME: Final[str] = "tensorpow.toml"
DEFAULT_DATA_DIR: Final[str] = "tensorpow-data"
DEFAULT_RPC_HOST: Final[str] = "127.0.0.1"
DEFAULT_RPC_PORT: Final[int] = 28332
DEFAULT_P2P_TCP_PORT: Final[int] = 28333
U16_MAX: Final[int] = 0xFFFF
U64_BYTES: Final[int] = 8
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF
FRUIT_META_PREFIX: Final[bytes] = b"fruitmeta:"
FRUIT_COINBASE_CLAIMED_PREFIX: Final[bytes] = b"fruitcoinbase:"
ANCHOR_HEIGHT_KEY: Final[bytes] = b"meta:anchor-height"
ANCHOR_TIP_KEY: Final[bytes] = b"meta:anchor-tip"
MINTED_SUPPLY_KEY: Final[bytes] = b"meta:minted-supply"
FRUIT_META_BYTES: Final[int] = U64_BYTES * 2
MAX_FUTURE_DRIFT_MS: Final[int] = 120_000
ANCHOR_REWARD_PREFIX: Final[bytes] = b"anchorreward:"

type PowVerifier = Callable[[PowHeader, bytes, Backend], bool]


@dataclass(frozen=True, slots=True)
class _FruitMeta:
    coinbase_claim_matoms: int
    tip_matoms: int


@dataclass(frozen=True, slots=True)
class TensorPowConfig:
    """Node runtime configuration loaded from ``tensorpow.toml``."""

    data_dir: Path = Path(DEFAULT_DATA_DIR)
    enable_network: bool = False
    listen_host: str = DEFAULT_RPC_HOST
    p2p_tcp_port: int = DEFAULT_P2P_TCP_PORT
    rpc_host: str = DEFAULT_RPC_HOST
    rpc_port: int = DEFAULT_RPC_PORT
    peer_key_path: Path | None = None
    enable_mining: bool = False
    chain_id: str | None = None
    expected_genesis_hash: bytes | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        if self.peer_key_path is not None:
            object.__setattr__(self, "peer_key_path", Path(self.peer_key_path))
        _require_bool("enable_network", self.enable_network)
        _require_bool("enable_mining", self.enable_mining)
        _require_host("listen_host", self.listen_host)
        _require_host("rpc_host", self.rpc_host)
        _require_port("p2p_tcp_port", self.p2p_tcp_port)
        _require_port("rpc_port", self.rpc_port)
        if self.chain_id is not None and self.chain_id not in (
            GENESIS_CHAIN_ID_MAINNET,
            GENESIS_CHAIN_ID_TESTNET,
        ):
            raise ValueError("chain_id must be a TensorPoW mainnet or testnet chain id")
        if self.expected_genesis_hash is not None:
            _require_hash("expected_genesis_hash", self.expected_genesis_hash)

    @classmethod
    def from_toml(cls, path: str | Path) -> TensorPowConfig:
        """Load node configuration from a TOML file."""

        config_path = Path(path)
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        node = _require_table(raw, "node")
        network = raw.get("network", {})
        rpc = raw.get("rpc", {})
        mining = raw.get("mining", {})
        chain = raw.get("chain", {})
        if (
            not isinstance(network, dict)
            or not isinstance(rpc, dict)
            or not isinstance(mining, dict)
            or not isinstance(chain, dict)
        ):
            raise ValueError("network, rpc, mining, and chain sections must be TOML tables")
        peer_key = network.get("peer_key_path")
        return cls(
            data_dir=Path(_get_str(node, "data_dir", DEFAULT_DATA_DIR)),
            enable_network=_get_bool(network, "enable", False),
            listen_host=_get_str(network, "listen_host", DEFAULT_RPC_HOST),
            p2p_tcp_port=_get_int(network, "tcp_port", DEFAULT_P2P_TCP_PORT),
            rpc_host=_get_str(rpc, "host", DEFAULT_RPC_HOST),
            rpc_port=_get_int(rpc, "port", DEFAULT_RPC_PORT),
            peer_key_path=(
                None if peer_key is None else Path(_require_str("peer_key_path", peer_key))
            ),
            enable_mining=_get_bool(mining, "enable", False),
            chain_id=_get_optional_str(chain, "chain_id"),
            expected_genesis_hash=_get_optional_hash_hex(chain, "genesis_hash"),
        )

    def to_toml(self) -> str:
        """Return a deterministic TOML representation."""

        peer_key = (
            ""
            if self.peer_key_path is None
            else f'peer_key_path = "{self.peer_key_path.as_posix()}"\n'
        )
        chain_section = ""
        if self.chain_id is not None or self.expected_genesis_hash is not None:
            chain_lines = ["[chain]"]
            if self.chain_id is not None:
                chain_lines.append(f'chain_id = "{self.chain_id}"')
            if self.expected_genesis_hash is not None:
                chain_lines.append(f'genesis_hash = "{self.expected_genesis_hash.hex()}"')
            chain_section = "\n".join(chain_lines) + "\n\n"
        return (
            "[node]\n"
            f'data_dir = "{self.data_dir.as_posix()}"\n\n'
            f"{chain_section}"
            "[network]\n"
            f"enable = {str(self.enable_network).lower()}\n"
            f'listen_host = "{self.listen_host}"\n'
            f"tcp_port = {self.p2p_tcp_port}\n"
            f"{peer_key}\n"
            "[rpc]\n"
            f'host = "{self.rpc_host}"\n'
            f"port = {self.rpc_port}\n\n"
            "[mining]\n"
            f"enable = {str(self.enable_mining).lower()}\n"
        )


@dataclass(frozen=True, slots=True)
class NodeResult:
    """Result of node validation or processing."""

    accepted: bool
    reason: str | None = None
    object_hash: bytes | None = None

    @classmethod
    def ok(cls, object_hash: bytes | None = None) -> NodeResult:
        return cls(accepted=True, object_hash=object_hash)

    @classmethod
    def reject(cls, reason: str) -> NodeResult:
        return cls(accepted=False, reason=reason)

    def __bool__(self) -> bool:
        return self.accepted


class TensorPowNode:
    """Reference full-node coordinator."""

    def __init__(
        self,
        config: TensorPowConfig,
        *,
        pow_backend: Backend = "auto",
        pow_verifier: PowVerifier | None = None,
    ) -> None:
        if not isinstance(config, TensorPowConfig):
            raise TypeError("config must be TensorPowConfig")
        self.config = config
        self._pow_backend = pow_backend
        self._pow_verifier = _default_pow_verifier if pow_verifier is None else pow_verifier
        self._state_lock = RLock()
        self.store = RocksDBStore(config.data_dir / "rocksdb")
        self.shard_tree = self.store.get_shard_tree() or ShardTree()
        self.utxo_set = UTXOSet(self.store.utxos())
        self.mempool = self._rebuild_mempool(self.store.mempool_txs())
        self.network_node: LibP2PNode | None = None
        self._running = False
        self._stop_event = asyncio.Event()

    @property
    def running(self) -> bool:
        """Return whether the node event loop is active."""

        return self._running

    async def start(self) -> None:
        """Start configured node subsystems."""

        if self._running:
            return
        if self.config.enable_network:
            identity = (
                _load_or_create_identity(self.config.peer_key_path)
                if self.config.peer_key_path is not None
                else NodeIdentity.generate()
            )
            self.network_node = LibP2PNode(
                identity=identity,
                listen_addrs=(f"/ip4/{self.config.listen_host}/tcp/{self.config.p2p_tcp_port}",),
            )
            await self.network_node.start()
        self._running = True
        self._stop_event.clear()

    async def stop(self) -> None:
        """Gracefully stop node subsystems and flush storage."""

        if self.network_node is not None:
            await self.network_node.stop()
            self.network_node = None
        self.store.flush()
        self._stop_event.set()
        self._running = False

    async def run_forever(self) -> None:
        """Run until ``stop`` is called."""

        await self.start()
        await self._stop_event.wait()

    def close(self) -> None:
        """Close persistent storage for tests and short-lived commands."""

        self.store.close()

    def process_raw_tx(self, data: bytes) -> tuple[NodeResult, MempoolAddResult | None]:
        """Validate, admit, and persist one inbound transaction."""

        with self._state_lock:
            try:
                tx = Transaction.from_bytes(data)
            except (TypeError, ValueError, TxDecodeError):
                return NodeResult.reject("malformed_tx"), None
            if not tx.inputs:
                return NodeResult.reject("coinbase_not_relayable"), None
            staged_mempool = self._rebuild_mempool(self.mempool_transactions())
            add_result = staged_mempool.add_tx(tx, utxo_view=self.utxo_set)
            if not add_result.accepted:
                return NodeResult.reject(add_result.reason or "mempool_rejected"), add_result
            self.store.put_mempool_tx(tx)
            self.mempool = staged_mempool
            return NodeResult.ok(tx.tx_id()), add_result

    def process_fruit_bytes(self, data: bytes) -> NodeResult:
        """Decode, validate, apply, and persist one fruit."""

        try:
            fruit = Fruit.deserialize(data)
        except (TypeError, ValueError, BlockDecodeError, HeaderDecodeError):
            return NodeResult.reject("malformed_fruit")
        return self.process_fruit(fruit)

    def process_fruit(self, fruit: Fruit) -> NodeResult:
        """Apply one validated fruit body to local state."""

        with self._state_lock:
            if not isinstance(fruit, Fruit):
                raise TypeError("fruit must be Fruit")
            fruit_hash = fruit.block_hash()
            if self.store.get_body_bytes(fruit_hash) is not None:
                return NodeResult.reject("duplicate_fruit")
            try:
                txs = tuple(Transaction.from_bytes(raw) for raw in fruit.transactions)
            except (TypeError, ValueError, TxDecodeError):
                return NodeResult.reject("malformed_fruit_tx")
            if not txs or txs[0].inputs:
                return NodeResult.reject("missing_coinbase")
            if any(not tx.inputs for tx in txs[1:]):
                return NodeResult.reject("extra_coinbase")
            dependency_error, parent_candidates = self._validate_fruit_dependencies(fruit)
            if dependency_error is not None:
                return NodeResult.reject(dependency_error)
            if not self._verify_pow(fruit.header.to_pow_header(parent_candidates), FRUIT_TARGET_LE):
                return NodeResult.reject("fruit_pow_invalid")

            current_anchor_height = self._anchor_height()
            fee_floor = self._fee_floor_for_shard(fruit.header.shard_id)
            validation_view = UTXOSet(self.utxo_set.utxos())
            tip_sum = 0
            for tx in txs[1:]:
                result, tip_matoms = _apply_spend_tx(
                    validation_view,
                    tx,
                    shard_tree=self.shard_tree,
                    required_shard_id=fruit.header.shard_id,
                    fee_floor_matoms_per_kb=fee_floor,
                    current_time_ms=fruit.header.timestamp_ms,
                    current_height=current_anchor_height,
                )
                if result is not None:
                    return NodeResult.reject(result)
                tip_sum += tip_matoms
                for output_index, _output in enumerate(tx.outputs):
                    utxo = _utxo_from_output(tx, output_index)
                    validation_view.add(utxo)

            try:
                coinbase_claim = _tx_output_sum(txs[0])
            except ValueError:
                return NodeResult.reject("coinbase_too_large")
            pre_anchor_subsidy_limit = interval_subsidy_matoms(
                current_anchor_height + 1,
                self._minted_supply(),
            )
            if coinbase_claim > pre_anchor_subsidy_limit + tip_sum:
                return NodeResult.reject("coinbase_too_large")

            state_batch = atomic_state_batch(
                headers=((fruit_hash, fruit.header),),
                bodies=((fruit_hash, fruit),),
                utxo_puts=(),
                utxo_deletes=(),
                mempool_deletes=tuple(tx.tx_id() for tx in txs),
            )
            batch = StorageBatch(
                puts=(
                    *state_batch.puts,
                    BatchPut(
                        COLUMN_DAG,
                        _fruit_meta_key(fruit_hash),
                        _encode_fruit_meta(_FruitMeta(coinbase_claim, tip_sum)),
                    ),
                ),
                deletes=state_batch.deletes,
            )
            self.store.write_batch(batch)
            confirmed_tx_ids = {confirmed.tx_id() for confirmed in txs}
            self.mempool = self._rebuild_mempool(
                tuple(tx for tx in self.store.mempool_txs() if tx.tx_id() not in confirmed_tx_ids)
            )
            return NodeResult.ok(fruit_hash)

    def process_anchor_bytes(self, data: bytes) -> NodeResult:
        """Decode, validate, apply, and persist one anchor."""

        try:
            anchor = Anchor.deserialize(data)
        except (TypeError, ValueError, BlockDecodeError, HeaderDecodeError):
            return NodeResult.reject("malformed_anchor")
        return self.process_anchor(anchor)

    def process_anchor(self, anchor: Anchor) -> NodeResult:
        """Apply one validated anchor body to local state."""

        with self._state_lock:
            if not isinstance(anchor, Anchor):
                raise TypeError("anchor must be Anchor")
            anchor_hash = anchor.block_hash()
            if self.store.get_body_bytes(anchor_hash) is not None:
                if anchor.genesis_commitment != bytes(HASH_LEN_BYTES):
                    return NodeResult.reject("genesis_not_first")
                return NodeResult.reject("duplicate_anchor")
            dependency_error = self._validate_anchor_dependencies(anchor)
            if dependency_error is not None:
                return NodeResult.reject(dependency_error)
            try:
                next_tree = ShardTree.deserialize(anchor.shard_tree_bytes)
            except (TypeError, ValueError):
                return NodeResult.reject("bad_shard_tree")
            if tuple(entry.shard_id for entry in anchor.fee_floor_entries) != tuple(
                next_tree.leaf_shard_ids
            ):
                return NodeResult.reject("bad_fee_floor_entries")

            is_genesis_anchor = anchor.genesis_commitment != bytes(HASH_LEN_BYTES)
            anchor_height = 0 if is_genesis_anchor else self._anchor_height() + 1
            target = ANCHOR_INITIAL_TARGET_LE
            if not is_genesis_anchor:
                history = self._anchor_history(anchor.header.parent_anchor)
                if history is None:
                    return NodeResult.reject("missing_anchor_parent")
                target = next_anchor_target(history)

            staged = UTXOSet(self.utxo_set.utxos())
            apply_result, spent_outpoints, created_utxos, effective_tips, confirmed_tx_ids = (
                self._apply_anchor_fruits(anchor, staged=staged)
            )
            if apply_result is not None:
                return NodeResult.reject(apply_result)
            claim_result, reward_utxos, minted_increment, claim_puts = self._claim_anchor_rewards(
                anchor,
                anchor_hash=anchor_hash,
                anchor_height=anchor_height,
                anchor_target=target,
                effective_tips=effective_tips,
                staged=staged,
            )
            if claim_result is not None:
                return NodeResult.reject(claim_result)
            for utxo in reward_utxos:
                staged.add(utxo)

            next_minted_supply = self._minted_supply() + minted_increment
            puts = [
                BatchPut(COLUMN_HEADERS, anchor_hash, anchor.header.serialize()),
                BatchPut(COLUMN_BODIES, anchor_hash, anchor.serialize()),
                BatchPut(COLUMN_SHARD_TREE, b"current", next_tree.serialize()),
                BatchPut(COLUMN_DAG, ANCHOR_HEIGHT_KEY, _u64(anchor_height)),
                BatchPut(COLUMN_DAG, ANCHOR_TIP_KEY, anchor_hash),
                BatchPut(COLUMN_DAG, MINTED_SUPPLY_KEY, _u64(next_minted_supply)),
                *claim_puts,
            ]
            puts.extend(
                BatchPut(COLUMN_UTXO, utxo.outpoint_key(), utxo.to_bytes())
                for utxo in created_utxos
            )
            puts.extend(
                BatchPut(COLUMN_UTXO, utxo.outpoint_key(), utxo.to_bytes()) for utxo in reward_utxos
            )
            puts.extend(
                BatchPut(
                    COLUMN_FEE_FLOORS,
                    entry.shard_id.to_bytes(4, "little"),
                    entry.floor_matoms_per_kb.to_bytes(8, "little"),
                )
                for entry in anchor.fee_floor_entries
            )
            next_fee_floor_keys = {
                entry.shard_id.to_bytes(4, "little") for entry in anchor.fee_floor_entries
            }
            deletes = [
                BatchDelete(COLUMN_FEE_FLOORS, key)
                for key, _value in self.store.items(COLUMN_FEE_FLOORS)
                if key not in next_fee_floor_keys
            ]
            deletes.extend(BatchDelete(COLUMN_UTXO, outpoint.key()) for outpoint in spent_outpoints)
            deletes.extend(BatchDelete(COLUMN_MEMPOOL, tx_id) for tx_id in confirmed_tx_ids)

            self.store.write_batch(StorageBatch(puts=tuple(puts), deletes=tuple(deletes)))
            self.shard_tree = next_tree
            self.utxo_set = staged
            self.mempool = self._rebuild_mempool(self.store.mempool_txs())
            return NodeResult.ok(anchor_hash)

    def _validate_anchor_dependencies(self, anchor: Anchor) -> str | None:
        is_genesis_anchor = anchor.genesis_commitment != bytes(HASH_LEN_BYTES)
        if anchor.header.timestamp_ms > _now_ms() + MAX_FUTURE_DRIFT_MS:
            return "anchor_time_too_new"
        if is_genesis_anchor:
            if self.store.items(COLUMN_BODIES):
                return "genesis_not_first"
            if (
                self.config.expected_genesis_hash is not None
                and anchor.block_hash() != self.config.expected_genesis_hash
            ):
                return "wrong_genesis"
            return None

        history = self._anchor_history(anchor.header.parent_anchor)
        if history is None:
            return "missing_anchor_parent"
        if history and anchor.header.timestamp_ms <= history[-1].timestamp_ms:
            return "anchor_time_not_increasing"
        target = next_anchor_target(history)
        if not self._verify_pow(anchor.header.to_pow_header(), target):
            return "anchor_pow_invalid"

        for fruit_hash in anchor.covered_fruit_hashes:
            if self._load_fruit(fruit_hash) is None:
                return "missing_covered_fruit"
        for parent_candidate_hash in anchor.parent_candidate_hashes:
            if self._load_fruit(parent_candidate_hash) is None:
                return "missing_parent_candidate"
        canonical_candidates = self._canonical_parent_candidates()
        if canonical_candidates is None:
            return "bad_parent_candidate_order"
        if anchor.parent_candidate_hashes != canonical_candidates:
            return "bad_parent_candidate_order"
        return None

    def _validate_fruit_dependencies(self, fruit: Fruit) -> tuple[str | None, tuple[bytes, ...]]:
        if fruit.header.timestamp_ms > _now_ms() + MAX_FUTURE_DRIFT_MS:
            return "fruit_time_too_new", ()
        latest_anchor = self._load_anchor(fruit.header.latest_anchor)
        if latest_anchor is None:
            return "missing_latest_anchor", ()

        parent_candidates = latest_anchor.parent_candidate_hashes
        try:
            effective_parents = fruit.header.effective_parent_hashes(parent_candidates)
        except ValueError:
            return "bad_parent_bitmap", ()

        selected_parent = fruit.header.parent_selected
        if selected_parent == GENESIS_PARENT_HASH:
            if self._has_stored_fruits():
                return "genesis_fruit_parent_not_first", ()
        elif self._load_fruit(selected_parent) is None:
            return "missing_fruit_parent", ()

        for parent_hash in effective_parents[1:]:
            if self._load_fruit(parent_hash) is None:
                return "missing_fruit_parent", ()
        for parent_hash in effective_parents:
            if parent_hash == GENESIS_PARENT_HASH:
                continue
            parent = self._load_fruit(parent_hash)
            if parent is None:
                return "missing_fruit_parent", ()
            if fruit.header.timestamp_ms <= parent.header.timestamp_ms:
                return "fruit_time_not_after_parent", ()

        selected_chain_timestamps = self._fruit_selected_chain_timestamps(selected_parent)
        if selected_chain_timestamps and fruit.header.timestamp_ms <= _median_int(
            selected_chain_timestamps
        ):
            return "fruit_time_too_old", ()
        return None, parent_candidates

    def _verify_pow(self, header: PowHeader, target: bytes) -> bool:
        try:
            return self._pow_verifier(header, target, self._pow_backend)
        except (TypeError, ValueError, RuntimeError):
            return False

    def _anchor_history(self, tip_hash: bytes) -> tuple[AnchorRecord, ...] | None:
        if tip_hash == GENESIS_PARENT_HASH:
            return None
        chain: list[tuple[bytes, Anchor]] = []
        seen: set[bytes] = set()
        current_hash = tip_hash
        while current_hash != GENESIS_PARENT_HASH:
            if current_hash in seen:
                return None
            seen.add(current_hash)
            anchor = self._load_anchor(current_hash)
            if anchor is None:
                return None
            chain.append((current_hash, anchor))
            current_hash = anchor.header.parent_anchor

        history: list[AnchorRecord] = []
        for anchor_hash, anchor in reversed(chain):
            target = (
                ANCHOR_INITIAL_TARGET_LE
                if anchor.genesis_commitment != bytes(HASH_LEN_BYTES)
                else next_anchor_target(tuple(history))
            )
            try:
                history.append(
                    AnchorRecord(
                        anchor_hash=anchor_hash,
                        parent_anchor=anchor.header.parent_anchor,
                        timestamp_ms=anchor.header.timestamp_ms,
                        target=target,
                    )
                )
            except (TypeError, ValueError):
                return None
        return tuple(history)

    def _fruit_selected_chain_timestamps(self, tip_hash: bytes) -> tuple[int, ...]:
        if tip_hash == GENESIS_PARENT_HASH:
            return ()
        timestamps: list[int] = []
        seen: set[bytes] = set()
        current_hash = tip_hash
        while current_hash != GENESIS_PARENT_HASH and len(timestamps) < 11:
            if current_hash in seen:
                return ()
            seen.add(current_hash)
            fruit = self._load_fruit(current_hash)
            if fruit is None:
                return ()
            timestamps.append(fruit.header.timestamp_ms)
            current_hash = fruit.header.parent_selected
        return tuple(timestamps)

    def _has_stored_fruits(self) -> bool:
        for block_hash, _body in self.store.items(COLUMN_BODIES):
            if self._load_fruit(block_hash) is not None:
                return True
        return False

    def _load_anchor(self, block_hash: bytes) -> Anchor | None:
        if block_hash == GENESIS_PARENT_HASH:
            return None
        try:
            body = self.store.get_body_bytes(block_hash)
        except (TypeError, ValueError):
            return None
        if body is None:
            return None
        try:
            return Anchor.deserialize(body)
        except (TypeError, ValueError, BlockDecodeError, HeaderDecodeError):
            return None

    def _load_fruit(self, block_hash: bytes) -> Fruit | None:
        if block_hash == GENESIS_PARENT_HASH:
            return None
        try:
            body = self.store.get_body_bytes(block_hash)
        except (TypeError, ValueError):
            return None
        if body is None:
            return None
        try:
            return Fruit.deserialize(body)
        except (TypeError, ValueError, BlockDecodeError, HeaderDecodeError):
            return None

    def _apply_anchor_fruits(
        self,
        anchor: Anchor,
        *,
        staged: UTXOSet,
    ) -> tuple[
        str | None, tuple[Outpoint, ...], tuple[UTXO, ...], dict[bytes, int], tuple[bytes, ...]
    ]:
        ordered_fruit_hashes = self._canonical_covered_fruit_order(anchor)
        if ordered_fruit_hashes is None:
            return "bad_covered_fruit_order", (), (), {}, ()

        current_anchor_height = self._anchor_height()
        spent_outpoints: set[Outpoint] = set()
        created_utxos: list[UTXO] = []
        effective_tips: dict[bytes, int] = {}
        covered_tx_ids: set[bytes] = set()
        skipped_tx_ids: set[bytes] = set()
        for fruit_hash in ordered_fruit_hashes:
            fruit = self._load_fruit(fruit_hash)
            if fruit is None:
                return "missing_covered_fruit", (), (), {}, ()
            fee_floor = self._fee_floor_for_shard(fruit.header.shard_id)
            try:
                txs = tuple(Transaction.from_bytes(raw) for raw in fruit.transactions)
            except (TypeError, ValueError, TxDecodeError):
                return "malformed_fruit_tx", (), (), {}, ()
            for tx in txs[1:]:
                tx_id = tx.tx_id()
                covered_tx_ids.add(tx_id)
                if any(
                    input_.previous_outpoint in spent_outpoints
                    or input_.previous_outpoint.tx_id in skipped_tx_ids
                    for input_ in tx.inputs
                ):
                    skipped_tx_ids.add(tx_id)
                    continue
                result, tip_matoms = _apply_spend_tx(
                    staged,
                    tx,
                    shard_tree=self.shard_tree,
                    required_shard_id=fruit.header.shard_id,
                    fee_floor_matoms_per_kb=fee_floor,
                    current_time_ms=fruit.header.timestamp_ms,
                    current_height=current_anchor_height,
                )
                if result is not None:
                    return result, (), (), {}, ()
                for input_ in tx.inputs:
                    spent_outpoints.add(input_.previous_outpoint)
                for output_index, _output in enumerate(tx.outputs):
                    try:
                        utxo = _utxo_from_output(tx, output_index)
                        staged.add(utxo)
                    except (KeyError, ValueError):
                        return "duplicate_tx_outpoint", (), (), {}, ()
                    created_utxos.append(utxo)
                effective_tips[fruit_hash] = effective_tips.get(fruit_hash, 0) + tip_matoms

        return (
            None,
            tuple(sorted(spent_outpoints, key=lambda outpoint: outpoint.to_bytes())),
            tuple(created_utxos),
            effective_tips,
            tuple(sorted(covered_tx_ids)),
        )

    def _canonical_covered_fruit_order(self, anchor: Anchor) -> tuple[bytes, ...] | None:
        dag = self._fruit_dag()
        if dag is None:
            return None
        metadata = dag.ghostdag_metadata(DYNAMIC_K_MIN)
        covered = set(anchor.covered_fruit_hashes)
        if not covered.issubset(metadata):
            return None
        return tuple(
            sorted(
                covered,
                key=lambda fruit_hash: (
                    metadata[fruit_hash].blue_score,
                    dag.get_block(fruit_hash).timestamp_ms,
                    fruit_hash,
                ),
            )
        )

    def _canonical_parent_candidates(self) -> tuple[bytes, ...] | None:
        dag = self._fruit_dag()
        if dag is None:
            return None
        if len(dag) == 0:
            return ()

        referenced: set[bytes] = set()
        fruit_hashes = self._stored_fruit_hashes()
        for fruit_hash in fruit_hashes:
            fruit = self._load_fruit(fruit_hash)
            if fruit is None:
                return None
            parents = self._fruit_parents_for_dag(fruit)
            if parents is None:
                return None
            referenced.update(parents)
        frontier = set(fruit_hashes) - referenced
        metadata = dag.ghostdag_metadata(DYNAMIC_K_MIN)
        return tuple(
            sorted(
                frontier,
                key=lambda fruit_hash: (-metadata[fruit_hash].blue_score, fruit_hash),
            )[:PARENT_CANDIDATE_MAX_COUNT]
        )

    def _fruit_dag(self) -> BlockDAG | None:
        fruits: dict[bytes, Fruit] = {}
        for fruit_hash in self._stored_fruit_hashes():
            fruit = self._load_fruit(fruit_hash)
            if fruit is None:
                return None
            fruits[fruit_hash] = fruit

        dag = BlockDAG()
        visiting: set[bytes] = set()
        visited: set[bytes] = set()

        def add_fruit(fruit_hash: bytes) -> bool:
            if fruit_hash in visited:
                return True
            if fruit_hash in visiting:
                return False
            fruit = fruits.get(fruit_hash)
            if fruit is None:
                return False
            parents = self._fruit_parents_for_dag(fruit)
            if parents is None:
                return False
            visiting.add(fruit_hash)
            for parent_hash in parents:
                if not add_fruit(parent_hash):
                    visiting.remove(fruit_hash)
                    return False
            visiting.remove(fruit_hash)
            try:
                dag.add_fruit(
                    fruit_hash,
                    parents,
                    timestamp_ms=fruit.header.timestamp_ms,
                )
            except (KeyError, TypeError, ValueError):
                return False
            visited.add(fruit_hash)
            return True

        for fruit_hash in sorted(fruits, key=lambda item: (fruits[item].header.timestamp_ms, item)):
            if not add_fruit(fruit_hash):
                return None
        return dag

    def _stored_fruit_hashes(self) -> tuple[bytes, ...]:
        fruit_hashes: list[bytes] = []
        for block_hash, _body in self.store.items(COLUMN_BODIES):
            if self._load_fruit(block_hash) is not None:
                fruit_hashes.append(block_hash)
        return tuple(sorted(fruit_hashes))

    def _best_fruit_tip(self, dag: BlockDAG) -> bytes | None:
        if len(dag) == 0:
            return None
        metadata = dag.ghostdag_metadata(DYNAMIC_K_MIN)
        return sorted(
            metadata,
            key=lambda fruit_hash: (
                -metadata[fruit_hash].blue_score,
                -dag.get_block(fruit_hash).timestamp_ms,
                fruit_hash,
            ),
        )[0]

    def _fruit_parents_for_dag(self, fruit: Fruit) -> tuple[bytes, ...] | None:
        latest_anchor = self._load_anchor(fruit.header.latest_anchor)
        if latest_anchor is None:
            return None
        try:
            parents = fruit.header.effective_parent_hashes(latest_anchor.parent_candidate_hashes)
        except ValueError:
            return None
        return tuple(parent_hash for parent_hash in parents if parent_hash != GENESIS_PARENT_HASH)

    def _claim_anchor_rewards(
        self,
        anchor: Anchor,
        *,
        anchor_hash: bytes,
        anchor_height: int,
        anchor_target: bytes,
        effective_tips: dict[bytes, int],
        staged: UTXOSet,
    ) -> tuple[str | None, tuple[UTXO, ...], int, tuple[BatchPut, ...]]:
        if not anchor.covered_fruit_hashes:
            return None, (), 0, ()

        minted_supply = self._minted_supply()
        interval_subsidy = interval_subsidy_matoms(anchor_height, minted_supply)
        _fruit_pool, anchor_pool = reward_pools(
            fruit_count=len(anchor.covered_fruit_hashes),
            interval_subsidy=interval_subsidy,
            anchor_target=anchor_target,
        )
        try:
            anchor_reward_claim = _output_sum(anchor.anchor_reward_outputs)
        except ValueError:
            return "anchor_reward_too_large", (), 0, ()
        if anchor_reward_claim > anchor_pool:
            return "anchor_reward_too_large", (), 0, ()
        metas: dict[bytes, _FruitMeta] = {}
        coinbases: dict[bytes, Transaction] = {}
        reward_keys: dict[bytes, bytes] = {}
        for fruit_hash in anchor.covered_fruit_hashes:
            if self.store.get(COLUMN_DAG, _fruit_coinbase_claimed_key(fruit_hash)) is not None:
                return "fruit_coinbase_already_claimed", (), 0, ()
            meta = self._fruit_meta(fruit_hash)
            if meta is None:
                return "missing_fruit_metadata", (), 0, ()
            fruit = self._load_fruit(fruit_hash)
            if fruit is None:
                return "missing_covered_fruit", (), 0, ()
            try:
                coinbase = Transaction.from_bytes(fruit.transactions[0])
            except (TypeError, ValueError, TxDecodeError):
                return "malformed_fruit_tx", (), 0, ()
            metas[fruit_hash] = meta
            coinbases[fruit_hash] = coinbase
            reward_keys[fruit_hash] = _coinbase_reward_key(coinbase)

        assignments = fruit_subsidy_assignments(
            reward_keys,
            interval_subsidy=interval_subsidy,
            anchor_target=anchor_target,
        )
        maturity_height = coinbase_maturity_height(anchor_height)
        reward_utxos: list[UTXO] = []
        seen_reward_outpoints: set[Outpoint] = set()
        claim_puts: list[BatchPut] = []
        minted_increment = anchor_reward_claim
        for fruit_hash in anchor.covered_fruit_hashes:
            meta = metas[fruit_hash]
            realized_tip_matoms = effective_tips.get(fruit_hash, 0)
            allowed_claim = assignments[fruit_hash] + realized_tip_matoms
            if meta.coinbase_claim_matoms > allowed_claim:
                return "coinbase_too_large", (), 0, ()
            minted_increment += max(0, meta.coinbase_claim_matoms - realized_tip_matoms)
            coinbase = coinbases[fruit_hash]
            for output_index, _output in enumerate(coinbase.outputs):
                utxo = _utxo_from_output(
                    coinbase,
                    output_index,
                    min_lockheight=maturity_height,
                )
                if utxo.outpoint in seen_reward_outpoints or staged.get(utxo.outpoint) is not None:
                    return "duplicate_coinbase_outpoint", (), 0, ()
                seen_reward_outpoints.add(utxo.outpoint)
                reward_utxos.append(utxo)
            claim_puts.append(
                BatchPut(COLUMN_DAG, _fruit_coinbase_claimed_key(fruit_hash), _u64(anchor_height))
            )

        for output_index, output in enumerate(anchor.anchor_reward_outputs):
            utxo = _utxo_from_anchor_output(anchor_hash, output_index, output)
            if utxo.outpoint in seen_reward_outpoints or staged.get(utxo.outpoint) is not None:
                return "duplicate_anchor_reward_outpoint", (), 0, ()
            seen_reward_outpoints.add(utxo.outpoint)
            reward_utxos.append(utxo)

        if minted_supply + minted_increment > MAX_SUPPLY_MATOMS:
            return "supply_cap_exceeded", (), 0, ()
        return None, tuple(reward_utxos), minted_increment, tuple(claim_puts)

    def mempool_transactions(self) -> tuple[Transaction, ...]:
        """Return current mempool transactions in deterministic order."""

        with self._state_lock:
            return tuple(entry.tx for entry in self.mempool.entries())

    def _rebuild_mempool(self, transactions: tuple[Transaction, ...]) -> Mempool:
        mempool = Mempool(
            shard_tree=self.shard_tree,
            utxo_view=self.utxo_set,
            initial_fee_floors=self._stored_fee_floors(),
        )
        for tx in transactions:
            mempool.add_tx(
                tx,
                utxo_view=self.utxo_set,
                current_height=self._anchor_height(),
            )
        return mempool

    def _stored_fee_floors(self) -> dict[int, int]:
        floors: dict[int, int] = {}
        for key, value in self.store.items(COLUMN_FEE_FLOORS):
            if len(key) != 4 or len(value) != U64_BYTES:
                continue
            floors[int.from_bytes(key, "little")] = int.from_bytes(value, "little")
        return floors

    def _fee_floor_for_shard(self, shard_id: int) -> int:
        return self._stored_fee_floors().get(shard_id, 0)

    def _fruit_meta(self, fruit_hash: bytes) -> _FruitMeta | None:
        return _decode_fruit_meta(self.store.get(COLUMN_DAG, _fruit_meta_key(fruit_hash)))

    def _anchor_height(self) -> int:
        return _decode_u64_or_zero(self.store.get(COLUMN_DAG, ANCHOR_HEIGHT_KEY))

    def _minted_supply(self) -> int:
        return _decode_u64_or_zero(self.store.get(COLUMN_DAG, MINTED_SUPPLY_KEY))

    def sync_from(self, other: TensorPowNode) -> None:
        """Synchronize local storage from another node."""

        if not isinstance(other, TensorPowNode):
            raise TypeError("other must be TensorPowNode")
        if other is self:
            return
        first_lock, second_lock = (
            (self._state_lock, other._state_lock)
            if id(self) < id(other)
            else (other._state_lock, self._state_lock)
        )
        with first_lock, second_lock:
            puts: list[BatchPut] = []
            deletes: list[BatchDelete] = []
            for column in STORAGE_COLUMNS:
                source_items = dict(other.store.items(column))
                target_items = dict(self.store.items(column))
                puts.extend(BatchPut(column, key, value) for key, value in source_items.items())
                deletes.extend(
                    BatchDelete(column, key) for key in target_items if key not in source_items
                )
            self.store.write_batch(StorageBatch(puts=tuple(puts), deletes=tuple(deletes)))
            self.shard_tree = self.store.get_shard_tree() or ShardTree()
            self.utxo_set = UTXOSet(self.store.utxos())
            self.mempool = self._rebuild_mempool(self.store.mempool_txs())

    def get_block(self, block_hash: bytes) -> bytes | None:
        """Return canonical block body bytes by hash."""

        with self._state_lock:
            return self.store.get_body_bytes(block_hash)

    def get_tx(self, tx_id: bytes) -> Transaction | None:
        """Return a mempool transaction by id, if present."""

        with self._state_lock:
            tx = self.mempool.get(tx_id)
            if tx is not None:
                return tx
            raw = self.store.get(COLUMN_MEMPOOL, tx_id)
            return None if raw is None else Transaction.from_bytes(raw)

    def get_finality(self, block_hash: bytes) -> dict[str, object]:
        """Return node-derived finality information for a known fruit block."""

        with self._state_lock:
            if not isinstance(block_hash, bytes) or len(block_hash) != HASH_LEN_BYTES:
                raise ValueError("block_hash must be 32 bytes")
            fruit = self._load_fruit(block_hash)
            if fruit is None:
                return _finality_result(block_hash, seen=False)

            dag = self._fruit_dag()
            if dag is None or not dag.has_block(block_hash):
                return _finality_result(block_hash, seen=True)
            tip = self._best_fruit_tip(dag)
            blue_depth_value = 0 if tip is None else blue_depth(dag, block_hash, tip, DYNAMIC_K_MIN)
            covering_height = _decode_optional_u64(
                self.store.get(COLUMN_DAG, _fruit_coinbase_claimed_key(block_hash))
            )
            anchor_depth_value = anchor_depth(covering_height, self._anchor_height())
            return _finality_result(
                block_hash,
                seen=True,
                blue_depth_value=blue_depth_value,
                anchor_depth_value=anchor_depth_value,
            )

    getfinality = get_finality

    def status(self) -> dict[str, int | bool | str]:
        """Return a compact node status object for CLI/RPC."""

        with self._state_lock:
            return {
                "blocks": len(self.store.items(COLUMN_BODIES)),
                "mempool": len(self.mempool),
                "peers": 0 if self.network_node is None else 1,
                "running": self.running,
                "shard_leaves": len(self.shard_tree.leaf_shard_ids),
                "utxos": len(self.utxo_set),
            }


def write_default_config(path: str | Path, config: TensorPowConfig | None = None) -> Path:
    """Write a default ``tensorpow.toml`` file."""

    target = Path(path)
    if target.is_dir():
        target = target / DEFAULT_CONFIG_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    config = TensorPowConfig() if config is None else config
    target.write_text(config.to_toml(), encoding="utf-8")
    return target


def _apply_spend_tx(
    utxo_set: UTXOSet,
    tx: Transaction,
    *,
    shard_tree: ShardTree,
    required_shard_id: int,
    fee_floor_matoms_per_kb: int,
    current_time_ms: int,
    current_height: int,
) -> tuple[str | None, int]:
    if shard_tree.route_tx(tx.tx_id()) != required_shard_id:
        return "wrong_shard", 0
    try:
        check_locks(
            locktime_ms=tx.locktime_ms,
            lockheight=tx.lockheight,
            current_time_ms=current_time_ms,
            current_height=current_height,
        )
    except (TypeError, ValueError, ScriptError):
        return "tx_lock_unmatured", 0
    seen: set[Outpoint] = set()
    input_sum = 0
    for input_index, input_ in enumerate(tx.inputs):
        outpoint = input_.previous_outpoint
        if outpoint in seen:
            return "duplicate_input", 0
        seen.add(outpoint)
        utxo = utxo_set.get(outpoint)
        if utxo is None:
            return "missing_input", 0
        if not verify_utxo_spend(
            utxo,
            input_.witness,
            tx.sighash(input_index),
            sig_type=tx.sig_type,
            current_time_ms=current_time_ms,
            current_height=current_height,
        ):
            return "script_failed", 0
        input_sum += utxo.amount_matoms
    output_sum = sum(output.amount_matoms for output in tx.outputs)
    if output_sum > input_sum:
        return "negative_fee", 0
    fee_matoms = input_sum - output_sum
    tx_size_bytes = len(tx.to_bytes())
    if fee_matoms * BYTES_PER_KB // tx_size_bytes < fee_floor_matoms_per_kb:
        return "below_fee_floor", 0
    burned = burned_fee_matoms(fee_floor_matoms_per_kb, tx_size_bytes)
    tip_matoms = max(0, fee_matoms - burned)
    for input_ in tx.inputs:
        utxo_set.remove(input_.previous_outpoint)
    return None, tip_matoms


def _utxo_from_output(
    tx: Transaction,
    output_index: int,
    *,
    min_lockheight: int = 0,
) -> UTXO:
    output = tx.outputs[output_index]
    owner_pubkey_hash = output.payload[:32] if output.template_id == TEMPLATE_PKH else bytes(32)
    return UTXO(
        outpoint=Outpoint(tx.tx_id(), output_index),
        amount_matoms=output.amount_matoms,
        template_id=output.template_id,
        owner_pubkey_hash=owner_pubkey_hash,
        locktime_ms=output.locktime_ms,
        lockheight=max(output.lockheight, min_lockheight),
        payload=output.payload,
    )


def _utxo_from_anchor_output(anchor_hash: bytes, output_index: int, output: Output) -> UTXO:
    owner_pubkey_hash = output.payload[:32] if output.template_id == TEMPLATE_PKH else bytes(32)
    return UTXO(
        outpoint=Outpoint(hash_bytes(ANCHOR_REWARD_PREFIX + anchor_hash), output_index),
        amount_matoms=output.amount_matoms,
        template_id=output.template_id,
        owner_pubkey_hash=owner_pubkey_hash,
        locktime_ms=output.locktime_ms,
        lockheight=output.lockheight,
        payload=output.payload,
    )


def _default_pow_verifier(header: PowHeader, target: bytes, backend: Backend) -> bool:
    return verify_pow(header, target, backend=backend)


def _tx_output_sum(tx: Transaction) -> int:
    total = sum(output.amount_matoms for output in tx.outputs)
    if total > MAX_SUPPLY_MATOMS:
        raise ValueError("transaction output sum exceeds MAX_SUPPLY_MATOMS")
    return total


def _output_sum(outputs: tuple[Output, ...]) -> int:
    total = sum(output.amount_matoms for output in outputs)
    if total > MAX_SUPPLY_MATOMS:
        raise ValueError("output sum exceeds MAX_SUPPLY_MATOMS")
    return total


def _finality_result(
    block_hash: bytes,
    *,
    seen: bool,
    blue_depth_value: int = 0,
    anchor_depth_value: int = 0,
) -> dict[str, object]:
    tier = finality_tier_from_depths(blue_depth_value, anchor_depth_value, seen=seen)
    satisfied = sorted(
        finality.value
        for finality in satisfied_finality_tiers(
            blue_depth_value,
            anchor_depth_value,
            seen=seen,
        )
    )
    return {
        "block_hash": block_hash.hex(),
        "seen": seen,
        "tier": tier.value,
        "satisfied_tiers": satisfied,
        "blue_depth": blue_depth_value,
        "anchor_depth": anchor_depth_value,
    }


def _coinbase_reward_key(tx: Transaction) -> bytes:
    pkh_outputs = tuple(
        output.payload
        for output in tx.outputs
        if output.template_id == TEMPLATE_PKH and len(output.payload) == HASH_LEN_BYTES
    )
    return min(pkh_outputs) if pkh_outputs else tx.tx_id()


def _fruit_meta_key(fruit_hash: bytes) -> bytes:
    return FRUIT_META_PREFIX + fruit_hash


def _fruit_coinbase_claimed_key(fruit_hash: bytes) -> bytes:
    return FRUIT_COINBASE_CLAIMED_PREFIX + fruit_hash


def _encode_fruit_meta(meta: _FruitMeta) -> bytes:
    return _u64(meta.coinbase_claim_matoms) + _u64(meta.tip_matoms)


def _decode_fruit_meta(value: bytes | None) -> _FruitMeta | None:
    if value is None:
        return None
    if len(value) != FRUIT_META_BYTES:
        return None
    return _FruitMeta(
        coinbase_claim_matoms=int.from_bytes(value[:U64_BYTES], "little"),
        tip_matoms=int.from_bytes(value[U64_BYTES:], "little"),
    )


def _decode_u64_or_zero(value: bytes | None) -> int:
    if value is None:
        return 0
    if len(value) != U64_BYTES:
        return 0
    return int.from_bytes(value, "little")


def _decode_optional_u64(value: bytes | None) -> int | None:
    if value is None:
        return None
    if len(value) != U64_BYTES:
        return None
    return int.from_bytes(value, "little")


def _median_int(values: tuple[int, ...]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _u64(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("value must be int")
    if not 0 <= value <= U64_MAX:
        raise ValueError("value outside uint64 range")
    return value.to_bytes(U64_BYTES, "little")


def _load_or_create_identity(path: Path) -> NodeIdentity:
    if path.is_file():
        return NodeIdentity(path.read_bytes())
    identity = NodeIdentity.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(identity.private_key_bytes)
    return identity


def _require_table(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} section must be a TOML table")
    return value


def _get_str(table: dict[str, object], name: str, default: str) -> str:
    value = table.get(name, default)
    return _require_str(name, value)


def _get_bool(table: dict[str, object], name: str, default: bool) -> bool:
    value = table.get(name, default)
    _require_bool(name, value)
    return bool(value)


def _get_int(table: dict[str, object], name: str, default: int) -> int:
    value = table.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    return value


def _get_optional_str(table: dict[str, object], name: str) -> str | None:
    value = table.get(name)
    if value is None:
        return None
    return _require_str(name, value)


def _get_optional_hash_hex(table: dict[str, object], name: str) -> bytes | None:
    value = table.get(name)
    if value is None:
        return None
    text = _require_str(name, value)
    if text == "" or text.strip() != text or text.lower() != text:
        raise ValueError(f"{name} must be canonical lowercase hex")
    try:
        decoded = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be hex") from exc
    if decoded.hex() != text:
        raise ValueError(f"{name} must be canonical lowercase hex")
    _require_hash(name, decoded)
    return decoded


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")


def _require_host(name: str, value: str) -> None:
    _require_str(name, value)


def _require_port(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= U16_MAX:
        raise ValueError(f"{name} outside TCP port range")


def _require_hash(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != HASH_LEN_BYTES:
        raise ValueError(f"{name} must be {HASH_LEN_BYTES} bytes")


__all__ = [
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_DATA_DIR",
    "DEFAULT_P2P_TCP_PORT",
    "DEFAULT_RPC_HOST",
    "DEFAULT_RPC_PORT",
    "NodeResult",
    "TensorPowConfig",
    "TensorPowNode",
    "write_default_config",
]
