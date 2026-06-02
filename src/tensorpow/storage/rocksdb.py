"""RocksDB-backed persistent storage for TensorPoW."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from typing import Final, Literal, cast

from rocksdict import Checkpoint, Options, Rdict, WriteBatch

from tensorpow.chain.blocks import Anchor, Fruit
from tensorpow.chain.headers import AnchorHeader, FruitHeader
from tensorpow.chain.merkle import require_hash
from tensorpow.mempool.shard_tree import ShardTree
from tensorpow.state.utxo import UTXO, Outpoint
from tensorpow.tx.transaction import Transaction

StorageColumn = Literal[
    "headers",
    "bodies",
    "utxo",
    "dag",
    "shard_tree",
    "fee_floors",
    "mempool",
]

COLUMN_HEADERS: Final[StorageColumn] = "headers"
COLUMN_BODIES: Final[StorageColumn] = "bodies"
COLUMN_UTXO: Final[StorageColumn] = "utxo"
COLUMN_DAG: Final[StorageColumn] = "dag"
COLUMN_SHARD_TREE: Final[StorageColumn] = "shard_tree"
COLUMN_FEE_FLOORS: Final[StorageColumn] = "fee_floors"
COLUMN_MEMPOOL: Final[StorageColumn] = "mempool"

STORAGE_COLUMNS: Final[tuple[StorageColumn, ...]] = (
    COLUMN_HEADERS,
    COLUMN_BODIES,
    COLUMN_UTXO,
    COLUMN_DAG,
    COLUMN_SHARD_TREE,
    COLUMN_FEE_FLOORS,
    COLUMN_MEMPOOL,
)

STORAGE_FORMAT_MAGIC: Final[bytes] = b"TPSTOR"
STORAGE_META_KEY: Final[bytes] = b"meta:format"
STORAGE_FORMAT_BYTES: Final[bytes] = STORAGE_FORMAT_MAGIC + b"\x00"

U32_BYTES: Final[int] = 4
U64_BYTES: Final[int] = 8
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF
BENCHMARK_WRITE_COUNT: Final[int] = 10_000
BENCHMARK_MIN_WRITES_PER_SEC: Final[int] = 10_000


class StorageError(ValueError):
    """Raised when storage input bytes or state are invalid."""


@dataclass(frozen=True, slots=True)
class BatchPut:
    """One column-family put operation."""

    column: StorageColumn
    key: bytes
    value: bytes

    def __post_init__(self) -> None:
        _require_column(self.column)
        _require_key("key", self.key)
        _require_bytes("value", self.value)


@dataclass(frozen=True, slots=True)
class BatchDelete:
    """One column-family delete operation."""

    column: StorageColumn
    key: bytes

    def __post_init__(self) -> None:
        _require_column(self.column)
        _require_key("key", self.key)


@dataclass(frozen=True, slots=True)
class StorageBatch:
    """Atomic collection of RocksDB writes."""

    puts: tuple[BatchPut, ...] = ()
    deletes: tuple[BatchDelete, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.puts, tuple):
            raise TypeError("puts must be a tuple")
        if not isinstance(self.deletes, tuple):
            raise TypeError("deletes must be a tuple")
        for put in self.puts:
            if not isinstance(put, BatchPut):
                raise TypeError("puts must contain BatchPut values")
        for delete in self.deletes:
            if not isinstance(delete, BatchDelete):
                raise TypeError("deletes must contain BatchDelete values")

    @property
    def is_empty(self) -> bool:
        """Return whether the batch has no operations."""

        return not self.puts and not self.deletes


@dataclass(frozen=True, slots=True)
class WriteBenchmarkResult:
    """Storage write throughput measurement."""

    writes: int
    elapsed_seconds: float
    writes_per_second: float

    def __post_init__(self) -> None:
        _require_nonnegative_int("writes", self.writes)
        if not isinstance(self.elapsed_seconds, int | float):
            raise TypeError("elapsed_seconds must be numeric")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative")
        if not isinstance(self.writes_per_second, int | float):
            raise TypeError("writes_per_second must be numeric")
        if self.writes_per_second < 0:
            raise ValueError("writes_per_second must be nonnegative")


class RocksDBStore:
    """Column-family storage facade over RocksDB."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._options = _default_options()
        column_options: dict[str, Options] = dict.fromkeys(STORAGE_COLUMNS, self._options)
        self._db = Rdict(str(self.path), self._options, column_families=column_options)
        self._columns = {column: self._db.get_column_family(column) for column in STORAGE_COLUMNS}
        self._handles = {
            column: self._db.get_column_family_handle(column) for column in STORAGE_COLUMNS
        }
        existing_format = self.get(COLUMN_DAG, STORAGE_META_KEY)
        if existing_format is None:
            self.put(COLUMN_DAG, STORAGE_META_KEY, STORAGE_FORMAT_BYTES)
        elif existing_format != STORAGE_FORMAT_BYTES:
            raise StorageError("unsupported TensorPoW storage format")

    @classmethod
    def repair(cls, path: str | Path) -> None:
        """Ask RocksDB to repair an existing database directory."""

        Rdict.repair(str(Path(path)), _default_options())

    def close(self) -> None:
        """Flush and close all column-family handles."""

        first_error: BaseException | None = None
        try:
            self.flush(sync=True)
        except BaseException as error:
            first_error = error
        finally:
            for column in tuple(self._columns.values()):
                try:
                    column.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            self._columns.clear()
            self._handles.clear()
            try:
                self._db.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def flush(self, *, sync: bool = True) -> None:
        """Flush memtables and optionally fsync the WAL."""

        for column in self._columns.values():
            column.flush(wait=True)
        self._db.flush_wal(sync)

    def get(self, column: StorageColumn, key: bytes) -> bytes | None:
        """Return raw bytes for one key in a storage column."""

        column = _require_column(column)
        key = _require_key("key", key)
        value = self._columns[column].get(key)
        if value is None:
            return None
        return _require_bytes("stored value", value)

    def put(self, column: StorageColumn, key: bytes, value: bytes) -> None:
        """Write one raw key/value pair to a column."""

        self.write_batch(StorageBatch(puts=(BatchPut(column, key, value),)))

    def delete(self, column: StorageColumn, key: bytes) -> None:
        """Delete one key from a column."""

        self.write_batch(StorageBatch(deletes=(BatchDelete(column, key),)))

    def write_batch(self, batch: StorageBatch) -> None:
        """Atomically apply a storage batch."""

        if not isinstance(batch, StorageBatch):
            raise TypeError("batch must be StorageBatch")
        if batch.is_empty:
            return
        write_batch = WriteBatch()
        for put in batch.puts:
            write_batch.put(put.key, put.value, self._handles[put.column])
        for delete in batch.deletes:
            write_batch.delete(delete.key, self._handles[delete.column])
        self._db.write(write_batch)

    def items(self, column: StorageColumn) -> Iterator[tuple[bytes, bytes]]:
        """Iterate over items in one column in RocksDB's raw-key order."""

        column = _require_column(column)
        return self._iter_items(column)

    def _iter_items(self, column: StorageColumn) -> Iterator[tuple[bytes, bytes]]:
        for key, value in self._columns[column].items():
            yield (
                _require_bytes("key", cast(bytes, key)),
                _require_bytes("value", cast(bytes, value)),
            )

    def create_checkpoint(self, checkpoint_path: str | Path, *, replace: bool = False) -> Path:
        """Create a physical RocksDB checkpoint and return its path."""

        target = Path(checkpoint_path)
        if target.exists():
            if not replace:
                raise FileExistsError("checkpoint path already exists")
            if target.is_dir():
                rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.flush(sync=True)
        Checkpoint(self._db).create_checkpoint(str(target))
        return target

    def put_header(self, block_hash: bytes, header: FruitHeader | AnchorHeader) -> None:
        """Persist a fruit or anchor header by block hash."""

        require_hash("block_hash", block_hash)
        if not isinstance(header, FruitHeader | AnchorHeader):
            raise TypeError("header must be FruitHeader or AnchorHeader")
        self.put(COLUMN_HEADERS, block_hash, header.serialize())

    def get_header_bytes(self, block_hash: bytes) -> bytes | None:
        """Return serialized header bytes by block hash."""

        require_hash("block_hash", block_hash)
        return self.get(COLUMN_HEADERS, block_hash)

    def put_body(self, block_hash: bytes, body: Fruit | Anchor) -> None:
        """Persist a fruit or anchor body by block hash."""

        require_hash("block_hash", block_hash)
        if not isinstance(body, Fruit | Anchor):
            raise TypeError("body must be Fruit or Anchor")
        self.put(COLUMN_BODIES, block_hash, body.serialize())

    def get_body_bytes(self, block_hash: bytes) -> bytes | None:
        """Return serialized body bytes by block hash."""

        require_hash("block_hash", block_hash)
        return self.get(COLUMN_BODIES, block_hash)

    def put_utxo(self, utxo: UTXO) -> None:
        """Persist one UTXO by sparse outpoint key."""

        if not isinstance(utxo, UTXO):
            raise TypeError("utxo must be UTXO")
        self.put(COLUMN_UTXO, utxo.outpoint_key(), utxo.to_bytes())

    def get_utxo(self, outpoint: Outpoint) -> UTXO | None:
        """Return a persisted UTXO by outpoint."""

        if not isinstance(outpoint, Outpoint):
            raise TypeError("outpoint must be Outpoint")
        key = outpoint.key()
        value = self.get(COLUMN_UTXO, key)
        return None if value is None else UTXO.from_bytes(value, expected_outpoint_key=key)

    def delete_utxo(self, outpoint: Outpoint) -> None:
        """Delete one UTXO by outpoint."""

        if not isinstance(outpoint, Outpoint):
            raise TypeError("outpoint must be Outpoint")
        self.delete(COLUMN_UTXO, outpoint.key())

    def utxos(self) -> Iterator[UTXO]:
        """Iterate over persisted UTXOs in deterministic key order."""

        return (
            UTXO.from_bytes(value, expected_outpoint_key=key)
            for key, value in self.items(COLUMN_UTXO)
        )

    def put_mempool_tx(self, tx: Transaction) -> None:
        """Persist one pending transaction by tx id."""

        if not isinstance(tx, Transaction):
            raise TypeError("tx must be Transaction")
        self.put(COLUMN_MEMPOOL, tx.tx_id(), tx.to_bytes())

    def mempool_txs(self) -> Iterator[Transaction]:
        """Iterate over persisted mempool transactions in tx-id order."""

        return (Transaction.from_bytes(value) for _, value in self.items(COLUMN_MEMPOOL))

    def put_shard_tree(self, tree: ShardTree) -> None:
        """Persist current shard tree state."""

        if not isinstance(tree, ShardTree):
            raise TypeError("tree must be ShardTree")
        self.put(COLUMN_SHARD_TREE, b"current", tree.serialize())

    def get_shard_tree(self) -> ShardTree | None:
        """Return current shard tree state, if present."""

        data = self.get(COLUMN_SHARD_TREE, b"current")
        return None if data is None else ShardTree.deserialize(data)

    def put_fee_floor(self, shard_id: int, floor_matoms_per_kb: int) -> None:
        """Persist current fee floor for one shard."""

        _require_u32("shard_id", shard_id)
        _require_u64("floor_matoms_per_kb", floor_matoms_per_kb)
        self.put(
            COLUMN_FEE_FLOORS,
            shard_id.to_bytes(U32_BYTES, "little"),
            floor_matoms_per_kb.to_bytes(U64_BYTES, "little"),
        )

    def fee_floor(self, shard_id: int) -> int | None:
        """Return persisted current fee floor for one shard."""

        _require_u32("shard_id", shard_id)
        data = self.get(COLUMN_FEE_FLOORS, shard_id.to_bytes(U32_BYTES, "little"))
        return None if data is None else int.from_bytes(_require_len(data, U64_BYTES), "little")

    def benchmark_writes(self, write_count: int = BENCHMARK_WRITE_COUNT) -> WriteBenchmarkResult:
        """Run a deterministic batched write-throughput benchmark."""

        _require_positive_int("write_count", write_count)
        key_prefix = b"bench:"
        batch = StorageBatch(
            puts=tuple(
                BatchPut(
                    COLUMN_DAG,
                    key_prefix + index.to_bytes(U32_BYTES, "little"),
                    index.to_bytes(U64_BYTES, "little"),
                )
                for index in range(write_count)
            )
        )
        start = perf_counter()
        self.write_batch(batch)
        self.flush(sync=True)
        elapsed = perf_counter() - start
        self.write_batch(
            StorageBatch(
                deletes=tuple(
                    BatchDelete(COLUMN_DAG, key_prefix + index.to_bytes(U32_BYTES, "little"))
                    for index in range(write_count)
                )
            )
        )
        return WriteBenchmarkResult(
            writes=write_count,
            elapsed_seconds=elapsed,
            writes_per_second=write_count / max(elapsed, 1e-9),
        )


def atomic_state_batch(
    *,
    headers: Iterable[tuple[bytes, FruitHeader | AnchorHeader]] = (),
    bodies: Iterable[tuple[bytes, Fruit | Anchor]] = (),
    utxo_puts: Iterable[UTXO] = (),
    utxo_deletes: Iterable[Outpoint] = (),
    mempool_puts: Iterable[Transaction] = (),
    mempool_deletes: Iterable[bytes] = (),
) -> StorageBatch:
    """Build one atomic state-transition batch."""

    puts: list[BatchPut] = []
    deletes: list[BatchDelete] = []
    for block_hash, header in headers:
        require_hash("block_hash", block_hash)
        if not isinstance(header, FruitHeader | AnchorHeader):
            raise TypeError("header must be FruitHeader or AnchorHeader")
        puts.append(BatchPut(COLUMN_HEADERS, block_hash, header.serialize()))
    for block_hash, body in bodies:
        require_hash("block_hash", block_hash)
        if not isinstance(body, Fruit | Anchor):
            raise TypeError("body must be Fruit or Anchor")
        puts.append(BatchPut(COLUMN_BODIES, block_hash, body.serialize()))
    for utxo in utxo_puts:
        if not isinstance(utxo, UTXO):
            raise TypeError("utxo_puts must contain UTXO values")
        puts.append(BatchPut(COLUMN_UTXO, utxo.outpoint_key(), utxo.to_bytes()))
    for outpoint in utxo_deletes:
        if not isinstance(outpoint, Outpoint):
            raise TypeError("utxo_deletes must contain Outpoint values")
        deletes.append(BatchDelete(COLUMN_UTXO, outpoint.key()))
    for tx in mempool_puts:
        if not isinstance(tx, Transaction):
            raise TypeError("mempool_puts must contain Transaction values")
        puts.append(BatchPut(COLUMN_MEMPOOL, tx.tx_id(), tx.to_bytes()))
    for tx_id in mempool_deletes:
        require_hash("tx_id", tx_id)
        deletes.append(BatchDelete(COLUMN_MEMPOOL, tx_id))
    return StorageBatch(puts=tuple(puts), deletes=tuple(deletes))


def _default_options() -> Options:
    options = Options()
    options.create_if_missing(True)
    options.create_missing_column_families(True)
    return options


def _require_column(column: StorageColumn) -> StorageColumn:
    if column not in STORAGE_COLUMNS:
        raise ValueError("unknown storage column")
    return column


def _require_key(name: str, value: bytes) -> bytes:
    _require_bytes(name, value)
    if len(value) == 0:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_bytes(name: str, value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    return value


def _require_len(value: bytes, expected_len: int) -> bytes:
    _require_bytes("value", value)
    if len(value) != expected_len:
        raise StorageError("stored value has invalid length")
    return value


def _require_positive_int(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_u32(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value > 0xFFFFFFFF:
        raise ValueError(f"{name} must fit in uint32")
    return value


def _require_u64(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value > U64_MAX:
        raise ValueError(f"{name} must fit in uint64")
    return value


def _require_nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


__all__ = [
    "BENCHMARK_MIN_WRITES_PER_SEC",
    "BENCHMARK_WRITE_COUNT",
    "COLUMN_BODIES",
    "COLUMN_DAG",
    "COLUMN_FEE_FLOORS",
    "COLUMN_HEADERS",
    "COLUMN_MEMPOOL",
    "COLUMN_SHARD_TREE",
    "COLUMN_UTXO",
    "STORAGE_COLUMNS",
    "STORAGE_FORMAT_BYTES",
    "STORAGE_FORMAT_MAGIC",
    "STORAGE_META_KEY",
    "BatchDelete",
    "BatchPut",
    "RocksDBStore",
    "StorageBatch",
    "StorageColumn",
    "StorageError",
    "WriteBenchmarkResult",
    "atomic_state_batch",
]
