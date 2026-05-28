"""Per-shard mempool admission, selection, and fee-floor policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from tensorpow.mempool.shard_tree import (
    MAX_FRUIT_PAYLOAD_BYTES,
    ShardId,
    ShardTree,
    require_shard_id,
)
from tensorpow.state.utxo import UTXO, Outpoint
from tensorpow.tx.script import check_locks, verify_utxo_spend
from tensorpow.tx.transaction import MAX_TX_BYTES, Transaction, TxDecodeError

BYTES_PER_KB: Final[int] = 1000
FEE_FLOOR_WINDOW_FRUITS: Final[int] = 1024
FEE_FLOOR_MIN_MATOMS_PER_KB: Final[int] = 0
FEE_FLOOR_EWMA_PREV_WEIGHT: Final[int] = 7
FEE_FLOOR_EWMA_NEW_WEIGHT: Final[int] = 1
FEE_FLOOR_EWMA_DEN: Final[int] = 8


class UTXOView(Protocol):
    """Minimal UTXO lookup interface needed for mempool spend validation."""

    def get(self, outpoint: Outpoint) -> UTXO | None:
        """Return the UTXO for ``outpoint`` when it is currently spendable."""


@dataclass(frozen=True, slots=True)
class MempoolAddResult:
    """Result returned by ``Mempool.add_tx``."""

    accepted: bool
    tx_id: bytes | None = None
    shard_id: ShardId | None = None
    reason: str | None = None
    fee_matoms: int | None = None
    fee_rate_matoms_per_kb: int | None = None

    @classmethod
    def ok(
        cls,
        *,
        tx_id: bytes,
        shard_id: ShardId,
        fee_matoms: int,
        fee_rate_matoms_per_kb: int,
    ) -> MempoolAddResult:
        return cls(
            accepted=True,
            tx_id=tx_id,
            shard_id=shard_id,
            fee_matoms=fee_matoms,
            fee_rate_matoms_per_kb=fee_rate_matoms_per_kb,
        )

    @classmethod
    def reject(
        cls,
        reason: str,
        *,
        tx_id: bytes | None = None,
        shard_id: ShardId | None = None,
    ) -> MempoolAddResult:
        return cls(accepted=False, tx_id=tx_id, shard_id=shard_id, reason=reason)

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True, slots=True)
class FeeFloorSample:
    """One shard fruit's fee and payload contribution to the floor window."""

    shard_id: ShardId
    floor_eligible_fees_matoms: int
    payload_bytes: int

    def __post_init__(self) -> None:
        require_shard_id(self.shard_id)
        _require_nonnegative_int("floor_eligible_fees_matoms", self.floor_eligible_fees_matoms)
        _require_nonnegative_int("payload_bytes", self.payload_bytes)
        if self.payload_bytes > MAX_FRUIT_PAYLOAD_BYTES:
            raise ValueError("payload_bytes exceeds MAX_FRUIT_PAYLOAD_BYTES")


@dataclass(frozen=True, slots=True)
class MempoolEntry:
    """Validated transaction plus deterministic mempool metadata."""

    tx: Transaction
    tx_id: bytes
    shard_id: ShardId
    tx_size_bytes: int
    fee_matoms: int
    fee_rate_matoms_per_kb: int
    is_coinbase: bool


@dataclass(frozen=True, slots=True)
class _SpendAssessment:
    fee_matoms: int
    fee_rate_matoms_per_kb: int
    is_coinbase: bool
    reason: str | None = None


class Mempool:
    """In-memory mempool partitioned by routed leaf shard."""

    def __init__(
        self,
        *,
        shard_tree: ShardTree | None = None,
        utxo_view: UTXOView | None = None,
        initial_fee_floors: Mapping[ShardId, int] | None = None,
    ) -> None:
        self._shard_tree = ShardTree() if shard_tree is None else shard_tree
        if not isinstance(self._shard_tree, ShardTree):
            raise TypeError("shard_tree must be ShardTree")
        self._utxo_view = utxo_view
        self._entries: dict[bytes, MempoolEntry] = {}
        self._entries_by_shard: dict[ShardId, dict[bytes, MempoolEntry]] = {}
        self._spent_outpoints: dict[Outpoint, bytes] = {}
        self._fee_floors: dict[ShardId, int] = {}
        self._fee_histories: dict[ShardId, deque[FeeFloorSample]] = {}
        for shard_id, floor in (initial_fee_floors or {}).items():
            self.set_fee_floor(shard_id, floor)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, tx_id: object) -> bool:
        return isinstance(tx_id, bytes) and tx_id in self._entries

    def add_tx(
        self,
        tx_or_bytes: Transaction | bytes,
        *,
        shard_id: ShardId | None = None,
        utxo_view: UTXOView | None = None,
        current_time_ms: int = 0,
        current_height: int = 0,
    ) -> MempoolAddResult:
        """Validate and add one transaction to its routed shard."""

        decoded = _decode_canonical_tx(tx_or_bytes)
        if decoded is None:
            return MempoolAddResult.reject("malformed")
        tx, raw = decoded
        tx_size_bytes = len(raw)
        if tx_size_bytes > MAX_TX_BYTES:
            return MempoolAddResult.reject("too_large", tx_id=tx.tx_id())

        tx_id = tx.tx_id()
        routed_shard_id = self._shard_tree.route_tx(tx_id)
        if shard_id is not None:
            try:
                requested_shard_id = require_shard_id(shard_id)
            except (TypeError, ValueError):
                return MempoolAddResult.reject("bad_shard", tx_id=tx_id)
            if requested_shard_id != routed_shard_id:
                return MempoolAddResult.reject(
                    "wrong_shard",
                    tx_id=tx_id,
                    shard_id=requested_shard_id,
                )

        if tx_id in self._entries:
            return MempoolAddResult.reject("duplicate", tx_id=tx_id, shard_id=routed_shard_id)

        spend_view = utxo_view if utxo_view is not None else self._utxo_view
        assessment = self._assess_spends(
            tx,
            tx_size_bytes=tx_size_bytes,
            utxo_view=spend_view,
            current_time_ms=current_time_ms,
            current_height=current_height,
        )
        if assessment.reason is not None:
            return MempoolAddResult.reject(
                assessment.reason,
                tx_id=tx_id,
                shard_id=routed_shard_id,
            )

        floor = self.fee_floor(routed_shard_id)
        if not assessment.is_coinbase and assessment.fee_rate_matoms_per_kb < floor:
            return MempoolAddResult.reject("below_fee_floor", tx_id=tx_id, shard_id=routed_shard_id)

        entry = MempoolEntry(
            tx=tx,
            tx_id=tx_id,
            shard_id=routed_shard_id,
            tx_size_bytes=tx_size_bytes,
            fee_matoms=assessment.fee_matoms,
            fee_rate_matoms_per_kb=assessment.fee_rate_matoms_per_kb,
            is_coinbase=assessment.is_coinbase,
        )
        self._entries[tx_id] = entry
        self._entries_by_shard.setdefault(routed_shard_id, {})[tx_id] = entry
        for input_ in tx.inputs:
            self._spent_outpoints[input_.previous_outpoint] = tx_id

        return MempoolAddResult.ok(
            tx_id=tx_id,
            shard_id=routed_shard_id,
            fee_matoms=assessment.fee_matoms,
            fee_rate_matoms_per_kb=assessment.fee_rate_matoms_per_kb,
        )

    def remove(self, tx_id: bytes) -> Transaction | None:
        """Remove and return a pending transaction by id, if present."""

        entry = self._remove_entry(tx_id)
        return None if entry is None else entry.tx

    def contains(self, tx_id: bytes) -> bool:
        """Return whether ``tx_id`` is currently pending."""

        return tx_id in self._entries

    def get(self, tx_id: bytes) -> Transaction | None:
        """Return a pending transaction by id, if present."""

        entry = self._entries.get(tx_id)
        return None if entry is None else entry.tx

    def get_entry(self, tx_id: bytes) -> MempoolEntry | None:
        """Return pending transaction metadata by id, if present."""

        return self._entries.get(tx_id)

    def entries(self) -> tuple[MempoolEntry, ...]:
        """Return all pending entries in deterministic tx-id order."""

        return tuple(self._entries[tx_id] for tx_id in sorted(self._entries))

    def evict_below_floor(self, shard_id: ShardId, floor: int | None = None) -> tuple[bytes, ...]:
        """Evict non-coinbase transactions below ``floor`` for one shard."""

        shard_id = require_shard_id(shard_id)
        if floor is None:
            floor = self.fee_floor(shard_id)
        else:
            self.set_fee_floor(shard_id, floor)
        evicted_tx_ids = tuple(
            sorted(
                tx_id
                for tx_id, entry in self._entries_by_shard.get(shard_id, {}).items()
                if not entry.is_coinbase and entry.fee_rate_matoms_per_kb < floor
            )
        )
        for tx_id in evicted_tx_ids:
            self._remove_entry(tx_id)
        return evicted_tx_ids

    def select_for_fruit(
        self,
        shard_id: ShardId,
        payload_budget: int = MAX_FRUIT_PAYLOAD_BYTES,
    ) -> list[Transaction]:
        """Select pending transactions for one fruit without exceeding payload budget."""

        shard_id = require_shard_id(shard_id)
        _require_nonnegative_int("payload_budget", payload_budget)
        if payload_budget > MAX_FRUIT_PAYLOAD_BYTES:
            raise ValueError("payload_budget exceeds MAX_FRUIT_PAYLOAD_BYTES")

        entries = [
            entry
            for entry in self._entries_by_shard.get(shard_id, {}).values()
            if entry.is_coinbase or entry.fee_rate_matoms_per_kb >= self.fee_floor(shard_id)
        ]
        entries.sort(key=_selection_key)

        selected: list[Transaction] = []
        used_bytes = 0
        for entry in entries:
            if used_bytes + entry.tx_size_bytes > payload_budget:
                continue
            selected.append(entry.tx)
            used_bytes += entry.tx_size_bytes
        return selected

    def set_fee_floor(self, shard_id: ShardId, floor_matoms_per_kb: int) -> None:
        """Set the current local fee floor for one shard."""

        shard_id = require_shard_id(shard_id)
        floor_matoms_per_kb = _require_nonnegative_int(
            "floor_matoms_per_kb",
            floor_matoms_per_kb,
        )
        self._fee_floors[shard_id] = max(FEE_FLOOR_MIN_MATOMS_PER_KB, floor_matoms_per_kb)

    def fee_floor(self, shard_id: ShardId) -> int:
        """Return the current floor for one shard."""

        shard_id = require_shard_id(shard_id)
        return self._fee_floors.get(shard_id, FEE_FLOOR_MIN_MATOMS_PER_KB)

    def record_confirmed_fruit(
        self,
        shard_id: ShardId,
        floor_eligible_fees_matoms: int,
        payload_bytes: int,
    ) -> int:
        """Record one confirmed fruit aggregate and update the shard fee floor."""

        sample = FeeFloorSample(shard_id, floor_eligible_fees_matoms, payload_bytes)
        history = self._fee_histories.setdefault(
            sample.shard_id,
            deque(maxlen=FEE_FLOOR_WINDOW_FRUITS),
        )
        history.append(sample)
        next_floor = self.calculate_next_fee_floor(sample.shard_id)
        self._fee_floors[sample.shard_id] = next_floor
        return next_floor

    def recent_fee_rate(self, shard_id: ShardId) -> int:
        """Return the current window's aggregate fee rate for one shard."""

        shard_id = require_shard_id(shard_id)
        history = self._fee_histories.get(shard_id)
        if not history:
            return FEE_FLOOR_MIN_MATOMS_PER_KB
        total_fees = sum(sample.floor_eligible_fees_matoms for sample in history)
        total_payload_bytes = sum(sample.payload_bytes for sample in history)
        return total_fees * BYTES_PER_KB // max(1, total_payload_bytes)

    def calculate_next_fee_floor(
        self,
        shard_id: ShardId,
        *,
        previous_floor: int | None = None,
    ) -> int:
        """Calculate the next EWMA fee floor from recent shard history."""

        shard_id = require_shard_id(shard_id)
        previous = self.fee_floor(shard_id) if previous_floor is None else previous_floor
        previous = _require_nonnegative_int("previous_floor", previous)
        recent_rate = self.recent_fee_rate(shard_id)
        return max(
            FEE_FLOOR_MIN_MATOMS_PER_KB,
            (FEE_FLOOR_EWMA_PREV_WEIGHT * previous + FEE_FLOOR_EWMA_NEW_WEIGHT * recent_rate)
            // FEE_FLOOR_EWMA_DEN,
        )

    def fee_floor_history(self, shard_id: ShardId) -> tuple[FeeFloorSample, ...]:
        """Return recent fee-floor samples for one shard."""

        shard_id = require_shard_id(shard_id)
        return tuple(self._fee_histories.get(shard_id, ()))

    def _assess_spends(
        self,
        tx: Transaction,
        *,
        tx_size_bytes: int,
        utxo_view: UTXOView | None,
        current_time_ms: int,
        current_height: int,
    ) -> _SpendAssessment:
        if not tx.inputs:
            return _SpendAssessment(
                fee_matoms=0,
                fee_rate_matoms_per_kb=0,
                is_coinbase=True,
            )
        if utxo_view is None:
            return _SpendAssessment(0, 0, False, "missing_utxo_view")
        try:
            check_locks(
                locktime_ms=tx.locktime_ms,
                lockheight=tx.lockheight,
                current_time_ms=current_time_ms,
                current_height=current_height,
            )
        except (TypeError, ValueError):
            return _SpendAssessment(0, 0, False, "tx_lock_unmatured")

        seen_outpoints: set[Outpoint] = set()
        input_sum = 0
        for input_index, input_ in enumerate(tx.inputs):
            outpoint = input_.previous_outpoint
            if outpoint in seen_outpoints:
                return _SpendAssessment(0, 0, False, "duplicate_input")
            seen_outpoints.add(outpoint)

            if outpoint in self._spent_outpoints:
                return _SpendAssessment(0, 0, False, "conflict")

            utxo = utxo_view.get(outpoint)
            if utxo is None:
                return _SpendAssessment(0, 0, False, "missing_input")

            if not verify_utxo_spend(
                utxo,
                input_.witness,
                tx.sighash(input_index),
                sig_type=tx.sig_type,
                current_time_ms=current_time_ms,
                current_height=current_height,
            ):
                return _SpendAssessment(0, 0, False, "script_failed")
            input_sum += utxo.amount_matoms

        output_sum = sum(output.amount_matoms for output in tx.outputs)
        fee_matoms = input_sum - output_sum
        if fee_matoms < 0:
            return _SpendAssessment(0, 0, False, "negative_fee")
        return _SpendAssessment(
            fee_matoms=fee_matoms,
            fee_rate_matoms_per_kb=_fee_rate_matoms_per_kb(fee_matoms, tx_size_bytes),
            is_coinbase=False,
        )

    def _remove_entry(self, tx_id: bytes) -> MempoolEntry | None:
        entry = self._entries.pop(tx_id, None)
        if entry is None:
            return None
        shard_entries = self._entries_by_shard.get(entry.shard_id)
        if shard_entries is not None:
            shard_entries.pop(tx_id, None)
            if not shard_entries:
                del self._entries_by_shard[entry.shard_id]
        for input_ in entry.tx.inputs:
            if self._spent_outpoints.get(input_.previous_outpoint) == tx_id:
                del self._spent_outpoints[input_.previous_outpoint]
        return entry


def burned_fee_matoms(fee_floor_matoms_per_kb: int, tx_size_bytes: int) -> int:
    """Return the fee amount burned at confirmation for a floor and tx size."""

    fee_floor_matoms_per_kb = _require_nonnegative_int(
        "fee_floor_matoms_per_kb",
        fee_floor_matoms_per_kb,
    )
    tx_size_bytes = _require_positive_int("tx_size_bytes", tx_size_bytes)
    return fee_floor_matoms_per_kb * tx_size_bytes // BYTES_PER_KB


def _decode_canonical_tx(tx_or_bytes: Transaction | bytes) -> tuple[Transaction, bytes] | None:
    try:
        raw = tx_or_bytes if isinstance(tx_or_bytes, bytes) else tx_or_bytes.to_bytes()
        tx = Transaction.from_bytes(raw)
    except (TypeError, ValueError, TxDecodeError):
        return None
    if isinstance(tx_or_bytes, Transaction) and tx != tx_or_bytes:
        return None
    return tx, raw


def _selection_key(entry: MempoolEntry) -> tuple[int, bytes]:
    return -entry.fee_rate_matoms_per_kb, entry.tx_id


def _fee_rate_matoms_per_kb(fee_matoms: int, tx_size_bytes: int) -> int:
    fee_matoms = _require_nonnegative_int("fee_matoms", fee_matoms)
    tx_size_bytes = _require_positive_int("tx_size_bytes", tx_size_bytes)
    return fee_matoms * BYTES_PER_KB // tx_size_bytes


def _require_nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_positive_int(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = [
    "BYTES_PER_KB",
    "FEE_FLOOR_EWMA_DEN",
    "FEE_FLOOR_EWMA_NEW_WEIGHT",
    "FEE_FLOOR_EWMA_PREV_WEIGHT",
    "FEE_FLOOR_MIN_MATOMS_PER_KB",
    "FEE_FLOOR_WINDOW_FRUITS",
    "FeeFloorSample",
    "Mempool",
    "MempoolAddResult",
    "MempoolEntry",
    "UTXOView",
    "burned_fee_matoms",
]
