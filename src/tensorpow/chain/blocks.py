"""Canonical fruit and anchor body structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tensorpow.chain.headers import (
    AnchorHeader,
    FruitHeader,
    _require_shard_id,
)
from tensorpow.chain.merkle import ordered_merkle_root, require_hash
from tensorpow.crypto.hash import (
    DOMAIN_ANCHOR_REWARD_ROOT,
    DOMAIN_FEE_FLOOR,
    DOMAIN_FRUIT_SET_ROOT,
    DOMAIN_PARENT_CANDIDATE_ROOT,
    DOMAIN_SHARD_TREE,
    DOMAIN_TX_ID,
    DOMAIN_TX_MERKLE_ROOT,
    HASH_LEN_BYTES,
    domain_hash,
    hash_bytes,
)
from tensorpow.pow.challenge import U16_MAX, U32_MAX, U64_MAX
from tensorpow.tx.transaction import Output, TxDecodeError

MAX_FRUIT_PAYLOAD_BYTES: Final[int] = 8192
MIN_FRUIT_TX_COUNT: Final[int] = 1
PARENT_CANDIDATE_MAX_COUNT: Final[int] = 10_000
SHARD_TREE_MAX_BYTES: Final[int] = 262_144


class BlockDecodeError(ValueError):
    """Raised when block body bytes are malformed or non-canonical."""


@dataclass(frozen=True, slots=True)
class Fruit:
    """Fruit body with opaque canonical transaction bytes."""

    header: FruitHeader
    transactions: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.header, FruitHeader):
            raise TypeError("header must be FruitHeader")
        _require_tx_tuple(self.transactions)
        if len(self.transactions) < MIN_FRUIT_TX_COUNT:
            raise ValueError("fruit must contain the coinbase transaction")
        if sum(len(tx) for tx in self.transactions) > MAX_FRUIT_PAYLOAD_BYTES:
            raise ValueError("fruit payload exceeds MAX_FRUIT_PAYLOAD_BYTES")
        if self.tx_merkle_root() != self.header.tx_merkle_root:
            raise ValueError("fruit tx_merkle_root does not match transactions")

    def serialize(self) -> bytes:
        header_bytes = self.header.serialize()
        payload = bytearray()
        payload.extend(_u16(len(header_bytes)))
        payload.extend(header_bytes)
        payload.extend(_u16(len(self.transactions)))
        for tx in self.transactions:
            payload.extend(_u16(len(tx)))
            payload.extend(tx)
        return bytes(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> Fruit:
        reader = _BlockReader(data)
        header = FruitHeader.deserialize(reader.bytes(reader.u16()))
        transactions = tuple(reader.bytes(reader.u16()) for _ in range(reader.u16()))
        reader.finish()
        try:
            return cls(header=header, transactions=transactions)
        except (TypeError, ValueError) as exc:
            raise BlockDecodeError(str(exc)) from exc

    def tx_merkle_root(self) -> bytes:
        return tx_merkle_root(self.transactions)

    def block_hash(self) -> bytes:
        return self.header.header_hash()


@dataclass(frozen=True, slots=True)
class FeeFloorEntry:
    """Per-shard fee floor entry."""

    shard_id: int
    floor_matoms_per_kb: int

    def __post_init__(self) -> None:
        _require_shard_id(self.shard_id)
        _require_uint("floor_matoms_per_kb", self.floor_matoms_per_kb, U64_MAX)

    def serialize(self) -> bytes:
        return _u32(self.shard_id) + _u64(self.floor_matoms_per_kb)


@dataclass(frozen=True, slots=True)
class Anchor:
    """Anchor body with topology and shard/fee-floor commitments."""

    header: AnchorHeader
    covered_fruit_hashes: tuple[bytes, ...]
    parent_candidate_hashes: tuple[bytes, ...]
    shard_tree_bytes: bytes
    fee_floor_entries: tuple[FeeFloorEntry, ...]
    anchor_reward_outputs: tuple[Output, ...] = ()
    genesis_commitment: bytes = bytes(HASH_LEN_BYTES)

    def __post_init__(self) -> None:
        if not isinstance(self.header, AnchorHeader):
            raise TypeError("header must be AnchorHeader")
        _require_hash_tuple("covered_fruit_hashes", self.covered_fruit_hashes)
        _require_hash_tuple("parent_candidate_hashes", self.parent_candidate_hashes)
        if len(self.parent_candidate_hashes) > PARENT_CANDIDATE_MAX_COUNT:
            raise ValueError("too many parent candidates")
        if tuple(sorted(self.covered_fruit_hashes)) != self.covered_fruit_hashes:
            raise ValueError("covered fruit hashes must be strictly ascending")
        _require_bytes("shard_tree_bytes", self.shard_tree_bytes, max_len=SHARD_TREE_MAX_BYTES)
        for entry in self.fee_floor_entries:
            if not isinstance(entry, FeeFloorEntry):
                raise TypeError("fee_floor_entries must contain FeeFloorEntry")
        if not isinstance(self.anchor_reward_outputs, tuple):
            raise TypeError("anchor_reward_outputs must be a tuple")
        for output in self.anchor_reward_outputs:
            if not isinstance(output, Output):
                raise TypeError("anchor_reward_outputs must contain Output values")
        if (
            tuple(sorted(self.fee_floor_entries, key=lambda item: item.shard_id))
            != self.fee_floor_entries
        ):
            raise ValueError("fee floor entries must be sorted by shard_id")
        if len({entry.shard_id for entry in self.fee_floor_entries}) != len(self.fee_floor_entries):
            raise ValueError("duplicate fee floor shard_id")
        require_hash("genesis_commitment", self.genesis_commitment)
        is_genesis_anchor = self.genesis_commitment != bytes(HASH_LEN_BYTES)
        if is_genesis_anchor:
            if self.header.parent_anchor != bytes(HASH_LEN_BYTES):
                raise ValueError("genesis anchor must use GENESIS_PARENT_HASH")
            if self.covered_fruit_hashes:
                raise ValueError("genesis anchor must not cover fruits")
            if self.parent_candidate_hashes:
                raise ValueError("genesis anchor must not carry parent candidates")
            if self.anchor_reward_outputs:
                raise ValueError("genesis anchor must not carry reward outputs")
        elif not self.covered_fruit_hashes:
            raise ValueError("non-genesis anchor must cover at least one fruit")
        if self.fruit_set_root() != self.header.fruit_set_root:
            raise ValueError("fruit_set_root mismatch")
        if self.parent_candidate_root() != self.header.parent_candidate_root:
            raise ValueError("parent_candidate_root mismatch")
        if self.shard_tree_state_root() != self.header.shard_tree_state_root:
            raise ValueError("shard_tree_state_root mismatch")
        if self.fee_floor_set_root() != self.header.fee_floor_set_root:
            raise ValueError("fee_floor_set_root mismatch")
        if self.anchor_reward_root() != self.header.anchor_reward_root:
            raise ValueError("anchor_reward_root mismatch")

    def serialize(self) -> bytes:
        header_bytes = self.header.serialize()
        payload = bytearray()
        payload.extend(_u16(len(header_bytes)))
        payload.extend(header_bytes)
        payload.extend(_u32(len(self.covered_fruit_hashes)))
        payload.extend(b"".join(self.covered_fruit_hashes))
        payload.extend(_u32(len(self.parent_candidate_hashes)))
        payload.extend(b"".join(self.parent_candidate_hashes))
        payload.extend(_u32(len(self.shard_tree_bytes)))
        payload.extend(self.shard_tree_bytes)
        payload.extend(_u32(len(self.fee_floor_entries)))
        for entry in self.fee_floor_entries:
            payload.extend(entry.serialize())
        payload.extend(_u16(len(self.anchor_reward_outputs)))
        for output in self.anchor_reward_outputs:
            output_bytes = output.to_bytes()
            payload.extend(_u16(len(output_bytes)))
            payload.extend(output_bytes)
        payload.extend(self.genesis_commitment)
        return bytes(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> Anchor:
        reader = _BlockReader(data)
        header = AnchorHeader.deserialize(reader.bytes(reader.u16()))
        covered = tuple(reader.bytes(HASH_LEN_BYTES) for _ in range(reader.u32()))
        candidates = tuple(reader.bytes(HASH_LEN_BYTES) for _ in range(reader.u32()))
        shard_tree_bytes = reader.bytes(reader.u32())
        fee_floor_entries = tuple(
            FeeFloorEntry(shard_id=reader.u32(), floor_matoms_per_kb=reader.u64())
            for _ in range(reader.u32())
        )
        anchor_reward_outputs = tuple(
            _decode_anchor_reward_output(reader.bytes(reader.u16())) for _ in range(reader.u16())
        )
        genesis_commitment = reader.bytes(HASH_LEN_BYTES)
        reader.finish()
        try:
            return cls(
                header=header,
                covered_fruit_hashes=covered,
                parent_candidate_hashes=candidates,
                shard_tree_bytes=shard_tree_bytes,
                fee_floor_entries=fee_floor_entries,
                anchor_reward_outputs=anchor_reward_outputs,
                genesis_commitment=genesis_commitment,
            )
        except (TypeError, ValueError) as exc:
            raise BlockDecodeError(str(exc)) from exc

    def fruit_set_root(self) -> bytes:
        return fruit_set_root(self.covered_fruit_hashes)

    def parent_candidate_root(self) -> bytes:
        return parent_candidate_root(self.parent_candidate_hashes)

    def shard_tree_state_root(self) -> bytes:
        return domain_hash(DOMAIN_SHARD_TREE, self.shard_tree_bytes)

    def fee_floor_set_root(self) -> bytes:
        return fee_floor_set_root(self.fee_floor_entries)

    def anchor_reward_root(self) -> bytes:
        return anchor_reward_root(self.anchor_reward_outputs)

    def block_hash(self) -> bytes:
        if self.genesis_commitment != bytes(HASH_LEN_BYTES):
            return hash_bytes(self.serialize())
        return self.header.header_hash()


def tx_id(tx_bytes: bytes) -> bytes:
    return domain_hash(DOMAIN_TX_ID, _require_bytes("tx_bytes", tx_bytes))


def tx_merkle_root(transactions: tuple[bytes, ...]) -> bytes:
    return ordered_merkle_root(DOMAIN_TX_MERKLE_ROOT, tuple(tx_id(tx) for tx in transactions))


def fruit_set_root(fruit_hashes: tuple[bytes, ...]) -> bytes:
    _require_hash_tuple("fruit_hashes", fruit_hashes)
    return ordered_merkle_root(DOMAIN_FRUIT_SET_ROOT, fruit_hashes, reject_duplicates=True)


def parent_candidate_root(parent_candidates: tuple[bytes, ...]) -> bytes:
    _require_hash_tuple("parent_candidates", parent_candidates)
    return ordered_merkle_root(
        DOMAIN_PARENT_CANDIDATE_ROOT,
        parent_candidates,
        reject_duplicates=True,
    )


def fee_floor_set_root(entries: tuple[FeeFloorEntry, ...]) -> bytes:
    return ordered_merkle_root(DOMAIN_FEE_FLOOR, tuple(entry.serialize() for entry in entries))


def anchor_reward_root(outputs: tuple[Output, ...]) -> bytes:
    if not isinstance(outputs, tuple):
        raise TypeError("outputs must be a tuple")
    for output in outputs:
        if not isinstance(output, Output):
            raise TypeError("outputs must contain Output values")
    return ordered_merkle_root(
        DOMAIN_ANCHOR_REWARD_ROOT,
        tuple(output.to_bytes() for output in outputs),
    )


def _decode_anchor_reward_output(data: bytes) -> Output:
    try:
        return Output.from_bytes(data)
    except (TypeError, ValueError, TxDecodeError) as exc:
        raise BlockDecodeError("malformed anchor reward output") from exc


class _BlockReader:
    def __init__(self, data: bytes) -> None:
        self._data = _require_bytes("data", data)
        self._offset = 0

    def bytes(self, length: int) -> bytes:
        _require_uint("length", length, U32_MAX)
        end = self._offset + length
        if end > len(self._data):
            raise BlockDecodeError("truncated block")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def u16(self) -> int:
        return int.from_bytes(self.bytes(2), "little")

    def u32(self) -> int:
        return int.from_bytes(self.bytes(4), "little")

    def u64(self) -> int:
        return int.from_bytes(self.bytes(8), "little")

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise BlockDecodeError("trailing block bytes")


def _require_tx_tuple(transactions: tuple[bytes, ...]) -> None:
    if not isinstance(transactions, tuple):
        raise TypeError("transactions must be a tuple")
    for tx in transactions:
        _require_bytes("transaction", tx, max_len=U16_MAX)


def _require_hash_tuple(name: str, values: tuple[bytes, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    for value in values:
        require_hash(name, value)


def _require_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} outside uint range")


def _require_bytes(name: str, value: bytes, *, max_len: int | None = None) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if max_len is not None and len(value) > max_len:
        raise ValueError(f"{name} exceeds max length")
    return value


def _u16(value: int) -> bytes:
    return value.to_bytes(2, "little")


def _u32(value: int) -> bytes:
    return value.to_bytes(4, "little")


def _u64(value: int) -> bytes:
    return value.to_bytes(8, "little")
