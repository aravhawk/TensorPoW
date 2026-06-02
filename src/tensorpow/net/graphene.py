"""Deterministic Graphene-style compact fruit relay sketches."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from blake3 import blake3

from tensorpow._iblt import peel_iblt_cells
from tensorpow.chain.blocks import Fruit, tx_id
from tensorpow.chain.headers import FruitHeader
from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.mempool.mempool import Mempool, MempoolEntry

CODEC_GRAPHENE: Final[int] = 0x0002
GRAPHENE_RECEIVER_MEMPOOL_PCT: Final[int] = 99
GRAPHENE_TARGET_COMPRESSION_PCT: Final[int] = 95

GRAPHENE_SKETCH_MAGIC: Final[bytes] = b"TPGR"
GRAPHENE_BLOOM_BITS_PER_TX: Final[int] = 2
GRAPHENE_BLOOM_HASH_COUNT: Final[int] = 3
GRAPHENE_MIN_BLOOM_BITS: Final[int] = 8
GRAPHENE_IBLT_HASH_COUNT: Final[int] = 3
GRAPHENE_IBLT_KEY_BYTES: Final[int] = 8
GRAPHENE_IBLT_MIN_CELLS: Final[int] = 3
GRAPHENE_IBLT_OVERHEAD_NUM: Final[int] = 3
GRAPHENE_IBLT_OVERHEAD_DEN: Final[int] = 1

_IBLT_CELL_BYTES: Final[int] = 2 + GRAPHENE_IBLT_KEY_BYTES + 4
_U8_MAX: Final[int] = 0xFF
_U16_MAX: Final[int] = 0xFFFF
_U32_MAX: Final[int] = 0xFFFFFFFF
_SHORT_ID_BYTES_OPTIONS: Final[tuple[int, ...]] = (2, 3, 4, 6, 8, 16, HASH_LEN_BYTES)
_BLOOM_HASH_DOMAIN: Final[bytes] = b"TensorPoW Graphene bloom"
_IBLT_CHECKSUM_DOMAIN: Final[bytes] = b"TensorPoW Graphene IBLT checksum"
_IBLT_POSITION_DOMAIN: Final[bytes] = b"TensorPoW Graphene IBLT position"
_SHORT_ID_DOMAIN: Final[bytes] = b"TensorPoW Graphene short-id"


class _GrapheneDecodeError(ValueError):
    """Raised internally when a sketch is malformed or non-canonical."""


@dataclass(frozen=True, slots=True)
class _GrapheneSketch:
    uncompressed_len: int
    header: FruitHeader
    tx_count: int
    bloom_bits: int
    short_id_bytes: int
    short_id_salt: int
    iblt_cell_count: int
    bloom: bytes
    iblt_cells: tuple[_IBLTCell, ...]
    ordered_short_ids: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class _CandidateTx:
    tx_id: bytes
    raw_tx: bytes


@dataclass(frozen=True, slots=True)
class _IBLTCell:
    count: int
    key_sum: bytes
    checksum_sum: int


def announce_fruit(fruit: Fruit) -> bytes:
    """Return a deterministic Graphene relay sketch for ``fruit``."""

    if not isinstance(fruit, Fruit):
        raise TypeError("fruit must be Fruit")
    if len(fruit.transactions) > _U16_MAX:
        raise ValueError("fruit transaction count exceeds uint16")

    header_bytes = fruit.header.serialize()
    transaction_ids = tuple(tx_id(tx) for tx in fruit.transactions)
    if len(set(transaction_ids)) != len(transaction_ids):
        raise ValueError("fruit transactions must have unique tx_ids")

    short_id_bytes, short_id_salt = _canonical_short_id_params(transaction_ids)
    bloom_bits = _canonical_bloom_bits(len(transaction_ids))
    bloom = _build_bloom(transaction_ids, bloom_bits=bloom_bits)
    iblt_cell_count = _canonical_iblt_cell_count(len(transaction_ids))
    iblt_cells = _build_iblt(transaction_ids, cell_count=iblt_cell_count)
    ordered_short_ids = tuple(
        _short_id(transaction_id, width=short_id_bytes, salt=short_id_salt)
        for transaction_id in transaction_ids
    )

    body = bytearray()
    body.extend(GRAPHENE_SKETCH_MAGIC)
    body.extend(_u16(len(header_bytes)))
    body.extend(header_bytes)
    body.extend(_u16(len(transaction_ids)))
    body.extend(_u16(bloom_bits))
    body.extend(_u8(GRAPHENE_BLOOM_HASH_COUNT))
    body.extend(_u8(short_id_bytes))
    body.extend(_u8(short_id_salt))
    body.extend(_u16(iblt_cell_count))
    body.extend(_u8(GRAPHENE_IBLT_HASH_COUNT))
    body.extend(bloom)
    body.extend(b"".join(_encode_iblt_cell(cell) for cell in iblt_cells))
    body.extend(b"".join(ordered_short_ids))

    full_fruit_len = len(fruit.serialize())
    if full_fruit_len > _U32_MAX or len(body) > _U32_MAX:
        raise ValueError("graphene sketch length exceeds uint32")

    return b"".join(
        (
            _u16(CODEC_GRAPHENE),
            _u32(full_fruit_len),
            _u32(len(body)),
            bytes(body),
        )
    )


def reconstruct_fruit(sketch: bytes, mempool: Mempool) -> Fruit | None:
    """Reconstruct a fruit from ``sketch`` and ``mempool``, or return ``None``."""

    try:
        decoded = _decode_sketch(sketch)
    except (TypeError, ValueError):
        return None

    candidates = _matching_mempool_candidates(decoded, mempool)
    if candidates is None:
        return None
    iblt_difference = _decode_iblt_difference(decoded, candidates.values())
    if iblt_difference is None:
        return None
    sender_only_tx_ids, _receiver_only_tx_ids = iblt_difference
    if sender_only_tx_ids:
        return None

    transactions: list[bytes] = []
    seen_tx_ids: set[bytes] = set()
    for short_id in decoded.ordered_short_ids:
        candidate = candidates.get(short_id)
        if candidate is None or candidate.tx_id in seen_tx_ids:
            return None
        transactions.append(candidate.raw_tx)
        seen_tx_ids.add(candidate.tx_id)

    try:
        fruit = Fruit(header=decoded.header, transactions=tuple(transactions))
    except (TypeError, ValueError):
        return None
    if len(fruit.serialize()) != decoded.uncompressed_len:
        return None
    if announce_fruit(fruit) != sketch:
        return None
    return fruit


def _decode_sketch(sketch: bytes) -> _GrapheneSketch:
    if not isinstance(sketch, bytes):
        raise TypeError("sketch must be bytes")
    reader = _Reader(sketch)
    codec_id = reader.u16()
    if codec_id != CODEC_GRAPHENE:
        raise _GrapheneDecodeError("wrong codec")
    uncompressed_len = reader.u32()
    compressed_len = reader.u32()
    body = reader.bytes(compressed_len)
    reader.finish()
    if len(body) != compressed_len:
        raise _GrapheneDecodeError("truncated sketch body")

    body_reader = _Reader(body)
    if body_reader.bytes(len(GRAPHENE_SKETCH_MAGIC)) != GRAPHENE_SKETCH_MAGIC:
        raise _GrapheneDecodeError("bad graphene magic")
    header_len = body_reader.u16()
    header_bytes = body_reader.bytes(header_len)
    header = FruitHeader.deserialize(header_bytes)
    if header.serialize() != header_bytes:
        raise _GrapheneDecodeError("non-canonical header")

    tx_count = body_reader.u16()
    if tx_count == 0:
        raise _GrapheneDecodeError("empty graphene tx list")
    bloom_bits = body_reader.u16()
    if bloom_bits != _canonical_bloom_bits(tx_count):
        raise _GrapheneDecodeError("non-canonical bloom size")
    bloom_hash_count = body_reader.u8()
    if bloom_hash_count != GRAPHENE_BLOOM_HASH_COUNT:
        raise _GrapheneDecodeError("non-canonical bloom hash count")
    short_id_bytes = body_reader.u8()
    if short_id_bytes not in _SHORT_ID_BYTES_OPTIONS:
        raise _GrapheneDecodeError("bad short-id width")
    short_id_salt = body_reader.u8()
    iblt_cell_count = body_reader.u16()
    if iblt_cell_count != _canonical_iblt_cell_count(tx_count):
        raise _GrapheneDecodeError("non-canonical IBLT cell count")
    iblt_hash_count = body_reader.u8()
    if iblt_hash_count != GRAPHENE_IBLT_HASH_COUNT:
        raise _GrapheneDecodeError("non-canonical IBLT hash count")

    bloom_len = _byte_len(bloom_bits)
    bloom = body_reader.bytes(bloom_len)
    if _has_nonzero_tail_bits(bloom, bloom_bits):
        raise _GrapheneDecodeError("non-canonical bloom tail bits")
    if _bloom_population(bloom) > min(bloom_bits, tx_count * GRAPHENE_BLOOM_HASH_COUNT):
        raise _GrapheneDecodeError("non-canonical bloom population")

    iblt_cells = tuple(
        _decode_iblt_cell(body_reader.bytes(_IBLT_CELL_BYTES)) for _ in range(iblt_cell_count)
    )
    ordered_short_ids = tuple(body_reader.bytes(short_id_bytes) for _ in range(tx_count))
    body_reader.finish()
    if len(set(ordered_short_ids)) != len(ordered_short_ids):
        raise _GrapheneDecodeError("duplicate short ids")
    if uncompressed_len < _minimum_fruit_len(header_len, tx_count):
        raise _GrapheneDecodeError("bad uncompressed length")

    return _GrapheneSketch(
        uncompressed_len=uncompressed_len,
        header=header,
        tx_count=tx_count,
        bloom_bits=bloom_bits,
        short_id_bytes=short_id_bytes,
        short_id_salt=short_id_salt,
        iblt_cell_count=iblt_cell_count,
        bloom=bloom,
        iblt_cells=iblt_cells,
        ordered_short_ids=ordered_short_ids,
    )


def _matching_mempool_candidates(
    sketch: _GrapheneSketch,
    mempool: Mempool,
) -> dict[bytes, _CandidateTx] | None:
    candidates: dict[bytes, _CandidateTx] = {}
    max_candidates = sketch.tx_count + sketch.iblt_cell_count
    for entry in mempool.entries():
        candidate = _candidate_from_entry(entry)
        if candidate is None:
            return None
        if not _bloom_contains(candidate.tx_id, bloom=sketch.bloom, bloom_bits=sketch.bloom_bits):
            continue
        short_id = _short_id(
            candidate.tx_id,
            width=sketch.short_id_bytes,
            salt=sketch.short_id_salt,
        )
        existing = candidates.get(short_id)
        if existing is not None and existing.tx_id != candidate.tx_id:
            return None
        candidates[short_id] = candidate
        if len(candidates) > max_candidates:
            return None
    return candidates


def _candidate_from_entry(entry: MempoolEntry) -> _CandidateTx | None:
    raw_tx = entry.tx.to_bytes()
    if len(entry.tx_id) != HASH_LEN_BYTES or tx_id(raw_tx) != entry.tx_id:
        return None
    return _CandidateTx(tx_id=entry.tx_id, raw_tx=raw_tx)


def _decode_iblt_difference(
    sketch: _GrapheneSketch,
    candidates: Iterable[_CandidateTx],
) -> tuple[tuple[bytes, ...], tuple[bytes, ...]] | None:
    candidate_tuple = tuple(candidates)
    receiver_iblt = _build_iblt(
        tuple(candidate.tx_id for candidate in candidate_tuple),
        cell_count=sketch.iblt_cell_count,
    )
    difference = _subtract_iblt(sketch.iblt_cells, receiver_iblt)
    return _peel_iblt(difference)


def _canonical_iblt_cell_count(tx_count: int) -> int:
    expected_difference = max(1, _ceil_div(tx_count * (100 - GRAPHENE_RECEIVER_MEMPOOL_PCT), 100))
    return max(
        GRAPHENE_IBLT_MIN_CELLS,
        _ceil_div(expected_difference * GRAPHENE_IBLT_OVERHEAD_NUM, GRAPHENE_IBLT_OVERHEAD_DEN),
    )


def _build_iblt(transaction_ids: tuple[bytes, ...], *, cell_count: int) -> tuple[_IBLTCell, ...]:
    _require_iblt_cell_count(cell_count)
    cells = [
        _IBLTCell(count=0, key_sum=bytes(GRAPHENE_IBLT_KEY_BYTES), checksum_sum=0)
        for _ in range(cell_count)
    ]
    iblt_keys = tuple(_iblt_key(transaction_id) for transaction_id in transaction_ids)
    if len(set(iblt_keys)) != len(iblt_keys):
        raise ValueError("duplicate Graphene IBLT keys")
    for transaction_id in transaction_ids:
        _apply_iblt_key(cells, _iblt_key(transaction_id), delta=1)
    return tuple(cells)


def _subtract_iblt(
    left: tuple[_IBLTCell, ...],
    right: tuple[_IBLTCell, ...],
) -> tuple[_IBLTCell, ...]:
    if len(left) != len(right):
        raise _GrapheneDecodeError("IBLT cell count mismatch")
    return tuple(
        _IBLTCell(
            count=left_cell.count - right_cell.count,
            key_sum=_xor_bytes(left_cell.key_sum, right_cell.key_sum),
            checksum_sum=left_cell.checksum_sum ^ right_cell.checksum_sum,
        )
        for left_cell, right_cell in zip(left, right, strict=True)
    )


def _peel_iblt(cells: tuple[_IBLTCell, ...]) -> tuple[tuple[bytes, ...], tuple[bytes, ...]] | None:
    peeled = peel_iblt_cells(
        cells,
        is_pure=_is_pure_iblt_cell,
        is_empty=_is_empty_iblt_cell,
        cell_key=lambda cell: cell.key_sum,
        peel_delta=lambda cell: -1 if cell.count == 1 else 1,
        apply_key=lambda working, key, delta: _apply_iblt_key(working, key, delta=delta),
    )
    if peeled is None:
        return None
    sender_only = tuple(key for key, delta in peeled if delta == -1)
    receiver_only = tuple(key for key, delta in peeled if delta == 1)
    return tuple(sorted(sender_only)), tuple(sorted(receiver_only))


def _apply_iblt_key(cells: list[_IBLTCell], transaction_id: bytes, *, delta: int) -> None:
    _require_iblt_key(transaction_id)
    if delta not in (-1, 1):
        raise ValueError("IBLT delta must be -1 or 1")
    checksum = _iblt_checksum(transaction_id)
    for cell_index in _iblt_positions(transaction_id, len(cells)):
        cell = cells[cell_index]
        cells[cell_index] = _IBLTCell(
            count=cell.count + delta,
            key_sum=_xor_bytes(cell.key_sum, transaction_id),
            checksum_sum=cell.checksum_sum ^ checksum,
        )


def _is_pure_iblt_cell(cell: _IBLTCell) -> bool:
    return (
        abs(cell.count) == 1
        and _require_iblt_key(cell.key_sum) == cell.key_sum
        and cell.checksum_sum == _iblt_checksum(cell.key_sum)
    )


def _is_empty_iblt_cell(cell: _IBLTCell) -> bool:
    return (
        cell.count == 0
        and cell.key_sum == bytes(GRAPHENE_IBLT_KEY_BYTES)
        and cell.checksum_sum == 0
    )


def _iblt_positions(transaction_id: bytes, cell_count: int) -> tuple[int, ...]:
    _require_iblt_key(transaction_id)
    _require_iblt_cell_count(cell_count)
    positions: list[int] = []
    attempt = 0
    while len(positions) < GRAPHENE_IBLT_HASH_COUNT:
        digest = blake3(_IBLT_POSITION_DOMAIN + transaction_id + _u16(attempt)).digest(length=4)
        position = int.from_bytes(digest, "little") % cell_count
        if position not in positions:
            positions.append(position)
        attempt += 1
        if attempt > _U16_MAX:
            raise RuntimeError("unable to derive unique IBLT positions")
    return tuple(positions)


def _iblt_checksum(transaction_id: bytes) -> int:
    _require_iblt_key(transaction_id)
    digest = blake3(_IBLT_CHECKSUM_DOMAIN + transaction_id).digest(length=4)
    return int.from_bytes(digest, "little")


def _encode_iblt_cell(cell: _IBLTCell) -> bytes:
    if cell.count < 0 or cell.count > _U16_MAX:
        raise ValueError("serialized IBLT cell count outside uint16 range")
    _require_iblt_key(cell.key_sum)
    return _u16(cell.count) + cell.key_sum + _u32(cell.checksum_sum)


def _decode_iblt_cell(data: bytes) -> _IBLTCell:
    if len(data) != _IBLT_CELL_BYTES:
        raise _GrapheneDecodeError("bad IBLT cell length")
    count = int.from_bytes(data[:2], "little")
    key_sum = data[2 : 2 + GRAPHENE_IBLT_KEY_BYTES]
    checksum_sum = int.from_bytes(data[2 + GRAPHENE_IBLT_KEY_BYTES :], "little")
    return _IBLTCell(count=count, key_sum=key_sum, checksum_sum=checksum_sum)


def _require_iblt_cell_count(cell_count: int) -> None:
    if not max(GRAPHENE_IBLT_MIN_CELLS, GRAPHENE_IBLT_HASH_COUNT) <= cell_count <= _U16_MAX:
        raise ValueError("IBLT cell count is outside supported range")


def _require_tx_id(transaction_id: bytes) -> bytes:
    if not isinstance(transaction_id, bytes):
        raise TypeError("transaction_id must be bytes")
    if len(transaction_id) != HASH_LEN_BYTES:
        raise ValueError("transaction_id must be 32 bytes")
    return transaction_id


def _require_iblt_key(key: bytes) -> bytes:
    if not isinstance(key, bytes):
        raise TypeError("IBLT key must be bytes")
    if len(key) != GRAPHENE_IBLT_KEY_BYTES:
        raise ValueError("IBLT key has invalid length")
    return key


def _iblt_key(transaction_id: bytes) -> bytes:
    _require_tx_id(transaction_id)
    return transaction_id[:GRAPHENE_IBLT_KEY_BYTES]


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("byte strings must have matching length")
    return bytes(a ^ b for a, b in zip(left, right, strict=True))


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def _canonical_short_id_params(transaction_ids: tuple[bytes, ...]) -> tuple[int, int]:
    for width in _SHORT_ID_BYTES_OPTIONS:
        for salt in range(_U8_MAX + 1):
            short_ids = tuple(
                _short_id(transaction_id, width=width, salt=salt)
                for transaction_id in transaction_ids
            )
            if len(set(short_ids)) == len(short_ids):
                return width, salt
    raise ValueError("unable to choose unique short-id width")


def _short_id(transaction_id: bytes, *, width: int, salt: int) -> bytes:
    return blake3(_SHORT_ID_DOMAIN + bytes((salt,)) + transaction_id).digest(length=width)


def _build_bloom(transaction_ids: tuple[bytes, ...], *, bloom_bits: int) -> bytes:
    bloom = bytearray(_byte_len(bloom_bits))
    for transaction_id in transaction_ids:
        for bit_index in _bloom_positions(transaction_id, bloom_bits):
            bloom[bit_index // 8] |= 1 << (bit_index % 8)
    return bytes(bloom)


def _bloom_contains(transaction_id: bytes, *, bloom: bytes, bloom_bits: int) -> bool:
    return all(
        bloom[bit_index // 8] & (1 << (bit_index % 8))
        for bit_index in _bloom_positions(transaction_id, bloom_bits)
    )


def _bloom_positions(transaction_id: bytes, bloom_bits: int) -> tuple[int, ...]:
    stream = blake3(_BLOOM_HASH_DOMAIN + transaction_id).digest(
        length=GRAPHENE_BLOOM_HASH_COUNT * 4,
    )
    return tuple(
        int.from_bytes(stream[offset : offset + 4], "little") % bloom_bits
        for offset in range(0, len(stream), 4)
    )


def _canonical_bloom_bits(tx_count: int) -> int:
    return max(GRAPHENE_MIN_BLOOM_BITS, tx_count * GRAPHENE_BLOOM_BITS_PER_TX)


def _has_nonzero_tail_bits(bloom: bytes, bloom_bits: int) -> bool:
    tail_bits = bloom_bits % 8
    if tail_bits == 0:
        return False
    tail_mask = (0xFF << tail_bits) & 0xFF
    return bool(bloom[-1] & tail_mask)


def _bloom_population(bloom: bytes) -> int:
    return sum(byte.bit_count() for byte in bloom)


def _minimum_fruit_len(header_len: int, tx_count: int) -> int:
    return 2 + header_len + 2 + (2 * tx_count)


def _byte_len(bit_count: int) -> int:
    return (bit_count + 7) // 8


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def bytes(self, length: int) -> bytes:
        if length < 0:
            raise _GrapheneDecodeError("negative read length")
        end = self._offset + length
        if end > len(self._data):
            raise _GrapheneDecodeError("truncated graphene sketch")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def u8(self) -> int:
        return int.from_bytes(self.bytes(1), "little")

    def u16(self) -> int:
        return int.from_bytes(self.bytes(2), "little")

    def u32(self) -> int:
        return int.from_bytes(self.bytes(4), "little")

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise _GrapheneDecodeError("trailing graphene sketch bytes")


def _u8(value: int) -> bytes:
    if not 0 <= value <= _U8_MAX:
        raise ValueError("value outside uint8 range")
    return value.to_bytes(1, "little")


def _u16(value: int) -> bytes:
    if not 0 <= value <= _U16_MAX:
        raise ValueError("value outside uint16 range")
    return value.to_bytes(2, "little")


def _u32(value: int) -> bytes:
    if not 0 <= value <= _U32_MAX:
        raise ValueError("value outside uint32 range")
    return value.to_bytes(4, "little")


__all__ = [
    "CODEC_GRAPHENE",
    "GRAPHENE_RECEIVER_MEMPOOL_PCT",
    "GRAPHENE_TARGET_COMPRESSION_PCT",
    "announce_fruit",
    "reconstruct_fruit",
]
