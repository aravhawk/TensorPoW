"""Data availability sampling for fruit payloads.

Payload bytes are split into 256-byte cells, padded to a square data matrix,
then extended with two-dimensional Reed-Solomon coding over GF(2^8). Each data
row is extended from ``k`` to ``2k`` symbols, then each resulting column is
extended from ``k`` to ``2k`` symbols. A sample proof ties the sampled cell to
committed row and column witnesses whose Reed-Solomon codewords both verify.

The protocol confidence claim is the probability that random sampling touches
at least one withheld cell:

    detection = 1 - available_fraction ** sample_count

For the protocol regression constants this is ``1 - 0.5**10``, or
99.90234375%. The 75% success threshold is a separate light-node availability
decision rule that tolerates a small number of failed peer responses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Final

from reedsolo import RSCodec  # type: ignore[import-untyped]

from tensorpow.chain.merkle import require_hash
from tensorpow.crypto.hash import (
    DOMAIN_DAS_SAMPLE,
    HASH_LEN_BYTES,
    MERKLE_EMPTY_TAG,
    MERKLE_LEAF_TAG,
    MERKLE_NODE_TAG,
    domain_hash,
)

DAS_CELL_BYTES: Final[int] = 256
DAS_RS_EXTENSION_FACTOR: Final[int] = 2
DAS_RS_PRIMITIVE_POLY: Final[int] = 0x11D
DAS_RS_GENERATOR: Final[int] = 2
DAS_RS_FIRST_CONSECUTIVE_ROOT: Final[int] = 0
DAS_RS_FIELD_EXPONENT: Final[int] = 8
DAS_SAMPLE_SUCCESS_THRESHOLD_PCT: Final[int] = 75
DAS_SAMPLES_PER_FRUIT: Final[int] = 10
DAS_CONFIDENCE_PCT: Final[int] = 99
DAS_WITHHOLDING_PCT: Final[int] = 50
DAS_WITHHOLDING_DETECTION_PCT: Final[int] = 99

U16_BYTES: Final[int] = 2
U32_BYTES: Final[int] = 4
U64_BYTES: Final[int] = 8
DAS_SAMPLE_REQUEST_BYTES: Final[int] = HASH_LEN_BYTES + (3 * U32_BYTES)
U32_MAX: Final[int] = 0xFFFFFFFF
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF
_DAS_MAX_EXTENDED_SIDE: Final[int] = 255
_DAS_MAX_MERKLE_SIBLINGS: Final[int] = 32

_DAS_COMMITMENT_PREFIX: Final[bytes] = b"TensorPoW:DAS:commit"
_DAS_CELL_PREFIX: Final[bytes] = b"TensorPoW:DAS:cell"
_SAMPLE_INDEX_BYTES: Final[int] = U32_BYTES


@dataclass(frozen=True, slots=True)
class DASCommitment:
    """Metadata-bound DAS commitment for one encoded payload."""

    payload_length: int
    data_side: int
    extended_side: int
    cell_root: bytes
    root: bytes

    def __post_init__(self) -> None:
        _require_u64("payload_length", self.payload_length)
        _require_positive_u32("data_side", self.data_side)
        _require_positive_u32("extended_side", self.extended_side)
        if self.extended_side != self.data_side * DAS_RS_EXTENSION_FACTOR:
            raise ValueError("extended_side must be DAS_RS_EXTENSION_FACTOR * data_side")
        require_hash("cell_root", self.cell_root)
        require_hash("root", self.root)
        expected_root = _commitment_root(
            self.payload_length,
            self.data_side,
            self.extended_side,
            self.cell_root,
        )
        if self.root != expected_root:
            raise ValueError("root does not match commitment fields")


@dataclass(frozen=True, slots=True)
class DASEncoding:
    """Encoded DAS cells plus their commitment."""

    commitment: DASCommitment
    cells: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.commitment, DASCommitment):
            raise TypeError("commitment must be DASCommitment")
        if not isinstance(self.cells, tuple):
            raise TypeError("cells must be a tuple")
        side = self.commitment.extended_side
        if len(self.cells) != side * side:
            raise ValueError("cell count must match extended_side squared")
        for cell in self.cells:
            _require_cell("cell", cell)
        cell_root = _ordered_merkle_root(_cell_records(self.cells, side))
        if cell_root != self.commitment.cell_root:
            raise ValueError("cells do not match commitment cell_root")
        if not _grid_has_valid_reed_solomon(self.cells, self.commitment.data_side):
            raise ValueError("cells do not satisfy 2D Reed-Solomon encoding")

    def cell(self, row: int, column: int) -> bytes:
        """Return one encoded cell by extended-grid coordinates."""

        side = self.commitment.extended_side
        return self.cells[_cell_index(side, row, column)]


@dataclass(frozen=True, slots=True)
class DASSampleRequest:
    """One deterministic light-node DAS sample request."""

    fruit_hash: bytes
    sample_index: int
    row: int
    column: int

    def __post_init__(self) -> None:
        require_hash("fruit_hash", self.fruit_hash)
        _require_u32("sample_index", self.sample_index)
        _require_u32("row", self.row)
        _require_u32("column", self.column)


@dataclass(frozen=True, slots=True)
class DASMerkleSibling:
    """One sibling hash in an ordered Merkle proof."""

    is_left: bool
    digest: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.is_left, bool):
            raise TypeError("is_left must be bool")
        require_hash("digest", self.digest)


@dataclass(frozen=True, slots=True)
class DASMerkleProof:
    """Ordered Merkle inclusion proof for a DAS cell record."""

    leaf_index: int
    leaf_count: int
    siblings: tuple[DASMerkleSibling, ...]

    def __post_init__(self) -> None:
        _require_u32("leaf_index", self.leaf_index)
        _require_positive_u32("leaf_count", self.leaf_count)
        if self.leaf_index >= self.leaf_count:
            raise ValueError("leaf_index must be less than leaf_count")
        if not isinstance(self.siblings, tuple):
            raise TypeError("siblings must be a tuple")
        expected_siblings = _ordered_merkle_depth(self.leaf_count)
        if len(self.siblings) != expected_siblings:
            raise ValueError("siblings length does not match leaf_count")
        for sibling in self.siblings:
            if not isinstance(sibling, DASMerkleSibling):
                raise TypeError("siblings must contain DASMerkleSibling values")


@dataclass(frozen=True, slots=True)
class DASSampleProof:
    """Committed row and column witnesses for a sampled DAS cell."""

    row: int
    column: int
    row_cells: tuple[bytes, ...]
    row_proofs: tuple[DASMerkleProof, ...]
    column_cells: tuple[bytes, ...]
    column_proofs: tuple[DASMerkleProof, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_int("row", self.row)
        _require_nonnegative_int("column", self.column)
        if not isinstance(self.row_cells, tuple):
            raise TypeError("row_cells must be a tuple")
        if not isinstance(self.row_proofs, tuple):
            raise TypeError("row_proofs must be a tuple")
        if not isinstance(self.column_cells, tuple):
            raise TypeError("column_cells must be a tuple")
        if not isinstance(self.column_proofs, tuple):
            raise TypeError("column_proofs must be a tuple")
        if len(self.row_cells) != len(self.row_proofs):
            raise ValueError("row_cells and row_proofs lengths must match")
        if len(self.column_cells) != len(self.column_proofs):
            raise ValueError("column_cells and column_proofs lengths must match")
        if len(self.row_cells) == 0 or len(self.column_cells) == 0:
            raise ValueError("sample proof witnesses must be nonempty")
        if len(self.row_cells) != len(self.column_cells):
            raise ValueError("row and column witness lengths must match")
        if self.row >= len(self.column_cells) or self.column >= len(self.row_cells):
            raise ValueError("sample proof coordinates outside witness grid")
        for cell in (*self.row_cells, *self.column_cells):
            _require_cell("cell", cell)
        for proof in (*self.row_proofs, *self.column_proofs):
            if not isinstance(proof, DASMerkleProof):
                raise TypeError("proofs must contain DASMerkleProof values")


def encode_payload(payload: bytes) -> DASEncoding:
    """Cellize and parity-extend fruit payload bytes deterministically."""

    _require_bytes("payload", payload)
    payload_length = len(payload)
    _require_u64("payload_length", payload_length)

    data_cell_count = max(1, _ceil_div(payload_length, DAS_CELL_BYTES))
    data_side = _ceil_sqrt(data_cell_count)
    if data_side >= U32_MAX:
        raise ValueError("payload is too large for DAS side encoding")

    padded_data_cell_count = data_side * data_side
    data_cells = tuple(
        payload[offset : offset + DAS_CELL_BYTES].ljust(DAS_CELL_BYTES, b"\x00")
        for offset in range(0, padded_data_cell_count * DAS_CELL_BYTES, DAS_CELL_BYTES)
    )
    data_rows = tuple(
        data_cells[row * data_side : (row + 1) * data_side] for row in range(data_side)
    )

    extended_rows = _rs_extend_matrix(data_rows)
    cells = tuple(cell for row_cells in extended_rows for cell in row_cells)
    extended_side = data_side * DAS_RS_EXTENSION_FACTOR

    cell_root = _ordered_merkle_root(_cell_records(cells, extended_side))
    commitment = DASCommitment(
        payload_length=payload_length,
        data_side=data_side,
        extended_side=extended_side,
        cell_root=cell_root,
        root=_commitment_root(payload_length, data_side, extended_side, cell_root),
    )
    return DASEncoding(commitment=commitment, cells=cells)


def select_sample_request(
    fruit_hash: bytes,
    commitment: DASCommitment,
    sample_index: int,
) -> DASSampleRequest:
    """Select one sample using DOMAIN_DAS_SAMPLE randomness.

    ``sample_index`` is encoded as a little-endian uint32 for the
    ``sample_index_le`` field named by the protocol spec.
    """

    require_hash("fruit_hash", fruit_hash)
    if not isinstance(commitment, DASCommitment):
        raise TypeError("commitment must be DASCommitment")
    _require_u32("sample_index", sample_index)
    side = commitment.extended_side
    offset = _uniform_sample_offset(fruit_hash, sample_index, side * side)
    row, column = divmod(offset, side)
    return DASSampleRequest(
        fruit_hash=fruit_hash,
        sample_index=sample_index,
        row=row,
        column=column,
    )


def select_sample_requests(
    fruit_hash: bytes,
    commitment: DASCommitment,
    sample_count: int = DAS_SAMPLES_PER_FRUIT,
) -> tuple[DASSampleRequest, ...]:
    """Select deterministic DAS sample requests for one fruit."""

    _require_nonnegative_int("sample_count", sample_count)
    return tuple(
        select_sample_request(fruit_hash, commitment, sample_index)
        for sample_index in range(sample_count)
    )


def create_sample_proof(encoding: DASEncoding, request: DASSampleRequest) -> DASSampleProof:
    """Build a committed row and column proof for a sample request."""

    if not isinstance(encoding, DASEncoding):
        raise TypeError("encoding must be DASEncoding")
    if not isinstance(request, DASSampleRequest):
        raise TypeError("request must be DASSampleRequest")
    expected_request = select_sample_request(
        request.fruit_hash,
        encoding.commitment,
        request.sample_index,
    )
    if request != expected_request:
        raise ValueError("request coordinates do not match deterministic DAS sampling")

    side = encoding.commitment.extended_side
    records = _cell_records(encoding.cells, side)
    layers = _ordered_merkle_layers(records)

    row_cells = tuple(encoding.cell(request.row, column) for column in range(side))
    row_proofs = tuple(
        _ordered_merkle_proof_from_layers(layers, _cell_index(side, request.row, column))
        for column in range(side)
    )
    column_cells = tuple(encoding.cell(row, request.column) for row in range(side))
    column_proofs = tuple(
        _ordered_merkle_proof_from_layers(layers, _cell_index(side, row, request.column))
        for row in range(side)
    )
    return DASSampleProof(
        row=request.row,
        column=request.column,
        row_cells=row_cells,
        row_proofs=row_proofs,
        column_cells=column_cells,
        column_proofs=column_proofs,
    )


def verify_sample(
    request: DASSampleRequest,
    commitment: DASCommitment,
    proof: DASSampleProof,
) -> bool:
    """Verify a DAS sample response against an expected commitment."""

    try:
        expected_request = select_sample_request(
            request.fruit_hash,
            commitment,
            request.sample_index,
        )
        if request != expected_request:
            return False

        side = commitment.extended_side
        if proof.row != request.row or proof.column != request.column:
            return False
        if len(proof.row_cells) != side or len(proof.row_proofs) != side:
            return False
        if len(proof.column_cells) != side or len(proof.column_proofs) != side:
            return False
        if proof.row_cells[request.column] != proof.column_cells[request.row]:
            return False

        for column, cell in enumerate(proof.row_cells):
            if not _verify_cell_proof(
                cell=cell,
                row=request.row,
                column=column,
                side=side,
                proof=proof.row_proofs[column],
                cell_root=commitment.cell_root,
            ):
                return False
        for row, cell in enumerate(proof.column_cells):
            if not _verify_cell_proof(
                cell=cell,
                row=row,
                column=request.column,
                side=side,
                proof=proof.column_proofs[row],
                cell_root=commitment.cell_root,
            ):
                return False

        return _is_valid_rs_codeword(
            proof.row_cells,
            commitment.data_side,
        ) and _is_valid_rs_codeword(
            proof.column_cells,
            commitment.data_side,
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def encode_sample_request(request: DASSampleRequest) -> bytes:
    """Encode one DAS sample request for ``MSG_TYPE_DAS_REQUEST`` payloads."""

    if not isinstance(request, DASSampleRequest):
        raise TypeError("request must be DASSampleRequest")
    return b"".join(
        (
            request.fruit_hash,
            request.sample_index.to_bytes(U32_BYTES, "little"),
            request.row.to_bytes(U32_BYTES, "little"),
            request.column.to_bytes(U32_BYTES, "little"),
        )
    )


def decode_sample_request(data: bytes) -> DASSampleRequest:
    """Decode one DAS sample request payload."""

    _require_bytes("data", data)
    if len(data) != DAS_SAMPLE_REQUEST_BYTES:
        raise ValueError("DAS sample request length is invalid")
    reader = _Reader(data)
    request = DASSampleRequest(
        fruit_hash=reader.bytes(HASH_LEN_BYTES),
        sample_index=reader.u32(),
        row=reader.u32(),
        column=reader.u32(),
    )
    reader.finish()
    return request


def encode_sample_response(proof: DASSampleProof) -> bytes:
    """Encode one DAS sample response proof for ``MSG_TYPE_DAS_RESPONSE`` payloads."""

    if not isinstance(proof, DASSampleProof):
        raise TypeError("proof must be DASSampleProof")
    body = bytearray()
    body.extend(proof.row.to_bytes(U32_BYTES, "little"))
    body.extend(proof.column.to_bytes(U32_BYTES, "little"))
    body.extend(len(proof.row_cells).to_bytes(U32_BYTES, "little"))
    for cell, merkle_proof in zip(proof.row_cells, proof.row_proofs, strict=True):
        body.extend(cell)
        body.extend(_encode_merkle_proof(merkle_proof))
    body.extend(len(proof.column_cells).to_bytes(U32_BYTES, "little"))
    for cell, merkle_proof in zip(proof.column_cells, proof.column_proofs, strict=True):
        body.extend(cell)
        body.extend(_encode_merkle_proof(merkle_proof))
    return bytes(body)


def decode_sample_response(data: bytes) -> DASSampleProof:
    """Decode one DAS sample response proof payload."""

    _require_bytes("data", data)
    reader = _Reader(data)
    row = reader.u32()
    column = reader.u32()
    row_count = _require_wire_side("row_count", reader.u32())
    row_cells: list[bytes] = []
    row_proofs: list[DASMerkleProof] = []
    for _ in range(row_count):
        row_cells.append(reader.bytes(DAS_CELL_BYTES))
        row_proofs.append(_decode_merkle_proof(reader))
    column_count = _require_wire_side("column_count", reader.u32())
    column_cells: list[bytes] = []
    column_proofs: list[DASMerkleProof] = []
    for _ in range(column_count):
        column_cells.append(reader.bytes(DAS_CELL_BYTES))
        column_proofs.append(_decode_merkle_proof(reader))
    reader.finish()
    return DASSampleProof(
        row=row,
        column=column,
        row_cells=tuple(row_cells),
        row_proofs=tuple(row_proofs),
        column_cells=tuple(column_cells),
        column_proofs=tuple(column_proofs),
    )


def is_available(
    sample_successes: Sequence[bool],
    threshold_pct: int = DAS_SAMPLE_SUCCESS_THRESHOLD_PCT,
) -> bool:
    """Return the light-node availability decision for sample outcomes."""

    _require_pct("threshold_pct", threshold_pct)
    if not isinstance(sample_successes, Sequence):
        raise TypeError("sample_successes must be a sequence")
    if len(sample_successes) == 0:
        return False

    successes = 0
    for success in sample_successes:
        if not isinstance(success, bool):
            raise TypeError("sample_successes must contain bool values")
        successes += int(success)
    return successes * 100 >= len(sample_successes) * threshold_pct


def availability_confidence_pct(
    withholding_pct: int = DAS_WITHHOLDING_PCT,
    sample_count: int = DAS_SAMPLES_PER_FRUIT,
) -> float:
    """Return exact at-least-one-withheld-cell detection confidence.

    The calculation is integer-rational until the final float conversion:
    ``100 * (1 - ((100 - withholding_pct) / 100) ** sample_count)``.
    """

    _require_pct("withholding_pct", withholding_pct)
    _require_nonnegative_int("sample_count", sample_count)
    denominator = 100**sample_count
    available_numerator = (100 - withholding_pct) ** sample_count
    detected_numerator = denominator - available_numerator
    return float(detected_numerator * 100 / denominator)


def _commitment_root(
    payload_length: int,
    data_side: int,
    extended_side: int,
    cell_root: bytes,
) -> bytes:
    return domain_hash(
        DOMAIN_DAS_SAMPLE,
        _DAS_COMMITMENT_PREFIX
        + payload_length.to_bytes(U64_BYTES, "little")
        + data_side.to_bytes(U32_BYTES, "little")
        + extended_side.to_bytes(U32_BYTES, "little")
        + require_hash("cell_root", cell_root),
    )


def _cell_records(cells: Sequence[bytes], side: int) -> tuple[bytes, ...]:
    return tuple(
        _cell_record(index // side, index % side, cell) for index, cell in enumerate(cells)
    )


def _cell_record(row: int, column: int, cell: bytes) -> bytes:
    _require_u32("row", row)
    _require_u32("column", column)
    _require_cell("cell", cell)
    return (
        _DAS_CELL_PREFIX
        + row.to_bytes(U32_BYTES, "little")
        + column.to_bytes(U32_BYTES, "little")
        + cell
    )


def _verify_cell_proof(
    *,
    cell: bytes,
    row: int,
    column: int,
    side: int,
    proof: DASMerkleProof,
    cell_root: bytes,
) -> bool:
    expected_index = _cell_index(side, row, column)
    if proof.leaf_index != expected_index or proof.leaf_count != side * side:
        return False
    record = _cell_record(row, column, cell)
    return _verify_ordered_merkle_proof(proof, record, cell_root)


def _ordered_merkle_root(items: Sequence[bytes]) -> bytes:
    return _ordered_merkle_layers(items)[-1][0]


def _ordered_merkle_layers(items: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    if len(items) == 0:
        raise ValueError("items must not be empty")

    level = tuple(_ordered_leaf_hash(index, item) for index, item in enumerate(items))
    layers = [level]
    level_number = 0
    while len(level) > 1:
        next_level = []
        for index in range(0, len(level), 2):
            parent_index = index // 2
            left = level[index]
            right = (
                level[index + 1]
                if index + 1 < len(level)
                else _ordered_empty_hash(level_number, parent_index)
            )
            next_level.append(_ordered_node_hash(level_number, parent_index, left, right))
        level = tuple(next_level)
        layers.append(level)
        level_number += 1
    return tuple(layers)


def _ordered_merkle_proof_from_layers(
    layers: tuple[tuple[bytes, ...], ...],
    leaf_index: int,
) -> DASMerkleProof:
    leaf_count = len(layers[0])
    _require_u32("leaf_index", leaf_index)
    if leaf_index >= leaf_count:
        raise ValueError("leaf_index must be less than leaf_count")

    siblings = []
    position = leaf_index
    for level_number, level in enumerate(layers[:-1]):
        parent_index = position // 2
        if position % 2 == 0:
            sibling_index = position + 1
            sibling_digest = (
                level[sibling_index]
                if sibling_index < len(level)
                else _ordered_empty_hash(level_number, parent_index)
            )
            siblings.append(DASMerkleSibling(is_left=False, digest=sibling_digest))
        else:
            siblings.append(DASMerkleSibling(is_left=True, digest=level[position - 1]))
        position = parent_index
    return DASMerkleProof(
        leaf_index=leaf_index,
        leaf_count=leaf_count,
        siblings=tuple(siblings),
    )


def _verify_ordered_merkle_proof(
    proof: DASMerkleProof,
    item: bytes,
    root: bytes,
) -> bool:
    require_hash("root", root)
    node = _ordered_leaf_hash(proof.leaf_index, item)
    position = proof.leaf_index
    level_width = proof.leaf_count

    for level_number, sibling in enumerate(proof.siblings):
        parent_index = position // 2
        if sibling.is_left:
            if position % 2 == 0:
                return False
            node = _ordered_node_hash(level_number, parent_index, sibling.digest, node)
        else:
            if position % 2 != 0:
                return False
            if position + 1 >= level_width:
                expected_empty = _ordered_empty_hash(level_number, parent_index)
                if sibling.digest != expected_empty:
                    return False
            node = _ordered_node_hash(level_number, parent_index, node, sibling.digest)
        position = parent_index
        level_width = (level_width + 1) // 2

    return position == 0 and level_width == 1 and node == root


def _encode_merkle_proof(proof: DASMerkleProof) -> bytes:
    if not isinstance(proof, DASMerkleProof):
        raise TypeError("proof must be DASMerkleProof")
    body = bytearray()
    body.extend(proof.leaf_index.to_bytes(U32_BYTES, "little"))
    body.extend(proof.leaf_count.to_bytes(U32_BYTES, "little"))
    body.extend(len(proof.siblings).to_bytes(U32_BYTES, "little"))
    for sibling in proof.siblings:
        body.append(1 if sibling.is_left else 0)
        body.extend(sibling.digest)
    return bytes(body)


def _decode_merkle_proof(reader: _Reader) -> DASMerkleProof:
    leaf_index = reader.u32()
    leaf_count = reader.u32()
    sibling_count = reader.u32()
    if sibling_count > _DAS_MAX_MERKLE_SIBLINGS:
        raise ValueError("DAS Merkle proof has too many siblings")
    siblings: list[DASMerkleSibling] = []
    for _ in range(sibling_count):
        side = reader.u8()
        if side not in (0, 1):
            raise ValueError("DAS Merkle sibling side is invalid")
        siblings.append(DASMerkleSibling(is_left=bool(side), digest=reader.bytes(HASH_LEN_BYTES)))
    return DASMerkleProof(
        leaf_index=leaf_index,
        leaf_count=leaf_count,
        siblings=tuple(siblings),
    )


def _ordered_leaf_hash(index: int, item: bytes) -> bytes:
    _require_u32("index", index)
    _require_bytes("item", item)
    if len(item) > U32_MAX:
        raise ValueError("item is too large")
    return domain_hash(
        DOMAIN_DAS_SAMPLE,
        bytes((MERKLE_LEAF_TAG,))
        + index.to_bytes(U32_BYTES, "little")
        + len(item).to_bytes(U32_BYTES, "little")
        + item,
    )


def _ordered_node_hash(level_number: int, parent_index: int, left: bytes, right: bytes) -> bytes:
    _require_u32("parent_index", parent_index)
    require_hash("left", left)
    require_hash("right", right)
    return domain_hash(
        DOMAIN_DAS_SAMPLE,
        bytes((MERKLE_NODE_TAG,))
        + level_number.to_bytes(U16_BYTES, "little")
        + parent_index.to_bytes(U32_BYTES, "little")
        + left
        + right,
    )


def _ordered_empty_hash(level_number: int, parent_index: int) -> bytes:
    _require_u32("parent_index", parent_index)
    return domain_hash(
        DOMAIN_DAS_SAMPLE,
        bytes((MERKLE_EMPTY_TAG,))
        + level_number.to_bytes(U16_BYTES, "little")
        + parent_index.to_bytes(U32_BYTES, "little"),
    )


def _ordered_merkle_depth(leaf_count: int) -> int:
    _require_positive_u32("leaf_count", leaf_count)
    depth = 0
    width = leaf_count
    while width > 1:
        depth += 1
        width = (width + 1) // 2
    return depth


def _uniform_sample_offset(fruit_hash: bytes, sample_index: int, cell_count: int) -> int:
    require_hash("fruit_hash", fruit_hash)
    _require_u32("sample_index", sample_index)
    _require_positive_u32("cell_count", cell_count)

    sample_index_le = sample_index.to_bytes(_SAMPLE_INDEX_BYTES, "little")
    limit = (1 << (HASH_LEN_BYTES * 8)) - ((1 << (HASH_LEN_BYTES * 8)) % cell_count)
    attempt = 0
    while True:
        attempt_suffix = b"" if attempt == 0 else attempt.to_bytes(U32_BYTES, "little")
        digest = domain_hash(DOMAIN_DAS_SAMPLE, fruit_hash + sample_index_le + attempt_suffix)
        value = int.from_bytes(digest, "little")
        if value < limit:
            return value % cell_count
        attempt += 1
        if attempt > U32_MAX:
            raise RuntimeError("DAS sample rejection loop exhausted")


def _rs_extend_matrix(data_rows: tuple[tuple[bytes, ...], ...]) -> tuple[tuple[bytes, ...], ...]:
    data_side = len(data_rows)
    _require_positive_u32("data_side", data_side)
    for row_cells in data_rows:
        if len(row_cells) != data_side:
            raise ValueError("data matrix must be square")
        for cell in row_cells:
            _require_cell("cell", cell)

    row_extended = tuple(_rs_extend_row(row_cells, data_side) for row_cells in data_rows)
    extended_side = data_side * DAS_RS_EXTENSION_FACTOR
    parity_rows = [
        [bytearray(DAS_CELL_BYTES) for _ in range(extended_side)] for _ in range(data_side)
    ]
    for column in range(extended_side):
        column_cells = tuple(row_extended[row][column] for row in range(data_side))
        extended_column = _rs_extend_row(column_cells, data_side)
        for row_offset, cell in enumerate(extended_column[data_side:]):
            parity_rows[row_offset][column][:] = cell

    return (*row_extended, *(tuple(bytes(cell) for cell in row) for row in parity_rows))


def _rs_extend_row(row_cells: tuple[bytes, ...], data_side: int) -> tuple[bytes, ...]:
    if len(row_cells) != data_side:
        raise ValueError("row length must match data_side")
    extended_side = data_side * DAS_RS_EXTENSION_FACTOR
    extended_cells = [bytearray(DAS_CELL_BYTES) for _ in range(extended_side)]
    for byte_index in range(DAS_CELL_BYTES):
        symbols = tuple(cell[byte_index] for cell in row_cells)
        encoded = _rs_encode_symbols(symbols, data_side)
        for column, symbol in enumerate(encoded):
            extended_cells[column][byte_index] = symbol
    return tuple(bytes(cell) for cell in extended_cells)


def _rs_encode_symbols(symbols: Sequence[int], data_side: int) -> tuple[int, ...]:
    _require_rs_data_symbols(symbols, data_side)
    encoded = bytes(_rs_codec(data_side).encode(bytes(symbols)))
    expected_len = data_side * DAS_RS_EXTENSION_FACTOR
    if len(encoded) != expected_len:
        raise RuntimeError("Reed-Solomon codec returned unexpected codeword length")
    return tuple(encoded)


def _is_valid_rs_codeword(cells: Sequence[bytes], data_side: int) -> bool:
    expected_len = data_side * DAS_RS_EXTENSION_FACTOR
    if len(cells) != expected_len:
        return False
    try:
        for cell in cells:
            _require_cell("cell", cell)
        for byte_index in range(DAS_CELL_BYTES):
            symbols = tuple(cell[byte_index] for cell in cells)
            if not _rs_check_symbols(symbols, data_side):
                return False
    except (TypeError, ValueError):
        return False
    return True


def _rs_check_symbols(symbols: Sequence[int], data_side: int) -> bool:
    _require_rs_codeword_symbols(symbols, data_side)
    checks = _rs_codec(data_side).check(bytes(symbols))
    return tuple(checks) == (True,)


@cache
def _rs_codec(data_side: int) -> RSCodec:
    _require_positive_u32("data_side", data_side)
    if data_side * DAS_RS_EXTENSION_FACTOR > 255:
        raise ValueError("DAS Reed-Solomon side exceeds GF(2^8) codeword limit")
    return RSCodec(
        data_side,
        nsize=data_side * DAS_RS_EXTENSION_FACTOR,
        fcr=DAS_RS_FIRST_CONSECUTIVE_ROOT,
        prim=DAS_RS_PRIMITIVE_POLY,
        generator=DAS_RS_GENERATOR,
        c_exp=DAS_RS_FIELD_EXPONENT,
    )


def _grid_has_valid_reed_solomon(cells: Sequence[bytes], data_side: int) -> bool:
    extended_side = data_side * DAS_RS_EXTENSION_FACTOR
    if len(cells) != extended_side * extended_side:
        return False
    for row in range(extended_side):
        row_cells = tuple(
            cells[_cell_index(extended_side, row, column)] for column in range(extended_side)
        )
        if not _is_valid_rs_codeword(row_cells, data_side):
            return False
    for column in range(extended_side):
        column_cells = tuple(
            cells[_cell_index(extended_side, row, column)] for row in range(extended_side)
        )
        if not _is_valid_rs_codeword(column_cells, data_side):
            return False
    return True


def _cell_index(side: int, row: int, column: int) -> int:
    _require_positive_u32("side", side)
    _require_nonnegative_int("row", row)
    _require_nonnegative_int("column", column)
    if row >= side or column >= side:
        raise ValueError("cell coordinates outside DAS grid")
    index = row * side + column
    _require_u32("cell index", index)
    return index


def _ceil_sqrt(value: int) -> int:
    _require_positive_u32("value", value)
    root = math.isqrt(value)
    return root if root * root == value else root + 1


def _ceil_div(numerator: int, denominator: int) -> int:
    _require_nonnegative_int("numerator", numerator)
    _require_positive_u32("denominator", denominator)
    return (numerator + denominator - 1) // denominator


def _require_rs_data_symbols(symbols: Sequence[int], data_side: int) -> None:
    _require_positive_u32("data_side", data_side)
    if len(symbols) != data_side:
        raise ValueError("Reed-Solomon data symbol count must match data_side")
    _require_u8_symbols(symbols)


def _require_rs_codeword_symbols(symbols: Sequence[int], data_side: int) -> None:
    _require_positive_u32("data_side", data_side)
    if len(symbols) != data_side * DAS_RS_EXTENSION_FACTOR:
        raise ValueError("Reed-Solomon codeword length is invalid")
    _require_u8_symbols(symbols)


def _require_u8_symbols(symbols: Sequence[int]) -> None:
    if not isinstance(symbols, Sequence):
        raise TypeError("symbols must be a sequence")
    for symbol in symbols:
        if not isinstance(symbol, int) or isinstance(symbol, bool):
            raise TypeError("Reed-Solomon symbols must be int")
        if not 0 <= symbol <= 0xFF:
            raise ValueError("Reed-Solomon symbols must fit in uint8")


def _require_cell(name: str, value: bytes) -> bytes:
    _require_bytes(name, value)
    if len(value) != DAS_CELL_BYTES:
        raise ValueError(f"{name} must be {DAS_CELL_BYTES} bytes")
    return value


def _require_wire_side(name: str, value: int) -> int:
    _require_positive_u32(name, value)
    if value > _DAS_MAX_EXTENDED_SIDE:
        raise ValueError(f"{name} exceeds DAS extended side limit")
    return value


def _require_bytes(name: str, value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    return value


def _require_pct(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value > 100:
        raise ValueError(f"{name} must be between 0 and 100")
    return value


def _require_positive_u32(name: str, value: int) -> int:
    value = _require_u32(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_u32(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value > U32_MAX:
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


@dataclass(slots=True)
class _Reader:
    data: bytes
    offset: int = 0

    def __post_init__(self) -> None:
        _require_bytes("data", self.data)

    def bytes(self, length: int) -> bytes:
        _require_u32("length", length)
        end = self.offset + length
        if end > len(self.data):
            raise ValueError("DAS wire bytes are truncated")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def u8(self) -> int:
        return self.bytes(1)[0]

    def u32(self) -> int:
        return int.from_bytes(self.bytes(U32_BYTES), "little")

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise ValueError("trailing DAS wire bytes")


__all__ = [
    "DAS_CELL_BYTES",
    "DAS_CONFIDENCE_PCT",
    "DAS_RS_EXTENSION_FACTOR",
    "DAS_RS_FIELD_EXPONENT",
    "DAS_RS_FIRST_CONSECUTIVE_ROOT",
    "DAS_RS_GENERATOR",
    "DAS_RS_PRIMITIVE_POLY",
    "DAS_SAMPLES_PER_FRUIT",
    "DAS_SAMPLE_REQUEST_BYTES",
    "DAS_SAMPLE_SUCCESS_THRESHOLD_PCT",
    "DAS_WITHHOLDING_DETECTION_PCT",
    "DAS_WITHHOLDING_PCT",
    "DOMAIN_DAS_SAMPLE",
    "DASCommitment",
    "DASEncoding",
    "DASMerkleProof",
    "DASMerkleSibling",
    "DASSampleProof",
    "DASSampleRequest",
    "availability_confidence_pct",
    "create_sample_proof",
    "decode_sample_request",
    "decode_sample_response",
    "encode_payload",
    "encode_sample_request",
    "encode_sample_response",
    "is_available",
    "select_sample_request",
    "select_sample_requests",
    "verify_sample",
]
