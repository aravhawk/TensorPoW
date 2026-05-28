"""Deterministic Turbine-style relay tree and erasure coding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Final

from blake3 import blake3
from reedsolo import ReedSolomonError, RSCodec  # type: ignore[import-untyped]

from tensorpow.chain.merkle import require_hash

TURBINE_SIM_NODE_COUNT: Final[int] = 1000
TURBINE_MAX_PROPAGATION_MS: Final[int] = 200
TURBINE_DROPOUT_PCT: Final[int] = 30
TURBINE_DATA_SHARDS: Final[int] = 16
TURBINE_PARITY_SHARDS: Final[int] = 8
TURBINE_MIN_FANOUT: Final[int] = 2
TURBINE_MAX_FANOUT: Final[int] = 16
TURBINE_CHUNK_REPLICATION: Final[int] = 8

U32_MAX: Final[int] = 0xFFFFFFFF
_RS_MAX_SHARDS: Final[int] = 255
_CUSTODIAN_DOMAIN: Final[bytes] = b"TensorPoW Turbine custodian"
_TREE_DOMAIN: Final[bytes] = b"TensorPoW Turbine tree"


@dataclass(frozen=True, slots=True)
class TurbineChunk:
    """One Reed-Solomon encoded fruit payload shard."""

    index: int
    data_shard_count: int
    parity_shard_count: int
    payload_length: int
    data: bytes

    def __post_init__(self) -> None:
        _require_nonnegative_int("index", self.index)
        _require_positive_int("data_shard_count", self.data_shard_count)
        _require_positive_int("parity_shard_count", self.parity_shard_count)
        _require_u32("payload_length", self.payload_length)
        if self.total_shards > _RS_MAX_SHARDS:
            raise ValueError("total shard count exceeds Reed-Solomon limit")
        if self.index >= self.total_shards:
            raise ValueError("chunk index outside shard set")
        _require_bytes("data", self.data)
        if len(self.data) == 0:
            raise ValueError("chunk data must not be empty")
        expected_shard_size = _expected_shard_size(
            self.payload_length,
            self.data_shard_count,
        )
        if len(self.data) != expected_shard_size:
            raise ValueError("chunk data length does not match payload and shard count")

    @property
    def total_shards(self) -> int:
        """Return data plus parity shard count."""

        return self.data_shard_count + self.parity_shard_count


@dataclass(frozen=True, slots=True)
class TurbinePeer:
    """Relay peer metadata used for deterministic adaptive fanout."""

    peer_id: bytes
    latency_ms: int
    bandwidth_bytes_per_ms: int

    def __post_init__(self) -> None:
        _require_peer_id(self.peer_id)
        _require_positive_int("latency_ms", self.latency_ms)
        _require_positive_int("bandwidth_bytes_per_ms", self.bandwidth_bytes_per_ms)


@dataclass(frozen=True, slots=True)
class RelayAssignment:
    """Children assigned to one relay parent."""

    parent_peer_id: bytes
    child_peer_ids: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_peer_id(self.parent_peer_id)
        if not isinstance(self.child_peer_ids, tuple):
            raise TypeError("child_peer_ids must be a tuple")
        if len(set(self.child_peer_ids)) != len(self.child_peer_ids):
            raise ValueError("child_peer_ids must be unique")
        for peer_id in self.child_peer_ids:
            _require_peer_id(peer_id)


@dataclass(frozen=True, slots=True)
class TurbineRelayTree:
    """Seed-derived relay tree."""

    root_peer_id: bytes
    assignments: tuple[RelayAssignment, ...]

    def __post_init__(self) -> None:
        _require_peer_id(self.root_peer_id)
        if not isinstance(self.assignments, tuple):
            raise TypeError("assignments must be a tuple")
        seen_children: set[bytes] = set()
        for assignment in self.assignments:
            if not isinstance(assignment, RelayAssignment):
                raise TypeError("assignments must contain RelayAssignment values")
            for child_peer_id in assignment.child_peer_ids:
                if child_peer_id in seen_children:
                    raise ValueError("peer appears under more than one parent")
                seen_children.add(child_peer_id)

    def child_count(self, peer_id: bytes) -> int:
        """Return the number of relay children assigned to ``peer_id``."""

        _require_peer_id(peer_id)
        for assignment in self.assignments:
            if assignment.parent_peer_id == peer_id:
                return len(assignment.child_peer_ids)
        return 0

    def children_by_parent(self) -> dict[bytes, tuple[bytes, ...]]:
        """Return assignment lookup keyed by parent peer id."""

        return {
            assignment.parent_peer_id: assignment.child_peer_ids for assignment in self.assignments
        }


@dataclass(frozen=True, slots=True)
class TurbineSimulationResult:
    """Deterministic relay simulation outcome."""

    latency_ms: int
    reached_peer_count: int
    available_chunk_count: int
    reconstructed_payload: bytes
    tree: TurbineRelayTree

    def __post_init__(self) -> None:
        _require_nonnegative_int("latency_ms", self.latency_ms)
        _require_nonnegative_int("reached_peer_count", self.reached_peer_count)
        _require_nonnegative_int("available_chunk_count", self.available_chunk_count)
        _require_bytes("reconstructed_payload", self.reconstructed_payload)
        if not isinstance(self.tree, TurbineRelayTree):
            raise TypeError("tree must be TurbineRelayTree")


def encode_payload(
    payload: bytes,
    *,
    data_shard_count: int = TURBINE_DATA_SHARDS,
    parity_shard_count: int = TURBINE_PARITY_SHARDS,
) -> tuple[TurbineChunk, ...]:
    """Encode fruit payload bytes into deterministic Reed-Solomon shards."""

    _require_bytes("payload", payload)
    _require_u32("payload length", len(payload))
    _require_shard_counts(data_shard_count, parity_shard_count)
    shard_size = max(1, _ceil_div(len(payload), data_shard_count))
    padded = payload.ljust(shard_size * data_shard_count, b"\x00")
    data_shards = tuple(
        padded[index * shard_size : (index + 1) * shard_size] for index in range(data_shard_count)
    )
    total_shards = data_shard_count + parity_shard_count
    encoded_shards = [bytearray(shard_size) for _ in range(total_shards)]
    codec = _rs_codec(data_shard_count, parity_shard_count)

    for offset in range(shard_size):
        symbols = bytes(shard[offset] for shard in data_shards)
        encoded_symbols = bytes(codec.encode(symbols))
        if len(encoded_symbols) != total_shards:
            raise RuntimeError("Reed-Solomon codec returned unexpected shard count")
        for shard_index, symbol in enumerate(encoded_symbols):
            encoded_shards[shard_index][offset] = symbol

    return tuple(
        TurbineChunk(
            index=index,
            data_shard_count=data_shard_count,
            parity_shard_count=parity_shard_count,
            payload_length=len(payload),
            data=bytes(shard),
        )
        for index, shard in enumerate(encoded_shards)
    )


def decode_payload(chunks: Sequence[TurbineChunk]) -> bytes:
    """Recover a payload from any valid quorum of Turbine chunks."""

    chunk_tuple = _canonical_chunks(chunks)
    first = chunk_tuple[0]
    if len(chunk_tuple) < first.data_shard_count:
        raise ValueError("not enough chunks to recover payload")

    present = {chunk.index: chunk for chunk in chunk_tuple}
    missing_indexes = tuple(index for index in range(first.total_shards) if index not in present)
    if len(missing_indexes) > first.parity_shard_count:
        raise ValueError("too many missing chunks to recover payload")

    shard_size = len(first.data)
    recovered_data = [bytearray(shard_size) for _ in range(first.data_shard_count)]
    codec = _rs_codec(first.data_shard_count, first.parity_shard_count)
    for offset in range(shard_size):
        codeword = bytearray(first.total_shards)
        erase_pos: list[int] = []
        for shard_index in range(first.total_shards):
            chunk = present.get(shard_index)
            if chunk is None:
                erase_pos.append(shard_index)
            else:
                codeword[shard_index] = chunk.data[offset]
        try:
            decoded_symbols = bytes(codec.decode(bytes(codeword), erase_pos=erase_pos)[0])
        except ReedSolomonError as exc:
            raise ValueError("invalid Reed-Solomon chunk set") from exc
        for shard_index, symbol in enumerate(decoded_symbols):
            recovered_data[shard_index][offset] = symbol

    payload = b"".join(bytes(shard) for shard in recovered_data)[: first.payload_length]
    canonical_chunks = encode_payload(
        payload,
        data_shard_count=first.data_shard_count,
        parity_shard_count=first.parity_shard_count,
    )
    for chunk in chunk_tuple:
        if canonical_chunks[chunk.index].data != chunk.data:
            raise ValueError("chunk data does not match recovered payload")
    return payload


def derive_relay_tree(
    anchor_seed: bytes,
    peers: Sequence[TurbinePeer],
    *,
    max_fanout: int = TURBINE_MAX_FANOUT,
) -> TurbineRelayTree:
    """Build a deterministic adaptive relay tree from an anchor seed."""

    require_hash("anchor_seed", anchor_seed)
    peer_tuple = _canonical_peers(peers)
    _require_positive_int("max_fanout", max_fanout)
    if max_fanout < TURBINE_MIN_FANOUT:
        raise ValueError("max_fanout must be at least TURBINE_MIN_FANOUT")

    ordered = sorted(
        peer_tuple,
        key=lambda peer: (
            -_relay_score(peer),
            blake3(_TREE_DOMAIN + anchor_seed + peer.peer_id).digest(),
            peer.peer_id,
        ),
    )
    root = ordered[0]
    queue = [root]
    remaining = list(ordered[1:])
    assignments: list[RelayAssignment] = []
    while remaining:
        parent = queue.pop(0)
        fanout = min(_adaptive_fanout(parent, max_fanout), len(remaining))
        children = tuple(remaining[:fanout])
        del remaining[:fanout]
        assignments.append(
            RelayAssignment(
                parent_peer_id=parent.peer_id,
                child_peer_ids=tuple(child.peer_id for child in children),
            )
        )
        queue.extend(children)
    return TurbineRelayTree(root_peer_id=root.peer_id, assignments=tuple(assignments))


def select_dropout_peers(
    anchor_seed: bytes,
    peers: Sequence[TurbinePeer],
    dropout_pct: int = TURBINE_DROPOUT_PCT,
    *,
    protected_peer_ids: Sequence[bytes] = (),
) -> tuple[bytes, ...]:
    """Select deterministic dropped peers for relay simulation."""

    require_hash("anchor_seed", anchor_seed)
    peer_tuple = _canonical_peers(peers)
    _require_pct("dropout_pct", dropout_pct)
    protected = frozenset(protected_peer_ids)
    for peer_id in protected:
        _require_peer_id(peer_id)
    candidate_peers = tuple(peer for peer in peer_tuple if peer.peer_id not in protected)
    dropout_count = len(peer_tuple) * dropout_pct // 100
    ranked = sorted(
        candidate_peers,
        key=lambda peer: blake3(anchor_seed + peer.peer_id + b"dropout").digest(),
    )
    return tuple(sorted(peer.peer_id for peer in ranked[:dropout_count]))


def simulate_propagation(
    anchor_seed: bytes,
    peers: Sequence[TurbinePeer],
    payload: bytes,
    *,
    dropout_peer_ids: Sequence[bytes] = (),
) -> TurbineSimulationResult:
    """Simulate Turbine propagation and erasure recovery over a relay tree."""

    require_hash("anchor_seed", anchor_seed)
    _require_bytes("payload", payload)
    peer_tuple = _canonical_peers(peers)
    peer_by_id = {peer.peer_id: peer for peer in peer_tuple}
    dropout = frozenset(dropout_peer_ids)
    for peer_id in dropout:
        _require_peer_id(peer_id)

    chunks = encode_payload(payload)
    tree = derive_relay_tree(anchor_seed, peer_tuple)
    arrival_by_peer = _arrival_times(tree, peer_by_id, len(chunks[0].data), dropout)
    live_reached = frozenset(arrival_by_peer)
    available_chunks = tuple(
        chunk
        for chunk in chunks
        if any(
            custodian in live_reached
            for custodian in _chunk_custodians(anchor_seed, peer_tuple, chunk.index)
        )
    )
    reconstructed = decode_payload(available_chunks)
    return TurbineSimulationResult(
        latency_ms=max(arrival_by_peer.values(), default=0),
        reached_peer_count=len(arrival_by_peer),
        available_chunk_count=len(available_chunks),
        reconstructed_payload=reconstructed,
        tree=tree,
    )


def _arrival_times(
    tree: TurbineRelayTree,
    peer_by_id: dict[bytes, TurbinePeer],
    chunk_size: int,
    dropout: frozenset[bytes],
) -> dict[bytes, int]:
    if tree.root_peer_id in dropout:
        return {}
    children_by_parent = tree.children_by_parent()
    arrival_by_peer = {tree.root_peer_id: 0}
    queue = [tree.root_peer_id]
    while queue:
        parent_id = queue.pop(0)
        parent = peer_by_id[parent_id]
        parent_arrival = arrival_by_peer[parent_id]
        for child_id in children_by_parent.get(parent_id, ()):
            if child_id in dropout:
                continue
            child = peer_by_id[child_id]
            edge_ms = _edge_latency_ms(parent, child, chunk_size)
            arrival_by_peer[child_id] = parent_arrival + edge_ms
            queue.append(child_id)
    return arrival_by_peer


def _chunk_custodians(
    anchor_seed: bytes,
    peers: tuple[TurbinePeer, ...],
    chunk_index: int,
) -> tuple[bytes, ...]:
    ranked = sorted(
        peers,
        key=lambda peer: blake3(
            _CUSTODIAN_DOMAIN + anchor_seed + chunk_index.to_bytes(4, "little") + peer.peer_id
        ).digest(),
    )
    return tuple(peer.peer_id for peer in ranked[:TURBINE_CHUNK_REPLICATION])


def _canonical_chunks(chunks: Sequence[TurbineChunk]) -> tuple[TurbineChunk, ...]:
    if not isinstance(chunks, Sequence):
        raise TypeError("chunks must be a sequence")
    chunk_tuple = tuple(chunks)
    if len(chunk_tuple) == 0:
        raise ValueError("chunks must not be empty")
    first = chunk_tuple[0]
    if not isinstance(first, TurbineChunk):
        raise TypeError("chunks must contain TurbineChunk values")
    indexes: set[int] = set()
    for chunk in chunk_tuple:
        if not isinstance(chunk, TurbineChunk):
            raise TypeError("chunks must contain TurbineChunk values")
        if chunk.index in indexes:
            raise ValueError("duplicate chunk index")
        indexes.add(chunk.index)
        if chunk.data_shard_count != first.data_shard_count:
            raise ValueError("inconsistent data shard count")
        if chunk.parity_shard_count != first.parity_shard_count:
            raise ValueError("inconsistent parity shard count")
        if chunk.payload_length != first.payload_length:
            raise ValueError("inconsistent payload length")
        if len(chunk.data) != len(first.data):
            raise ValueError("inconsistent chunk data length")
    return tuple(sorted(chunk_tuple, key=lambda chunk: chunk.index))


def _canonical_peers(peers: Sequence[TurbinePeer]) -> tuple[TurbinePeer, ...]:
    if not isinstance(peers, Sequence):
        raise TypeError("peers must be a sequence")
    peer_tuple = tuple(peers)
    if len(peer_tuple) == 0:
        raise ValueError("peers must not be empty")
    peer_ids: set[bytes] = set()
    for peer in peer_tuple:
        if not isinstance(peer, TurbinePeer):
            raise TypeError("peers must contain TurbinePeer values")
        if peer.peer_id in peer_ids:
            raise ValueError("peer ids must be unique")
        peer_ids.add(peer.peer_id)
    return peer_tuple


def _adaptive_fanout(peer: TurbinePeer, max_fanout: int) -> int:
    score = _relay_score(peer)
    if score >= 4_000_000:
        fanout = 16
    elif score >= 2_000_000:
        fanout = 12
    elif score >= 1_000_000:
        fanout = 8
    elif score >= 500_000:
        fanout = 4
    else:
        fanout = TURBINE_MIN_FANOUT
    return min(max_fanout, fanout)


def _relay_score(peer: TurbinePeer) -> int:
    return peer.bandwidth_bytes_per_ms * 1000 // peer.latency_ms


def _edge_latency_ms(parent: TurbinePeer, child: TurbinePeer, chunk_size: int) -> int:
    transmission_ms = _ceil_div(
        chunk_size,
        min(parent.bandwidth_bytes_per_ms, child.bandwidth_bytes_per_ms),
    )
    return max(1, (parent.latency_ms + child.latency_ms) // 2 + transmission_ms)


@cache
def _rs_codec(data_shard_count: int, parity_shard_count: int) -> RSCodec:
    _require_shard_counts(data_shard_count, parity_shard_count)
    return RSCodec(parity_shard_count, nsize=data_shard_count + parity_shard_count)


def _require_shard_counts(data_shard_count: int, parity_shard_count: int) -> None:
    _require_positive_int("data_shard_count", data_shard_count)
    _require_positive_int("parity_shard_count", parity_shard_count)
    if data_shard_count + parity_shard_count > _RS_MAX_SHARDS:
        raise ValueError("total shard count exceeds Reed-Solomon limit")


def _require_peer_id(peer_id: bytes) -> None:
    _require_bytes("peer_id", peer_id)
    if len(peer_id) == 0:
        raise ValueError("peer_id must not be empty")


def _require_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")


def _require_pct(name: str, value: int) -> None:
    _require_nonnegative_int(name, value)
    if value > 100:
        raise ValueError(f"{name} must be between 0 and 100")


def _require_u32(name: str, value: int) -> None:
    _require_nonnegative_int(name, value)
    if value > U32_MAX:
        raise ValueError(f"{name} must fit in uint32")


def _require_positive_int(name: str, value: int) -> None:
    _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _ceil_div(numerator: int, denominator: int) -> int:
    _require_nonnegative_int("numerator", numerator)
    _require_positive_int("denominator", denominator)
    return (numerator + denominator - 1) // denominator


def _expected_shard_size(payload_length: int, data_shard_count: int) -> int:
    _require_u32("payload_length", payload_length)
    _require_positive_int("data_shard_count", data_shard_count)
    return max(1, _ceil_div(payload_length, data_shard_count))


__all__ = [
    "TURBINE_DATA_SHARDS",
    "TURBINE_DROPOUT_PCT",
    "TURBINE_MAX_FANOUT",
    "TURBINE_MAX_PROPAGATION_MS",
    "TURBINE_MIN_FANOUT",
    "TURBINE_PARITY_SHARDS",
    "TURBINE_SIM_NODE_COUNT",
    "RelayAssignment",
    "TurbineChunk",
    "TurbinePeer",
    "TurbineRelayTree",
    "TurbineSimulationResult",
    "decode_payload",
    "derive_relay_tree",
    "encode_payload",
    "select_dropout_peers",
    "simulate_propagation",
]
