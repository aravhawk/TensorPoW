"""Tests for Turbine-style relay and erasure coding."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tensorpow.crypto.hash import hash_bytes
from tensorpow.net.turbine import (
    TURBINE_DATA_SHARDS,
    TURBINE_DROPOUT_PCT,
    TURBINE_MAX_PROPAGATION_MS,
    TURBINE_PARITY_SHARDS,
    TURBINE_SIM_NODE_COUNT,
    TurbineChunk,
    TurbinePeer,
    decode_payload,
    derive_relay_tree,
    encode_payload,
    select_dropout_peers,
    simulate_propagation,
)


def test_reed_solomon_chunks_round_trip_with_parity_loss() -> None:
    payload = _payload(4097)

    chunks = encode_payload(payload)
    recovered = decode_payload(chunks[:8] + chunks[12:20])

    assert len(chunks) == TURBINE_DATA_SHARDS + TURBINE_PARITY_SHARDS
    assert {chunk.index for chunk in chunks} == set(range(24))
    assert recovered == payload


def test_reed_solomon_rejects_malformed_or_corrupt_chunks() -> None:
    payload = _payload(1024)
    chunks = encode_payload(payload)

    with pytest.raises(ValueError, match="not enough chunks"):
        decode_payload(chunks[: TURBINE_DATA_SHARDS - 1])
    with pytest.raises(ValueError, match="duplicate"):
        decode_payload((chunks[0], chunks[0], *chunks[1:TURBINE_DATA_SHARDS]))
    with pytest.raises(ValueError, match="inconsistent payload"):
        decode_payload((replace(chunks[0], payload_length=1023), *chunks[1:TURBINE_DATA_SHARDS]))

    corrupt = bytearray(chunks[0].data)
    corrupt[0] ^= 0x01
    bad_chunk = replace(chunks[0], data=bytes(corrupt))
    with pytest.raises(ValueError, match="does not match recovered payload"):
        decode_payload((bad_chunk, *chunks[1:]))

    with pytest.raises(ValueError, match="chunk data length"):
        replace(chunks[0], data=chunks[0].data + b"\x00")

    with pytest.raises(ValueError, match="outside shard set"):
        TurbineChunk(
            index=24,
            data_shard_count=TURBINE_DATA_SHARDS,
            parity_shard_count=TURBINE_PARITY_SHARDS,
            payload_length=1,
            data=b"x",
        )


def test_relay_tree_is_deterministic_and_adaptive() -> None:
    seed = hash_bytes(b"turbine-tree")
    peers = _peers(64)

    first = derive_relay_tree(seed, peers)
    second = derive_relay_tree(seed, tuple(reversed(peers)))

    fast_peer = peers[0]
    slow_peer = peers[-1]
    assert first == second
    assert first.root_peer_id == fast_peer.peer_id
    assert first.child_count(fast_peer.peer_id) > first.child_count(slow_peer.peer_id)


def test_simulated_1000_node_network_stays_under_latency_target() -> None:
    seed = hash_bytes(b"turbine-sim-latency")
    peers = _peers(TURBINE_SIM_NODE_COUNT)
    payload = _payload(8192)

    result = simulate_propagation(seed, peers, payload)

    assert result.reached_peer_count == TURBINE_SIM_NODE_COUNT
    assert result.available_chunk_count >= TURBINE_DATA_SHARDS
    assert result.reconstructed_payload == payload
    assert result.latency_ms <= TURBINE_MAX_PROPAGATION_MS


def test_30_percent_dropout_still_recovers_payload() -> None:
    seed = hash_bytes(b"turbine-sim-dropout")
    peers = _peers(TURBINE_SIM_NODE_COUNT)
    tree = derive_relay_tree(seed, peers)
    dropped = select_dropout_peers(
        seed,
        peers,
        TURBINE_DROPOUT_PCT,
        protected_peer_ids=(tree.root_peer_id,),
    )
    payload = _payload(8192)

    result = simulate_propagation(seed, peers, payload, dropout_peer_ids=dropped)

    assert len(dropped) == (TURBINE_SIM_NODE_COUNT - 1) * TURBINE_DROPOUT_PCT // 100
    assert result.reached_peer_count > 0
    assert result.available_chunk_count >= TURBINE_DATA_SHARDS
    assert result.reconstructed_payload == payload
    assert result.latency_ms <= TURBINE_MAX_PROPAGATION_MS


def test_dropout_count_uses_unprotected_candidate_pool() -> None:
    seed = hash_bytes(b"turbine-dropout-protected")
    peers = _peers(10)
    protected = tuple(peer.peer_id for peer in peers[:8])

    dropped = select_dropout_peers(
        seed,
        peers,
        dropout_pct=50,
        protected_peer_ids=protected,
    )

    assert len(dropped) == 1
    assert set(dropped).isdisjoint(protected)


def _peers(count: int) -> tuple[TurbinePeer, ...]:
    peers = []
    for index in range(count):
        peer_id = hash_bytes(b"peer" + index.to_bytes(4, "little"))
        latency_ms = 4 + (index % 20)
        bandwidth_bytes_per_ms = 80_000 - (index % 11) * 1_000
        peers.append(
            TurbinePeer(
                peer_id=peer_id,
                latency_ms=latency_ms,
                bandwidth_bytes_per_ms=bandwidth_bytes_per_ms,
            )
        )
    return tuple(peers)


def _payload(length: int) -> bytes:
    return bytes((index * 17 + 23) % 251 for index in range(length))
