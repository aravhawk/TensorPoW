"""Peer admission control for expensive consensus checks.

Proof-of-work verification costs the verifier a full INT8 matmul while costing the
sender nothing for a junk block. This module keeps that work bounded:

* a per-peer token bucket limits how many PoW verifications one peer can trigger,
* a peer that sends an invalid proof is banned for ``ban_seconds``,
* repeated malformed payloads accumulate strikes and eventually ban the peer,
* a bounded slot pool caps concurrent verifications node-wide; callers that cannot
  get a slot are rejected immediately instead of queueing without bound.

All state is bounded (LRU over ``max_tracked_peers``) and local; nothing here
affects consensus validity.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock
from typing import Final

DEFAULT_POW_VERIFY_RATE_PER_SECOND: Final[float] = 10.0
DEFAULT_POW_VERIFY_BURST: Final[int] = 20
DEFAULT_BAN_SECONDS: Final[float] = 600.0
DEFAULT_MALFORMED_STRIKE_LIMIT: Final[int] = 8
DEFAULT_MAX_TRACKED_PEERS: Final[int] = 4096
DEFAULT_MAX_CONCURRENT_POW_VERIFICATIONS: Final[int] = 2

REASON_PEER_BANNED: Final[str] = "peer_banned"
REASON_PEER_RATE_LIMITED: Final[str] = "peer_rate_limited"
REASON_POW_VERIFIER_BUSY: Final[str] = "pow_verifier_busy"


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    """Local anti-DoS limits for inbound consensus objects."""

    pow_verify_rate_per_second: float = DEFAULT_POW_VERIFY_RATE_PER_SECOND
    pow_verify_burst: int = DEFAULT_POW_VERIFY_BURST
    ban_seconds: float = DEFAULT_BAN_SECONDS
    malformed_strike_limit: int = DEFAULT_MALFORMED_STRIKE_LIMIT
    max_tracked_peers: int = DEFAULT_MAX_TRACKED_PEERS
    max_concurrent_pow_verifications: int = DEFAULT_MAX_CONCURRENT_POW_VERIFICATIONS

    def __post_init__(self) -> None:
        if not self.pow_verify_rate_per_second > 0:
            raise ValueError("pow_verify_rate_per_second must be positive")
        for name in (
            "pow_verify_burst",
            "malformed_strike_limit",
            "max_tracked_peers",
            "max_concurrent_pow_verifications",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if not self.ban_seconds >= 0:
            raise ValueError("ban_seconds must be non-negative")


@dataclass(slots=True)
class _PeerState:
    tokens: float
    updated_at: float
    banned_until: float = 0.0
    strikes: int = 0


class PeerAdmission:
    """Thread-safe per-peer rate limiting, banning and PoW slot accounting."""

    def __init__(
        self,
        config: AdmissionConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = AdmissionConfig() if config is None else config
        self._clock = clock
        self._lock = Lock()
        self._peers: OrderedDict[str, _PeerState] = OrderedDict()
        self._pow_slots = BoundedSemaphore(self.config.max_concurrent_pow_verifications)

    def check_peer(self, peer_id: str | None) -> str | None:
        """Return a rejection reason if ``peer_id`` is currently banned."""

        if peer_id is None:
            return None
        with self._lock:
            state = self._peers.get(peer_id)
            if state is not None and state.banned_until > self._clock():
                return REASON_PEER_BANNED
        return None

    def consume_pow_budget(self, peer_id: str | None) -> str | None:
        """Charge one PoW verification to ``peer_id`` or return a rejection reason."""

        if peer_id is None:
            return None
        with self._lock:
            now = self._clock()
            state = self._state(peer_id, now)
            if state.banned_until > now:
                return REASON_PEER_BANNED
            elapsed = max(0.0, now - state.updated_at)
            state.tokens = min(
                float(self.config.pow_verify_burst),
                state.tokens + elapsed * self.config.pow_verify_rate_per_second,
            )
            state.updated_at = now
            if state.tokens < 1.0:
                return REASON_PEER_RATE_LIMITED
            state.tokens -= 1.0
        return None

    def ban(self, peer_id: str | None) -> None:
        """Ban ``peer_id`` (for example after an invalid proof of work)."""

        if peer_id is None:
            return
        with self._lock:
            now = self._clock()
            state = self._state(peer_id, now)
            state.banned_until = now + self.config.ban_seconds
            state.strikes = 0

    def strike(self, peer_id: str | None) -> None:
        """Record one malformed payload; ban after ``malformed_strike_limit`` strikes."""

        if peer_id is None:
            return
        with self._lock:
            now = self._clock()
            state = self._state(peer_id, now)
            state.strikes += 1
            if state.strikes >= self.config.malformed_strike_limit:
                state.banned_until = now + self.config.ban_seconds
                state.strikes = 0

    def is_banned(self, peer_id: str) -> bool:
        return self.check_peer(peer_id) == REASON_PEER_BANNED

    @contextmanager
    def pow_slot(self, *, blocking: bool = False) -> Iterator[bool]:
        """Yield whether a node-wide PoW verification slot was acquired.

        Remote submissions use ``blocking=False`` so excess load is shed immediately;
        trusted local submissions (no peer id) wait for a slot.
        """

        acquired = self._pow_slots.acquire(blocking=blocking)
        try:
            yield acquired
        finally:
            if acquired:
                self._pow_slots.release()

    def _state(self, peer_id: str, now: float) -> _PeerState:
        if not isinstance(peer_id, str) or peer_id == "":
            raise ValueError("peer_id must be a non-empty string")
        state = self._peers.get(peer_id)
        if state is None:
            state = _PeerState(tokens=float(self.config.pow_verify_burst), updated_at=now)
            self._peers[peer_id] = state
            self._evict(now)
        else:
            self._peers.move_to_end(peer_id)
        return state

    def _evict(self, now: float) -> None:
        while len(self._peers) > self.config.max_tracked_peers:
            # Prefer evicting an unbanned peer so cycling ids cannot flush active bans.
            for candidate, state in self._peers.items():
                if state.banned_until <= now:
                    del self._peers[candidate]
                    break
            else:
                self._peers.popitem(last=False)


__all__ = [
    "DEFAULT_BAN_SECONDS",
    "DEFAULT_MALFORMED_STRIKE_LIMIT",
    "DEFAULT_MAX_CONCURRENT_POW_VERIFICATIONS",
    "DEFAULT_MAX_TRACKED_PEERS",
    "DEFAULT_POW_VERIFY_BURST",
    "DEFAULT_POW_VERIFY_RATE_PER_SECOND",
    "REASON_PEER_BANNED",
    "REASON_PEER_RATE_LIMITED",
    "REASON_POW_VERIFIER_BUSY",
    "AdmissionConfig",
    "PeerAdmission",
]
