"""Nonce search for TensorPoW headers."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import monotonic

from tensorpow.pow.challenge import U64_MAX, PowHeader, with_nonce
from tensorpow.pow.kernel import Backend, resolve_backend, target_allows_digest
from tensorpow.pow.verify import pow_digest_for_header


@dataclass(frozen=True, slots=True)
class FoundNonce:
    """Successful nonce search result."""

    header: PowHeader
    nonce: int
    digest: bytes
    attempts: int
    elapsed_seconds: float
    backend: str


def mine(
    template: PowHeader,
    target: bytes,
    stop_event: Event,
    *,
    backend: Backend = "auto",
    start_nonce: int = 0,
    max_nonce: int | None = None,
) -> FoundNonce | None:
    """Search nonces until target is met, stop_event is set, or max_nonce is reached."""

    if not isinstance(stop_event, Event):
        raise TypeError("stop_event must be threading.Event")
    if not isinstance(start_nonce, int):
        raise TypeError("start_nonce must be int")
    if not 0 <= start_nonce <= U64_MAX:
        raise ValueError("start_nonce outside uint64 range")
    if max_nonce is not None:
        if not isinstance(max_nonce, int):
            raise TypeError("max_nonce must be int or None")
        if not start_nonce <= max_nonce <= U64_MAX:
            raise ValueError("max_nonce outside uint64 range or before start_nonce")

    nonce = start_nonce
    attempts = 0
    started = monotonic()
    resolved_backend = resolve_backend(backend)
    while max_nonce is None or nonce <= max_nonce:
        if stop_event.is_set():
            return None

        header = with_nonce(template, nonce)
        digest = pow_digest_for_header(header, backend=resolved_backend)
        attempts += 1
        if stop_event.is_set():
            return None
        if target_allows_digest(digest, target):
            return FoundNonce(
                header=header,
                nonce=nonce,
                digest=digest,
                attempts=attempts,
                elapsed_seconds=monotonic() - started,
                backend=resolved_backend,
            )
        if nonce == U64_MAX:
            return None
        nonce += 1

    return None
