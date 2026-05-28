"""Tests for PoW verification and nonce search."""

from __future__ import annotations

from threading import Event

import pytest

from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH, U64_MAX, FruitPowHeader
from tensorpow.pow.kernel import target_allows_digest
from tensorpow.pow.miner import mine
from tensorpow.pow.verify import pow_digest_for_header, verify_pow


def _template(nonce: int = 0) -> FruitPowHeader:
    return FruitPowHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        effective_parent_hashes=(bytes([1]) * 32,),
        latest_anchor=GENESIS_PARENT_HASH,
        tx_merkle_root=bytes([2]) * 32,
        timestamp_ms=123,
        shard_id=0,
        nonce=nonce,
    )


def test_verify_pow_accepts_exact_target_and_rejects_one_less() -> None:
    header = _template()
    digest = pow_digest_for_header(header, backend="cpu")
    target = int.from_bytes(digest, "little").to_bytes(HASH_LEN_BYTES, "little")
    too_hard = (int.from_bytes(digest, "little") - 1).to_bytes(HASH_LEN_BYTES, "little")

    assert verify_pow(header, target, backend="cpu")
    assert not verify_pow(header, too_hard, backend="cpu")


def test_mine_finds_low_difficulty_nonce_zero() -> None:
    result = mine(_template(), bytes([0xFF]) * HASH_LEN_BYTES, Event(), backend="cpu", max_nonce=0)

    assert result is not None
    assert result.nonce == 0
    assert result.attempts == 1
    assert result.backend == "cpu"
    assert target_allows_digest(result.digest, bytes([0xFF]) * HASH_LEN_BYTES)


def test_mine_honors_pre_set_stop_event(monkeypatch: pytest.MonkeyPatch) -> None:
    stop_event = Event()
    stop_event.set()

    def fail_if_called(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("miner should not compute when already stopped")

    monkeypatch.setattr("tensorpow.pow.miner.pow_digest_for_header", fail_if_called)

    assert mine(_template(), bytes([0xFF]) * HASH_LEN_BYTES, stop_event, backend="cpu") is None


def test_mine_honors_stop_event_after_expensive_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    stop_event = Event()

    def stop_during_attempt(*args: object, **kwargs: object) -> bytes:
        stop_event.set()
        return bytes(HASH_LEN_BYTES)

    monkeypatch.setattr("tensorpow.pow.miner.pow_digest_for_header", stop_during_attempt)

    assert mine(_template(), bytes([0xFF]) * HASH_LEN_BYTES, stop_event, backend="cpu") is None


def test_mine_rejects_bad_bounds_and_does_not_wrap_uint64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError):
        mine(_template(), bytes(HASH_LEN_BYTES), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mine(_template(), bytes(HASH_LEN_BYTES), Event(), start_nonce=-1)
    with pytest.raises(ValueError):
        mine(_template(), bytes(HASH_LEN_BYTES), Event(), start_nonce=2, max_nonce=1)

    def failing_digest(*args: object, **kwargs: object) -> bytes:
        return b"\x01" + bytes(HASH_LEN_BYTES - 1)

    monkeypatch.setattr("tensorpow.pow.miner.pow_digest_for_header", failing_digest)

    assert (
        mine(
            _template(U64_MAX),
            bytes(HASH_LEN_BYTES),
            Event(),
            backend="cpu",
            start_nonce=U64_MAX,
            max_nonce=U64_MAX,
        )
        is None
    )
