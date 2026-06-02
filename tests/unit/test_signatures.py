"""Tests for TensorPoW Ed25519 signature helpers."""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidSignature

from tensorpow.crypto import signatures
from tensorpow.crypto.signatures import (
    ED25519_PRIVATE_KEY_BYTES,
    ED25519_PUBLIC_KEY_BYTES,
    ED25519_SIGNATURE_BYTES,
    SIG_TYPE_ED25519,
    SIG_TYPE_ML_DSA_RESERVED,
    Keypair,
    sign,
    verify,
    verify_batch,
    verify_by_sig_type,
)

RFC8032_TEST1_PRIVATE_KEY = bytes.fromhex(
    "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
)
RFC8032_TEST1_PUBLIC_KEY = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
)
RFC8032_TEST1_SIGNATURE = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a"
    "84877f1eb8e5d974d873e06522490155"
    "5fb8821590a33bacc61e39701cf9b46b"
    "d25bf5f0595bbe24655141438e7a100b"
)


def _corrupt(value: bytes) -> bytes:
    return bytes((value[0] ^ 0x01,)) + value[1:]


def test_rfc8032_known_answer_vector_signs_and_verifies() -> None:
    assert sign(b"", RFC8032_TEST1_PRIVATE_KEY) == RFC8032_TEST1_SIGNATURE
    assert verify(b"", RFC8032_TEST1_SIGNATURE, RFC8032_TEST1_PUBLIC_KEY)


def test_keypair_generate_returns_raw_key_bytes_and_round_trips() -> None:
    keypair = Keypair.generate()
    message = b"tensorpow signature smoke"
    signature = sign(message, keypair.private_key)

    assert len(keypair.public_key) == ED25519_PUBLIC_KEY_BYTES
    assert len(keypair.private_key) == ED25519_PRIVATE_KEY_BYTES
    assert len(signature) == ED25519_SIGNATURE_BYTES
    assert verify(message, signature, keypair.public_key)


def test_sign_rejects_malformed_inputs() -> None:
    with pytest.raises(TypeError):
        sign("message", RFC8032_TEST1_PRIVATE_KEY)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sign(b"message", "private")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sign(b"message", b"short")


def test_verify_rejects_malformed_inputs() -> None:
    with pytest.raises(TypeError):
        verify("message", RFC8032_TEST1_SIGNATURE, RFC8032_TEST1_PUBLIC_KEY)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        verify(b"", "signature", RFC8032_TEST1_PUBLIC_KEY)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        verify(b"", RFC8032_TEST1_SIGNATURE, "public")  # type: ignore[arg-type]

    assert not verify(b"", RFC8032_TEST1_SIGNATURE[:-1], RFC8032_TEST1_PUBLIC_KEY)
    assert not verify(b"", RFC8032_TEST1_SIGNATURE, RFC8032_TEST1_PUBLIC_KEY[:-1])
    assert not verify(b"wrong", RFC8032_TEST1_SIGNATURE, RFC8032_TEST1_PUBLIC_KEY)
    assert not verify(b"", _corrupt(RFC8032_TEST1_SIGNATURE), RFC8032_TEST1_PUBLIC_KEY)


def test_keypair_constructor_rejects_malformed_lengths_and_types() -> None:
    with pytest.raises(ValueError):
        Keypair(public_key=b"short", private_key=RFC8032_TEST1_PRIVATE_KEY)
    with pytest.raises(ValueError):
        Keypair(public_key=RFC8032_TEST1_PUBLIC_KEY, private_key=b"short")
    with pytest.raises(TypeError):
        Keypair(public_key="public", private_key=RFC8032_TEST1_PRIVATE_KEY)  # type: ignore[arg-type]


def test_verify_batch_accepts_all_valid_items_and_empty_batch() -> None:
    keypair = Keypair.generate()
    items = []
    for index in range(12):
        message = f"batch message {index}".encode()
        items.append((message, sign(message, keypair.private_key), keypair.public_key))

    assert verify_batch([])
    assert verify_batch(items)


def test_verify_batch_rejects_one_bad_signature() -> None:
    keypair = Keypair.generate()
    items = []
    for index in range(12):
        message = f"batch message {index}".encode()
        items.append((message, sign(message, keypair.private_key), keypair.public_key))

    message, signature, public_key = items[5]
    items[5] = (message, _corrupt(signature), public_key)

    assert not verify_batch(items)


def test_verify_batch_validates_shape() -> None:
    with pytest.raises(TypeError):
        verify_batch(())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        verify_batch([b"not-a-tuple"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        verify_batch([(b"message", b"signature")])  # type: ignore[list-item]


def test_verify_batch_short_circuits_after_first_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class FakePublicKey:
        def verify(self, signature: bytes, message: bytes) -> None:
            nonlocal calls
            calls += 1
            if signature == b"bad".ljust(ED25519_SIGNATURE_BYTES, b"\x00"):
                raise InvalidSignature

    def fake_load_public_key(public_key: bytes) -> FakePublicKey | None:
        return FakePublicKey()

    monkeypatch.setattr(signatures, "_load_public_key", fake_load_public_key)

    assert not verify_batch(
        [
            (b"message-0", b"good".ljust(ED25519_SIGNATURE_BYTES, b"\x00"), bytes(32)),
            (b"message-1", b"bad".ljust(ED25519_SIGNATURE_BYTES, b"\x00"), bytes(32)),
            (b"message-2", b"good".ljust(ED25519_SIGNATURE_BYTES, b"\x00"), bytes(32)),
        ]
    )
    assert calls == 2


def test_verify_batch_reuses_loaded_public_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = 0

    class FakePublicKey:
        def verify(self, signature: bytes, message: bytes) -> None:
            return None

    def fake_load_public_key(public_key: bytes) -> FakePublicKey | None:
        nonlocal loads
        loads += 1
        return FakePublicKey()

    monkeypatch.setattr(signatures, "_load_public_key", fake_load_public_key)

    assert verify_batch(
        [
            (b"message-0", bytes(ED25519_SIGNATURE_BYTES), bytes(ED25519_PUBLIC_KEY_BYTES)),
            (b"message-1", bytes(ED25519_SIGNATURE_BYTES), bytes(ED25519_PUBLIC_KEY_BYTES)),
            (b"message-2", bytes(ED25519_SIGNATURE_BYTES), bytes(ED25519_PUBLIC_KEY_BYTES)),
        ]
    )
    assert loads == 1


def test_verify_by_sig_type_dispatches_ed25519_and_rejects_unknown_types() -> None:
    assert verify_by_sig_type(
        SIG_TYPE_ED25519,
        b"",
        RFC8032_TEST1_SIGNATURE,
        RFC8032_TEST1_PUBLIC_KEY,
    )
    assert not verify_by_sig_type(
        SIG_TYPE_ML_DSA_RESERVED,
        b"",
        RFC8032_TEST1_SIGNATURE,
        RFC8032_TEST1_PUBLIC_KEY,
    )
    assert not verify_by_sig_type(
        255,
        b"",
        RFC8032_TEST1_SIGNATURE,
        RFC8032_TEST1_PUBLIC_KEY,
    )
    with pytest.raises(TypeError):
        verify_by_sig_type("0", b"", RFC8032_TEST1_SIGNATURE, RFC8032_TEST1_PUBLIC_KEY)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        verify_by_sig_type(False, b"", RFC8032_TEST1_SIGNATURE, RFC8032_TEST1_PUBLIC_KEY)


def test_verify_by_sig_type_uses_dispatch_table(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
        return message == b"message" and signature == b"signature" and public_key == b"public"

    monkeypatch.setitem(signatures._SIG_DISPATCH, 42, fake_verify)

    assert verify_by_sig_type(42, b"message", b"signature", b"public")
    assert not verify_by_sig_type(42, b"wrong", b"signature", b"public")
