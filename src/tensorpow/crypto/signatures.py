"""Ed25519 signatures and signature-type dispatch for TensorPoW."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

SIG_TYPE_ED25519: Final[int] = 0
SIG_TYPE_ML_DSA_RESERVED: Final[int] = 1
SIG_TYPE_ED25519_BIT: Final[int] = 1 << SIG_TYPE_ED25519

ED25519_PUBLIC_KEY_BYTES: Final[int] = 32
ED25519_PRIVATE_KEY_BYTES: Final[int] = 32
ED25519_SIGNATURE_BYTES: Final[int] = 64

_Verifier = Callable[[bytes, bytes, bytes], bool]

# Explicit Ed25519 acceptance rules (PROTOCOL_SPEC section 2), checked before the
# library call so validity never depends on a particular backend's edge-case choices.
ED25519_FIELD_PRIME: Final[int] = 2**255 - 19
ED25519_GROUP_ORDER: Final[int] = 2**252 + 27742317777372353535851937790883648493
# Encodings of the small-order points with the sign bit cleared (libsodium's list):
# identity (order 1), (0, -1) (order 2), the two order-4 x-signs share y = 0, and the
# order-8 points. Non-canonical y >= p encodings are rejected separately.
_SMALL_ORDER_ENCODINGS_MASKED: Final[frozenset[bytes]] = frozenset(
    bytes.fromhex(value)
    for value in (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0100000000000000000000000000000000000000000000000000000000000000",
        "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
        "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    )
)


def _require_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")


def _require_exact_bytes(name: str, value: bytes, expected_len: int) -> None:
    _require_bytes(name, value)
    if len(value) != expected_len:
        raise ValueError(f"{name} must be {expected_len} bytes")


@dataclass(frozen=True, slots=True)
class Keypair:
    """Raw Ed25519 keypair bytes."""

    public_key: bytes
    private_key: bytes

    def __post_init__(self) -> None:
        _require_exact_bytes("public_key", self.public_key, ED25519_PUBLIC_KEY_BYTES)
        _require_exact_bytes("private_key", self.private_key, ED25519_PRIVATE_KEY_BYTES)

    @classmethod
    def generate(cls) -> Keypair:
        private_key = Ed25519PrivateKey.generate()
        return cls(
            public_key=private_key.public_key().public_bytes_raw(),
            private_key=private_key.private_bytes_raw(),
        )


def sign(message: bytes, private_key: bytes) -> bytes:
    """Sign message with a raw 32-byte Ed25519 private-key seed."""

    _require_bytes("message", message)
    _require_exact_bytes("private_key", private_key, ED25519_PRIVATE_KEY_BYTES)
    return Ed25519PrivateKey.from_private_bytes(private_key).sign(message)


def verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Return whether signature is a valid Ed25519 signature for message."""

    _require_bytes("message", message)
    _require_bytes("signature", signature)
    _require_bytes("public_key", public_key)
    if len(signature) != ED25519_SIGNATURE_BYTES:
        return False
    if not strict_encoding_ok(signature, public_key):
        return False
    loaded_public_key = _load_public_key(public_key)
    if loaded_public_key is None:
        return False
    return _verify_with_loaded_public_key(message, signature, loaded_public_key)


def strict_encoding_ok(signature: bytes, public_key: bytes) -> bool:
    """Return whether ``signature`` and ``public_key`` pass the explicit encoding rules.

    * ``S`` (the second half of the signature) is canonical: ``S < L``.
    * ``A`` (the public key) and ``R`` (the first half of the signature) are canonical
      point encodings (``y < p``) and are not small-order points.
    """

    if len(signature) != ED25519_SIGNATURE_BYTES or len(public_key) != ED25519_PUBLIC_KEY_BYTES:
        return False
    if int.from_bytes(signature[32:], "little") >= ED25519_GROUP_ORDER:
        return False
    return _point_encoding_ok(public_key) and _point_encoding_ok(signature[:32])


def _point_encoding_ok(encoded: bytes) -> bool:
    masked = encoded[:31] + bytes((encoded[31] & 0x7F,))
    if int.from_bytes(masked, "little") >= ED25519_FIELD_PRIME:
        return False
    return masked not in _SMALL_ORDER_ENCODINGS_MASKED


def _load_public_key(public_key: bytes) -> Ed25519PublicKey | None:
    if len(public_key) != ED25519_PUBLIC_KEY_BYTES:
        return None
    try:
        return Ed25519PublicKey.from_public_bytes(public_key)
    except ValueError:
        return None


def _verify_with_loaded_public_key(
    message: bytes,
    signature: bytes,
    loaded_public_key: Ed25519PublicKey,
) -> bool:
    try:
        loaded_public_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def verify_batch(items: list[tuple[bytes, bytes, bytes]]) -> bool:
    """Return whether every ``(message, signature, public_key)`` item verifies."""

    if not isinstance(items, list):
        raise TypeError("items must be a list")

    loaded_public_keys: dict[bytes, Ed25519PublicKey] = {}
    for item in items:
        if not isinstance(item, tuple):
            raise TypeError("batch item must be a tuple")
        if len(item) != 3:
            raise ValueError("batch item must contain message, signature, and public_key")
        message, signature, public_key = item
        _require_bytes("message", message)
        _require_bytes("signature", signature)
        _require_bytes("public_key", public_key)
        if len(signature) != ED25519_SIGNATURE_BYTES:
            return False
        if not strict_encoding_ok(signature, public_key):
            return False

        loaded_public_key = loaded_public_keys.get(public_key)
        if loaded_public_key is None:
            loaded_public_key = _load_public_key(public_key)
            if loaded_public_key is None:
                return False
            loaded_public_keys[public_key] = loaded_public_key

        if not _verify_with_loaded_public_key(message, signature, loaded_public_key):
            return False
    return True


_SIG_DISPATCH: dict[int, _Verifier] = {
    SIG_TYPE_ED25519: verify,
}


def verify_by_sig_type(sig_type: int, message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify signature through the protocol signature-type dispatch table."""

    if isinstance(sig_type, bool):
        raise TypeError("sig_type must be int, not bool")
    if not isinstance(sig_type, int):
        raise TypeError("sig_type must be int")

    verifier = _SIG_DISPATCH.get(sig_type)
    if verifier is None:
        return False
    return verifier(message, signature, public_key)
