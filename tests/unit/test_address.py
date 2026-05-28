"""Tests for TensorPoW Bech32m addresses."""

from __future__ import annotations

import pytest

from tensorpow.crypto.address import (
    AddressDecodeError,
    address_to_pubkey_hash,
    pubkey_to_address,
    validate_address,
)
from tensorpow.crypto.hash import DOMAIN_ADDRESS, domain_hash


def test_pubkey_to_address_round_trips_to_pubkey_hash() -> None:
    pubkey = bytes(range(32))
    address = pubkey_to_address(pubkey)

    assert address == "tsc1jtxscgwnn8fchc760suu22hj9flazx5jqkfrh2cg6c84f5ql5pgs4rav4v"
    assert address_to_pubkey_hash(address) == domain_hash(DOMAIN_ADDRESS, pubkey)
    assert validate_address(address)


def test_pubkey_to_address_rejects_malformed_pubkeys() -> None:
    with pytest.raises(TypeError):
        pubkey_to_address("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pubkey_to_address(b"short")


def test_address_rejects_wrong_hrp() -> None:
    wrong_hrp = "btc1jtxscgwnn8fchc760suu22hj9flazx5jqkfrh2cg6c84f5ql5pgsy3a8vs"

    with pytest.raises(AddressDecodeError, match="HRP"):
        address_to_pubkey_hash(wrong_hrp)
    assert not validate_address(wrong_hrp)


def test_address_rejects_mixed_case() -> None:
    address = pubkey_to_address(bytes(range(32)))
    mixed = address[:4].upper() + address[4:]

    with pytest.raises(AddressDecodeError, match="lowercase"):
        address_to_pubkey_hash(mixed)
    assert not validate_address(mixed)


def test_address_rejects_bad_checksum_and_reports_position() -> None:
    address = pubkey_to_address(bytes(range(32)))
    typo = address[:10] + ("q" if address[10] != "q" else "p") + address[11:]

    with pytest.raises(AddressDecodeError) as exc_info:
        address_to_pubkey_hash(typo)

    assert exc_info.value.error_position == 10
    assert not validate_address(typo)


def test_address_rejects_noncanonical_padding() -> None:
    address = pubkey_to_address(bytes(range(32)))
    separator = address.rfind("1")
    data = address[separator + 1 : -6]
    checksum = address[-6:]
    malformed = address[: separator + 1] + data + "q" + checksum

    with pytest.raises(AddressDecodeError):
        address_to_pubkey_hash(malformed)
    assert not validate_address(malformed)
