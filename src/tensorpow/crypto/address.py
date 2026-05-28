"""TensorPoW Bech32m address encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tensorpow.crypto.hash import DOMAIN_ADDRESS, HASH_LEN_BYTES, domain_hash

HRP: Final[str] = "tsc"
BECH32M_CONST: Final[int] = 0x2BC830A3
CHARSET: Final[str] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
CHARSET_INDEX: Final[dict[str, int]] = {char: index for index, char in enumerate(CHARSET)}
CHECKSUM_LENGTH: Final[int] = 6
MAX_BECH32_LENGTH: Final[int] = 90
PUBKEY_BYTES: Final[int] = 32


@dataclass(frozen=True)
class AddressDecodeError(ValueError):
    """Address decoding failure with optional typo position."""

    message: str
    error_position: int | None = None

    def __str__(self) -> str:
        if self.error_position is None:
            return self.message
        return f"{self.message} at position {self.error_position}"


def pubkey_to_address(pubkey: bytes) -> str:
    """Encode an Ed25519 public key as a Tensorcoin address."""

    if not isinstance(pubkey, bytes):
        raise TypeError("pubkey must be bytes")
    if len(pubkey) != PUBKEY_BYTES:
        raise ValueError(f"pubkey must be {PUBKEY_BYTES} bytes")
    pubkey_hash = domain_hash(DOMAIN_ADDRESS, pubkey)
    return _bech32m_encode(HRP, _convertbits(pubkey_hash, 8, 5, pad=True))


def address_to_pubkey_hash(address: str) -> bytes:
    """Decode a Tensorcoin address into its 32-byte public-key hash."""

    hrp, data = _bech32m_decode(address)
    if hrp != HRP:
        raise AddressDecodeError(f"address HRP must be {HRP!r}")
    payload = _convertbits(data, 5, 8, pad=False)
    if len(payload) != HASH_LEN_BYTES:
        raise AddressDecodeError(f"address payload must be {HASH_LEN_BYTES} bytes")
    return bytes(payload)


def validate_address(address: str) -> bool:
    """Return True when address is a valid Tensorcoin Bech32m address."""

    try:
        address_to_pubkey_hash(address)
    except (AddressDecodeError, TypeError, ValueError):
        return False
    return True


def _bech32m_encode(hrp: str, data: list[int]) -> str:
    _validate_hrp(hrp)
    checksum = _create_checksum(hrp, data)
    combined = data + checksum
    return hrp + "1" + "".join(CHARSET[value] for value in combined)


def _bech32m_decode(address: str) -> tuple[str, list[int]]:
    if not isinstance(address, str):
        raise TypeError("address must be str")
    if len(address) > MAX_BECH32_LENGTH:
        raise AddressDecodeError("address is too long")
    if address != address.lower():
        raise AddressDecodeError("address must be lowercase")
    if any(ord(char) < 33 or ord(char) > 126 for char in address):
        raise AddressDecodeError("address contains invalid characters")

    separator = address.rfind("1")
    if separator < 1:
        raise AddressDecodeError("address is missing separator")
    if separator + CHECKSUM_LENGTH >= len(address):
        raise AddressDecodeError("address checksum is missing")

    hrp = address[:separator]
    _validate_hrp(hrp)
    data_part = address[separator + 1 :]
    try:
        data = [CHARSET_INDEX[char] for char in data_part]
    except KeyError as exc:
        position = separator + 1 + data_part.index(exc.args[0])
        raise AddressDecodeError("address contains invalid data character", position) from exc

    if not _verify_checksum(hrp, data):
        raise AddressDecodeError("address checksum is invalid", _locate_single_error(address))
    return hrp, data[:-CHECKSUM_LENGTH]


def _validate_hrp(hrp: str) -> None:
    if not hrp:
        raise AddressDecodeError("HRP must not be empty")
    if hrp != hrp.lower():
        raise AddressDecodeError("HRP must be lowercase")
    if any(ord(char) < 33 or ord(char) > 126 for char in hrp):
        raise AddressDecodeError("HRP contains invalid characters")


def _create_checksum(hrp: str, data: list[int]) -> list[int]:
    values = _hrp_expand(hrp) + data
    polymod = _polymod(values + [0] * CHECKSUM_LENGTH) ^ BECH32M_CONST
    return [
        (polymod >> (5 * (CHECKSUM_LENGTH - 1 - index))) & 31 for index in range(CHECKSUM_LENGTH)
    ]


def _verify_checksum(hrp: str, data: list[int]) -> bool:
    return _polymod(_hrp_expand(hrp) + data) == BECH32M_CONST


def _hrp_expand(hrp: str) -> list[int]:
    return [ord(char) >> 5 for char in hrp] + [0] + [ord(char) & 31 for char in hrp]


def _polymod(values: list[int]) -> int:
    generators = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    checksum = 1
    for value in values:
        top = checksum >> 25
        checksum = ((checksum & 0x1FFFFFF) << 5) ^ value
        for index, generator in enumerate(generators):
            if (top >> index) & 1:
                checksum ^= generator
    return checksum


def _convertbits(data: bytes | list[int], from_bits: int, to_bits: int, *, pad: bool) -> list[int]:
    accumulator = 0
    bits = 0
    output: list[int] = []
    max_value = (1 << to_bits) - 1
    max_accumulator = (1 << (from_bits + to_bits - 1)) - 1

    for value in data:
        if value < 0 or value >> from_bits:
            raise AddressDecodeError("address payload has invalid bit group")
        accumulator = ((accumulator << from_bits) | value) & max_accumulator
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            output.append((accumulator >> bits) & max_value)

    if pad:
        if bits:
            output.append((accumulator << (to_bits - bits)) & max_value)
    elif bits >= from_bits or ((accumulator << (to_bits - bits)) & max_value):
        raise AddressDecodeError("address payload padding is non-canonical")

    return output


def _locate_single_error(address: str) -> int | None:
    for position, original in enumerate(address):
        replacements = CHARSET if position > address.rfind("1") else "abcdefghijklmnopqrstuvwxyz"
        for replacement in replacements:
            if replacement == original:
                continue
            candidate = address[:position] + replacement + address[position + 1 :]
            try:
                _bech32m_decode_without_location(candidate)
            except AddressDecodeError:
                continue
            return position
    return None


def _bech32m_decode_without_location(address: str) -> None:
    separator = address.rfind("1")
    if separator < 1 or separator + CHECKSUM_LENGTH >= len(address):
        raise AddressDecodeError("invalid candidate")
    hrp = address[:separator]
    data_part = address[separator + 1 :]
    if address != address.lower() or any(char not in CHARSET for char in data_part):
        raise AddressDecodeError("invalid candidate")
    data = [CHARSET_INDEX[char] for char in data_part]
    if not _verify_checksum(hrp, data):
        raise AddressDecodeError("invalid candidate")
