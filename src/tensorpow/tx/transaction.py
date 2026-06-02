"""Canonical transaction byte layout for TensorPoW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tensorpow.crypto.hash import DOMAIN_TX_ID, DOMAIN_TX_SIGHASH, HASH_LEN_BYTES, domain_hash
from tensorpow.crypto.signatures import ED25519_PUBLIC_KEY_BYTES, SIG_TYPE_ED25519
from tensorpow.state.utxo import (
    ACTIVE_TEMPLATES,
    MAX_SUPPLY_MATOMS,
    OUTPOINT_BYTES,
    TEMPLATE_HASHLOCK,
    TEMPLATE_MULTISIG,
    TEMPLATE_PKH,
    TX_OUTPUT_PAYLOAD_MAX_BYTES,
    Outpoint,
)

FORMAT_EPOCH: Final[int] = 0

U8_BYTES: Final[int] = 1
U16_BYTES: Final[int] = 2
U32_BYTES: Final[int] = 4
U64_BYTES: Final[int] = 8

U8_MAX: Final[int] = 0xFF
U16_MAX: Final[int] = 0xFFFF
U32_MAX: Final[int] = 0xFFFFFFFF
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF

MAX_TX_BYTES: Final[int] = 8192
TX_SEQUENCE_FINAL: Final[int] = 0xFFFFFFFF
TX_WITNESS_MAX_BYTES: Final[int] = 2048
COINBASE_INPUT_COUNT: Final[int] = 0

TX_HEADER_BYTES: Final[int] = U16_BYTES + U8_BYTES + U64_BYTES + U64_BYTES + U16_BYTES
INPUT_FIXED_BYTES: Final[int] = OUTPOINT_BYTES + U32_BYTES + U16_BYTES
OUTPUT_FIXED_BYTES: Final[int] = U64_BYTES + U16_BYTES + U64_BYTES + U64_BYTES + U16_BYTES
MULTISIG_MAX_KEYS: Final[int] = 15


class TxDecodeError(ValueError):
    """Raised when transaction bytes are malformed or non-canonical."""


@dataclass(frozen=True, slots=True)
class Input:
    """Transaction input spending one previous outpoint."""

    previous_outpoint: Outpoint
    sequence: int = TX_SEQUENCE_FINAL
    witness: bytes = b""

    def __post_init__(self) -> None:
        if not isinstance(self.previous_outpoint, Outpoint):
            raise TypeError("previous_outpoint must be Outpoint")
        _require_u32("sequence", self.sequence)
        if self.sequence != TX_SEQUENCE_FINAL:
            raise ValueError("sequence must equal TX_SEQUENCE_FINAL")
        _require_bytes("witness", self.witness, max_len=TX_WITNESS_MAX_BYTES)

    @classmethod
    def from_bytes(cls, data: bytes) -> Input:
        """Decode one canonical input and reject trailing bytes."""

        try:
            reader = _Reader(data)
            input_ = _read_input(reader)
            reader.finish()
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TxDecodeError):
                raise
            raise TxDecodeError(str(exc)) from exc
        return input_

    def to_bytes(self) -> bytes:
        """Encode the section-10 input byte layout."""

        return b"".join(
            (
                self.previous_outpoint.to_bytes(),
                _u32(self.sequence),
                _u16(len(self.witness)),
                self.witness,
            )
        )


@dataclass(frozen=True, slots=True)
class Output:
    """Transaction output with a template-specific payload."""

    amount_matoms: int
    template_id: int
    locktime_ms: int = 0
    lockheight: int = 0
    payload: bytes = b""

    def __post_init__(self) -> None:
        _require_u64("amount_matoms", self.amount_matoms)
        if self.amount_matoms == 0:
            raise ValueError("amount_matoms must be nonzero")
        if self.amount_matoms > MAX_SUPPLY_MATOMS:
            raise ValueError("amount_matoms exceeds MAX_SUPPLY_MATOMS")
        _require_u16("template_id", self.template_id)
        if self.template_id not in ACTIVE_TEMPLATES:
            raise ValueError("template_id must be an active output template")
        _require_u64("locktime_ms", self.locktime_ms)
        _require_u64("lockheight", self.lockheight)
        _require_bytes("payload", self.payload, max_len=TX_OUTPUT_PAYLOAD_MAX_BYTES)
        _require_template_payload(self.template_id, self.payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> Output:
        """Decode one canonical output and reject trailing bytes."""

        try:
            reader = _Reader(data)
            output = _read_output(reader)
            reader.finish()
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TxDecodeError):
                raise
            raise TxDecodeError(str(exc)) from exc
        return output

    def to_bytes(self) -> bytes:
        """Encode the section-10 output byte layout."""

        return b"".join(
            (
                _u64(self.amount_matoms),
                _u16(self.template_id),
                _u64(self.locktime_ms),
                _u64(self.lockheight),
                _u16(len(self.payload)),
                self.payload,
            )
        )


@dataclass(frozen=True, slots=True)
class Transaction:
    """Canonical TensorPoW transaction."""

    version: int
    sig_type: int
    locktime_ms: int
    lockheight: int
    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]

    def __post_init__(self) -> None:
        _require_format_epoch(self.version)
        _require_sig_type(self.sig_type)
        _require_u64("locktime_ms", self.locktime_ms)
        _require_u64("lockheight", self.lockheight)
        _require_inputs(self.inputs)
        _require_outputs(self.outputs)
        if len(self.to_bytes()) > MAX_TX_BYTES:
            raise ValueError(f"transaction exceeds {MAX_TX_BYTES} bytes")

    @classmethod
    def coinbase(
        cls,
        outputs: tuple[Output, ...],
        *,
        locktime_ms: int = 0,
        lockheight: int = 0,
        sig_type: int = SIG_TYPE_ED25519,
    ) -> Transaction:
        """Build a zero-input coinbase transaction."""

        return cls(
            version=FORMAT_EPOCH,
            sig_type=sig_type,
            locktime_ms=locktime_ms,
            lockheight=lockheight,
            inputs=(),
            outputs=outputs,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Transaction:
        """Decode canonical transaction bytes."""

        try:
            _require_bytes("data", data, max_len=MAX_TX_BYTES)
            reader = _Reader(data)
            version = reader.u16()
            sig_type = reader.u8()
            locktime_ms = reader.u64()
            lockheight = reader.u64()
            input_count = reader.u16()
            inputs = tuple(_read_input(reader) for _ in range(input_count))
            output_count = reader.u16()
            outputs = tuple(_read_output(reader) for _ in range(output_count))
            reader.finish()
            tx = cls(
                version=version,
                sig_type=sig_type,
                locktime_ms=locktime_ms,
                lockheight=lockheight,
                inputs=inputs,
                outputs=outputs,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TxDecodeError):
                raise
            raise TxDecodeError(str(exc)) from exc
        if tx.to_bytes() != data:
            raise TxDecodeError("transaction bytes are non-canonical")
        return tx

    def to_bytes(self) -> bytes:
        """Encode the section-10 transaction byte layout."""

        parts = [
            _u16(self.version),
            _u8(self.sig_type),
            _u64(self.locktime_ms),
            _u64(self.lockheight),
            _u16(len(self.inputs)),
        ]
        parts.extend(input_.to_bytes() for input_ in self.inputs)
        parts.append(_u16(len(self.outputs)))
        parts.extend(output.to_bytes() for output in self.outputs)
        return b"".join(parts)

    serialize = to_bytes

    def tx_id(self) -> bytes:
        """Return `BLAKE3(DOMAIN_TX_ID || canonical_tx_bytes)`."""

        return domain_hash(DOMAIN_TX_ID, self.to_bytes())

    def sighash(self, input_index: int) -> bytes:
        """Return the section-10 signature hash for one input."""

        _require_u32("input_index", input_index)
        if input_index >= len(self.inputs):
            raise IndexError("input_index outside transaction inputs")
        payload = self._to_bytes(empty_witnesses=True) + _u32(input_index)
        return domain_hash(DOMAIN_TX_SIGHASH, payload)

    def without_witnesses(self) -> Transaction:
        """Return the same transaction with every input witness emptied."""

        return Transaction(
            version=self.version,
            sig_type=self.sig_type,
            locktime_ms=self.locktime_ms,
            lockheight=self.lockheight,
            inputs=tuple(
                Input(previous_outpoint=input_.previous_outpoint, sequence=input_.sequence)
                for input_ in self.inputs
            ),
            outputs=self.outputs,
        )

    def _to_bytes(self, *, empty_witnesses: bool) -> bytes:
        if not empty_witnesses:
            return self.to_bytes()

        parts = [
            _u16(self.version),
            _u8(self.sig_type),
            _u64(self.locktime_ms),
            _u64(self.lockheight),
            _u16(len(self.inputs)),
        ]
        for input_ in self.inputs:
            parts.extend(
                (
                    input_.previous_outpoint.to_bytes(),
                    _u32(input_.sequence),
                    _u16(0),
                )
            )
        parts.append(_u16(len(self.outputs)))
        parts.extend(output.to_bytes() for output in self.outputs)
        return b"".join(parts)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = _require_bytes("data", data)
        self._offset = 0

    def bytes(self, length: int) -> bytes:
        _require_u32("length", length)
        end = self._offset + length
        if end > len(self._data):
            raise TxDecodeError("truncated transaction")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def u8(self) -> int:
        return int.from_bytes(self.bytes(U8_BYTES), "little")

    def u16(self) -> int:
        return int.from_bytes(self.bytes(U16_BYTES), "little")

    def u32(self) -> int:
        return int.from_bytes(self.bytes(U32_BYTES), "little")

    def u64(self) -> int:
        return int.from_bytes(self.bytes(U64_BYTES), "little")

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise TxDecodeError("trailing transaction bytes")


def _read_input(reader: _Reader) -> Input:
    return Input(
        previous_outpoint=Outpoint.from_bytes(reader.bytes(OUTPOINT_BYTES)),
        sequence=reader.u32(),
        witness=reader.bytes(reader.u16()),
    )


def _read_output(reader: _Reader) -> Output:
    return Output(
        amount_matoms=reader.u64(),
        template_id=reader.u16(),
        locktime_ms=reader.u64(),
        lockheight=reader.u64(),
        payload=reader.bytes(reader.u16()),
    )


def _require_format_epoch(value: int) -> None:
    _require_u16("version", value)
    if value != FORMAT_EPOCH:
        raise ValueError("version must equal FORMAT_EPOCH")


def _require_sig_type(value: int) -> None:
    _require_u8("sig_type", value)
    if value != SIG_TYPE_ED25519:
        raise ValueError("sig_type must be an active signature type")


def _require_inputs(inputs: tuple[Input, ...]) -> None:
    if not isinstance(inputs, tuple):
        raise TypeError("inputs must be a tuple")
    if len(inputs) > U16_MAX:
        raise ValueError("input_count outside uint range")
    for input_ in inputs:
        if not isinstance(input_, Input):
            raise TypeError("inputs must contain Input values")


def _require_outputs(outputs: tuple[Output, ...]) -> None:
    if not isinstance(outputs, tuple):
        raise TypeError("outputs must be a tuple")
    if len(outputs) == 0:
        raise ValueError("output_count must be nonzero")
    if len(outputs) > U16_MAX:
        raise ValueError("output_count outside uint range")
    for output in outputs:
        if not isinstance(output, Output):
            raise TypeError("outputs must contain Output values")


def _require_template_payload(template_id: int, payload: bytes) -> None:
    if template_id == TEMPLATE_PKH:
        if len(payload) != HASH_LEN_BYTES:
            raise ValueError("PKH template payload must be 32 bytes")
        return
    if template_id == TEMPLATE_MULTISIG:
        _require_multisig_payload(payload)
        return
    if template_id == TEMPLATE_HASHLOCK:
        if len(payload) <= HASH_LEN_BYTES:
            raise ValueError("hashlock template payload must include hash plus inner payload")
        _require_inner_template_payload(payload[HASH_LEN_BYTES:])
        return
    raise ValueError("template_id must be an active output template")


def _require_inner_template_payload(payload: bytes) -> None:
    if len(payload) == HASH_LEN_BYTES:
        return
    _require_multisig_payload(payload)


def _require_multisig_payload(payload: bytes) -> None:
    if len(payload) < 2:
        raise ValueError("multisig template payload is truncated")
    threshold = payload[0]
    pubkey_count = payload[1]
    if not 1 <= pubkey_count <= MULTISIG_MAX_KEYS:
        raise ValueError("multisig template pubkey_count outside range")
    if not 1 <= threshold <= pubkey_count:
        raise ValueError("multisig template threshold outside range")
    expected_len = 2 + pubkey_count * ED25519_PUBLIC_KEY_BYTES
    if len(payload) != expected_len:
        raise ValueError("multisig template payload length mismatch")
    public_keys = tuple(
        payload[offset : offset + ED25519_PUBLIC_KEY_BYTES]
        for offset in range(2, len(payload), ED25519_PUBLIC_KEY_BYTES)
    )
    if len(set(public_keys)) != len(public_keys):
        raise ValueError("multisig template public keys must be distinct")


def _require_u8(name: str, value: int) -> None:
    _require_uint(name, value, U8_MAX)


def _require_u16(name: str, value: int) -> None:
    _require_uint(name, value, U16_MAX)


def _require_u32(name: str, value: int) -> None:
    _require_uint(name, value, U32_MAX)


def _require_u64(name: str, value: int) -> None:
    _require_uint(name, value, U64_MAX)


def _require_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} outside uint range")


def _require_bytes(name: str, value: bytes, *, max_len: int | None = None) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if max_len is not None and len(value) > max_len:
        raise ValueError(f"{name} exceeds max length")
    return value


def _u8(value: int) -> bytes:
    return value.to_bytes(U8_BYTES, "little")


def _u16(value: int) -> bytes:
    return value.to_bytes(U16_BYTES, "little")


def _u32(value: int) -> bytes:
    return value.to_bytes(U32_BYTES, "little")


def _u64(value: int) -> bytes:
    return value.to_bytes(U64_BYTES, "little")


__all__ = [
    "COINBASE_INPUT_COUNT",
    "FORMAT_EPOCH",
    "INPUT_FIXED_BYTES",
    "MAX_TX_BYTES",
    "MULTISIG_MAX_KEYS",
    "OUTPUT_FIXED_BYTES",
    "TX_HEADER_BYTES",
    "TX_SEQUENCE_FINAL",
    "TX_WITNESS_MAX_BYTES",
    "Input",
    "Output",
    "Transaction",
    "TxDecodeError",
]
