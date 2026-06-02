"""Deterministic per-template transaction compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.crypto.signatures import (
    ED25519_PUBLIC_KEY_BYTES,
    ED25519_SIGNATURE_BYTES,
    SIG_TYPE_ED25519,
)
from tensorpow.state.utxo import (
    OUTPOINT_BYTES,
    TEMPLATE_HASHLOCK,
    TEMPLATE_MULTISIG,
    TEMPLATE_PKH,
)
from tensorpow.tx.script import MULTISIG_MAX_KEYS, pubkey_hash
from tensorpow.tx.transaction import (
    FORMAT_EPOCH,
    MAX_TX_BYTES,
    TX_SEQUENCE_FINAL,
    TX_WITNESS_MAX_BYTES,
    Input,
    Output,
    Transaction,
)

CODEC_ID_BYTES: Final[int] = 2
CODEC_RAW: Final[int] = 0x0000
CODEC_TEMPLATE_RANGE: Final[int] = 0x0001

U8_BYTES: Final[int] = 1
U16_BYTES: Final[int] = 2
U32_BYTES: Final[int] = 4
U64_BYTES: Final[int] = 8
U16_MAX: Final[int] = 0xFFFF
U32_MAX: Final[int] = 0xFFFFFFFF
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF

COMPRESSED_OBJECT_HEADER_BYTES: Final[int] = CODEC_ID_BYTES + (2 * U32_BYTES)
TEMPLATE_CODEC_MAGIC: Final[bytes] = b"TPTC"
TEMPLATE_RANGE_CODER_ADAPTIVE: Final[int] = 0x01
MAX_TEMPLATE_COMPRESSED_BYTES: Final[int] = MAX_TX_BYTES + COMPRESSED_OBJECT_HEADER_BYTES

TX_LOCKTIME_FLAG: Final[int] = 0x01
TX_LOCKHEIGHT_FLAG: Final[int] = 0x02
KNOWN_TX_FLAGS: Final[int] = TX_LOCKTIME_FLAG | TX_LOCKHEIGHT_FLAG

WITNESS_RAW: Final[int] = 0x00
WITNESS_ED25519: Final[int] = 0x01

OUT_PKH_DIRECT: Final[int] = 0x00
OUT_PKH_SIGNER_REF: Final[int] = 0x01
OUT_MULTISIG: Final[int] = 0x02
OUT_HASHLOCK: Final[int] = 0x03

OUTPUT_LOCKTIME_FLAG: Final[int] = 0x01
OUTPUT_LOCKHEIGHT_FLAG: Final[int] = 0x02
KNOWN_OUTPUT_FLAGS: Final[int] = OUTPUT_LOCKTIME_FLAG | OUTPUT_LOCKHEIGHT_FLAG

RANGE_CODE_BITS: Final[int] = 32
RANGE_CODE_FULL: Final[int] = 1 << RANGE_CODE_BITS
RANGE_CODE_HALF: Final[int] = RANGE_CODE_FULL >> 1
RANGE_CODE_QUARTER: Final[int] = RANGE_CODE_HALF >> 1
RANGE_CODE_THREE_QUARTER: Final[int] = RANGE_CODE_QUARTER * 3
RANGE_MODEL_SYMBOLS: Final[int] = 256
RANGE_MODEL_MAX_TOTAL: Final[int] = 1 << 15


class TemplateCodecError(ValueError):
    """Raised when template-compressed transaction bytes are invalid."""


@dataclass(slots=True)
class _Reader:
    data: bytes
    offset: int = 0

    def __post_init__(self) -> None:
        _require_bytes("data", self.data)

    def bytes(self, length: int) -> bytes:
        _require_uint("length", length, U32_MAX)
        end = self.offset + length
        if end > len(self.data):
            raise TemplateCodecError("template codec bytes are truncated")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def u8(self) -> int:
        return self.bytes(U8_BYTES)[0]

    def u64(self) -> int:
        return int.from_bytes(self.bytes(U64_BYTES), "little")

    def uvarint(self, *, max_value: int = U64_MAX) -> int:
        value, shift = 0, 0
        for index in range(10):
            byte = self.u8()
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                if index > 0 and value < (1 << (7 * index)):
                    raise TemplateCodecError("non-canonical varint")
                if value > max_value:
                    raise TemplateCodecError("varint exceeds maximum")
                return value
            shift += 7
        raise TemplateCodecError("varint is too long")

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise TemplateCodecError("trailing template codec bytes")


def compress_tx(tx: Transaction) -> bytes:
    """Return a canonical `CODEC_TEMPLATE_RANGE` compressed transaction object."""

    if not isinstance(tx, Transaction):
        raise TypeError("tx must be Transaction")
    if tx.version != FORMAT_EPOCH or tx.sig_type != SIG_TYPE_ED25519:
        raise TemplateCodecError("template codec supports only active default transaction fields")

    raw = tx.to_bytes()
    if len(raw) > MAX_TX_BYTES:
        raise TemplateCodecError("transaction exceeds MAX_TX_BYTES")

    template_payload = _compress_template_payload(tx)
    range_coded = _range_encode(template_payload)
    compressed = b"".join(
        (
            TEMPLATE_CODEC_MAGIC,
            bytes((TEMPLATE_RANGE_CODER_ADAPTIVE,)),
            len(template_payload).to_bytes(U32_BYTES, "little"),
            range_coded,
        )
    )
    return b"".join(
        (
            CODEC_TEMPLATE_RANGE.to_bytes(CODEC_ID_BYTES, "little"),
            len(raw).to_bytes(U32_BYTES, "little"),
            len(compressed).to_bytes(U32_BYTES, "little"),
            compressed,
        )
    )


def decompress_tx(data: bytes) -> Transaction:
    """Decode a canonical template-compressed transaction object."""

    _require_bytes("data", data, max_len=MAX_TEMPLATE_COMPRESSED_BYTES)
    if len(data) < COMPRESSED_OBJECT_HEADER_BYTES:
        raise TemplateCodecError("compressed object header is truncated")

    reader = _Reader(data)
    codec_id = int.from_bytes(reader.bytes(CODEC_ID_BYTES), "little")
    if codec_id != CODEC_TEMPLATE_RANGE:
        raise TemplateCodecError("unsupported transaction codec_id")
    uncompressed_len = int.from_bytes(reader.bytes(U32_BYTES), "little")
    compressed_len = int.from_bytes(reader.bytes(U32_BYTES), "little")
    if uncompressed_len > MAX_TX_BYTES:
        raise TemplateCodecError("uncompressed transaction length exceeds MAX_TX_BYTES")
    compressed = reader.bytes(compressed_len)
    reader.finish()

    tx = _decompress_body(compressed)
    raw = tx.to_bytes()
    if len(raw) != uncompressed_len:
        raise TemplateCodecError("uncompressed transaction length mismatch")
    if compress_tx(tx) != data:
        raise TemplateCodecError("template codec bytes are non-canonical")
    return tx


def _compress_template_payload(tx: Transaction) -> bytes:
    payload = bytearray()
    tx_flags = 0
    if tx.locktime_ms != 0:
        tx_flags |= TX_LOCKTIME_FLAG
    if tx.lockheight != 0:
        tx_flags |= TX_LOCKHEIGHT_FLAG
    payload.append(tx_flags)
    if tx.locktime_ms != 0:
        payload.extend(tx.locktime_ms.to_bytes(U64_BYTES, "little"))
    if tx.lockheight != 0:
        payload.extend(tx.lockheight.to_bytes(U64_BYTES, "little"))

    payload.extend(_uvarint(len(tx.inputs)))
    payload.extend(_uvarint(len(tx.outputs)))

    signer_hashes: list[bytes] = []
    for input_ in tx.inputs:
        payload.extend(input_.previous_outpoint.to_bytes())
        if _is_standard_ed25519_witness(input_.witness):
            payload.append(WITNESS_ED25519)
            signature = input_.witness[:ED25519_SIGNATURE_BYTES]
            public_key = input_.witness[ED25519_SIGNATURE_BYTES:]
            payload.extend(signature)
            payload.extend(public_key)
            signer_hashes.append(pubkey_hash(public_key))
        else:
            payload.append(WITNESS_RAW)
            payload.extend(_uvarint(len(input_.witness)))
            payload.extend(input_.witness)

    for output in tx.outputs:
        payload.extend(_uvarint(output.amount_matoms))
        payload.extend(_encode_output_payload(output, signer_hashes))
        output_flags = 0
        if output.locktime_ms != 0:
            output_flags |= OUTPUT_LOCKTIME_FLAG
        if output.lockheight != 0:
            output_flags |= OUTPUT_LOCKHEIGHT_FLAG
        payload.append(output_flags)
        if output.locktime_ms != 0:
            payload.extend(output.locktime_ms.to_bytes(U64_BYTES, "little"))
        if output.lockheight != 0:
            payload.extend(output.lockheight.to_bytes(U64_BYTES, "little"))

    compressed = bytes(payload)
    if len(compressed) > MAX_TEMPLATE_COMPRESSED_BYTES:
        raise TemplateCodecError("compressed transaction exceeds maximum size")
    return compressed


def _decompress_body(compressed: bytes) -> Transaction:
    _require_bytes("compressed", compressed, max_len=MAX_TEMPLATE_COMPRESSED_BYTES)
    reader = _Reader(compressed)
    if reader.bytes(len(TEMPLATE_CODEC_MAGIC)) != TEMPLATE_CODEC_MAGIC:
        raise TemplateCodecError("template codec magic is invalid")
    coder_id = reader.u8()
    if coder_id != TEMPLATE_RANGE_CODER_ADAPTIVE:
        raise TemplateCodecError("unsupported template range coder")
    template_payload_len = int.from_bytes(reader.bytes(U32_BYTES), "little")
    if template_payload_len > MAX_TEMPLATE_COMPRESSED_BYTES:
        raise TemplateCodecError("template payload length exceeds maximum")
    range_bytes = reader.bytes(len(reader.data) - reader.offset)
    reader.finish()
    try:
        template_payload = _range_decode(range_bytes, template_payload_len)
        return _decompress_template_payload(template_payload)
    except TemplateCodecError:
        raise
    except (TypeError, ValueError) as exc:
        raise TemplateCodecError("template range payload is malformed") from exc


def _decompress_template_payload(template_payload: bytes) -> Transaction:
    _require_bytes("template_payload", template_payload, max_len=MAX_TEMPLATE_COMPRESSED_BYTES)
    reader = _Reader(template_payload)
    tx_flags = reader.u8()
    if tx_flags & ~KNOWN_TX_FLAGS:
        raise TemplateCodecError("unknown transaction flags")
    locktime_ms = reader.u64() if tx_flags & TX_LOCKTIME_FLAG else 0
    lockheight = reader.u64() if tx_flags & TX_LOCKHEIGHT_FLAG else 0
    input_count = reader.uvarint(max_value=U16_MAX)
    output_count = reader.uvarint(max_value=U16_MAX)
    if output_count == 0:
        raise TemplateCodecError("output_count must be nonzero")

    signer_hashes: list[bytes] = []
    inputs = []
    for _ in range(input_count):
        previous_outpoint = reader.bytes(OUTPOINT_BYTES)
        witness_kind = reader.u8()
        if witness_kind == WITNESS_ED25519:
            signature = reader.bytes(ED25519_SIGNATURE_BYTES)
            public_key = reader.bytes(ED25519_PUBLIC_KEY_BYTES)
            witness = signature + public_key
            signer_hashes.append(pubkey_hash(public_key))
        elif witness_kind == WITNESS_RAW:
            witness_len = reader.uvarint(max_value=TX_WITNESS_MAX_BYTES)
            witness = reader.bytes(witness_len)
        else:
            raise TemplateCodecError("unknown witness kind")
        inputs.append(
            Input.from_bytes(
                b"".join(
                    (
                        previous_outpoint,
                        TX_SEQUENCE_FINAL.to_bytes(4, "little"),
                        len(witness).to_bytes(2, "little"),
                        witness,
                    )
                )
            )
        )

    outputs = []
    for _ in range(output_count):
        amount = reader.uvarint()
        template_id, payload = _decode_output_payload(reader, signer_hashes)
        output_flags = reader.u8()
        if output_flags & ~KNOWN_OUTPUT_FLAGS:
            raise TemplateCodecError("unknown output flags")
        output_locktime_ms = reader.u64() if output_flags & OUTPUT_LOCKTIME_FLAG else 0
        output_lockheight = reader.u64() if output_flags & OUTPUT_LOCKHEIGHT_FLAG else 0
        outputs.append(
            Output(
                amount_matoms=amount,
                template_id=template_id,
                locktime_ms=output_locktime_ms,
                lockheight=output_lockheight,
                payload=payload,
            )
        )

    reader.finish()
    return Transaction(
        version=FORMAT_EPOCH,
        sig_type=SIG_TYPE_ED25519,
        locktime_ms=locktime_ms,
        lockheight=lockheight,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def _range_encode(data: bytes) -> bytes:
    _require_bytes("data", data, max_len=MAX_TEMPLATE_COMPRESSED_BYTES)
    low = 0
    high = RANGE_CODE_FULL - 1
    pending_bits = 0
    output = _BitWriter()
    model = _RangeModel()

    def write_bit_plus_follow(bit: int) -> None:
        nonlocal pending_bits
        output.write(bit)
        follow = 1 - bit
        while pending_bits:
            output.write(follow)
            pending_bits -= 1

    for symbol in data:
        cumulative_low, cumulative_high, total = model.interval(symbol)
        span = high - low + 1
        high = low + (span * cumulative_high // total) - 1
        low = low + (span * cumulative_low // total)

        while True:
            if high < RANGE_CODE_HALF:
                write_bit_plus_follow(0)
            elif low >= RANGE_CODE_HALF:
                write_bit_plus_follow(1)
                low -= RANGE_CODE_HALF
                high -= RANGE_CODE_HALF
            elif low >= RANGE_CODE_QUARTER and high < RANGE_CODE_THREE_QUARTER:
                pending_bits += 1
                low -= RANGE_CODE_QUARTER
                high -= RANGE_CODE_QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1

        model.update(symbol)

    pending_bits += 1
    if low < RANGE_CODE_QUARTER:
        write_bit_plus_follow(0)
    else:
        write_bit_plus_follow(1)
    return output.finish()


def _range_decode(data: bytes, output_len: int) -> bytes:
    _require_bytes("data", data, max_len=MAX_TEMPLATE_COMPRESSED_BYTES)
    _require_uint("output_len", output_len, MAX_TEMPLATE_COMPRESSED_BYTES)
    if output_len == 0:
        return b""
    if not data:
        raise ValueError("range-coded data must not be empty")

    reader = _BitReader(data)
    low = 0
    high = RANGE_CODE_FULL - 1
    code = 0
    for _ in range(RANGE_CODE_BITS):
        code = (code << 1) | reader.read()

    model = _RangeModel()
    output = bytearray()
    for _ in range(output_len):
        span = high - low + 1
        total = model.total
        scaled = ((code - low + 1) * total - 1) // span
        symbol, cumulative_low, cumulative_high = model.symbol_for_scaled(scaled)

        high = low + (span * cumulative_high // total) - 1
        low = low + (span * cumulative_low // total)

        while True:
            if high < RANGE_CODE_HALF:
                pass
            elif low >= RANGE_CODE_HALF:
                code -= RANGE_CODE_HALF
                low -= RANGE_CODE_HALF
                high -= RANGE_CODE_HALF
            elif low >= RANGE_CODE_QUARTER and high < RANGE_CODE_THREE_QUARTER:
                code -= RANGE_CODE_QUARTER
                low -= RANGE_CODE_QUARTER
                high -= RANGE_CODE_QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            code = (code << 1) | reader.read()

        output.append(symbol)
        model.update(symbol)

    return bytes(output)


class _RangeModel:
    def __init__(self) -> None:
        self.frequencies = [1] * RANGE_MODEL_SYMBOLS
        self.total = RANGE_MODEL_SYMBOLS

    def interval(self, symbol: int) -> tuple[int, int, int]:
        _require_uint("symbol", symbol, RANGE_MODEL_SYMBOLS - 1)
        cumulative_low = sum(self.frequencies[:symbol])
        cumulative_high = cumulative_low + self.frequencies[symbol]
        return cumulative_low, cumulative_high, self.total

    def symbol_for_scaled(self, scaled: int) -> tuple[int, int, int]:
        _require_uint("scaled", scaled, self.total - 1)
        cumulative = 0
        for symbol, frequency in enumerate(self.frequencies):
            next_cumulative = cumulative + frequency
            if scaled < next_cumulative:
                return symbol, cumulative, next_cumulative
            cumulative = next_cumulative
        raise ValueError("range scaled value outside model")

    def update(self, symbol: int) -> None:
        _require_uint("symbol", symbol, RANGE_MODEL_SYMBOLS - 1)
        self.frequencies[symbol] += 1
        self.total += 1
        if self.total >= RANGE_MODEL_MAX_TOTAL:
            self.frequencies = [max(1, (frequency + 1) // 2) for frequency in self.frequencies]
            self.total = sum(self.frequencies)


class _BitWriter:
    def __init__(self) -> None:
        self._data = bytearray()
        self._current = 0
        self._bits = 0

    def write(self, bit: int) -> None:
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        self._current = (self._current << 1) | bit
        self._bits += 1
        if self._bits == 8:
            self._data.append(self._current)
            self._current = 0
            self._bits = 0

    def finish(self) -> bytes:
        if self._bits:
            self._data.append(self._current << (8 - self._bits))
        return bytes(self._data)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        _require_bytes("data", data, max_len=MAX_TEMPLATE_COMPRESSED_BYTES)
        self._data = data
        self._bit_offset = 0

    def read(self) -> int:
        byte_offset, bit_in_byte = divmod(self._bit_offset, 8)
        self._bit_offset += 1
        if byte_offset >= len(self._data):
            return 0
        return (self._data[byte_offset] >> (7 - bit_in_byte)) & 1


def _encode_output_payload(output: Output, signer_hashes: list[bytes]) -> bytes:
    if output.template_id == TEMPLATE_PKH:
        try:
            signer_index = signer_hashes.index(output.payload)
        except ValueError:
            return bytes((OUT_PKH_DIRECT,)) + output.payload
        return bytes((OUT_PKH_SIGNER_REF,)) + _uvarint(signer_index)
    if output.template_id == TEMPLATE_MULTISIG:
        return bytes((OUT_MULTISIG,)) + output.payload
    if output.template_id == TEMPLATE_HASHLOCK:
        lock_hash = output.payload[:HASH_LEN_BYTES]
        inner_payload = output.payload[HASH_LEN_BYTES:]
        return bytes((OUT_HASHLOCK,)) + lock_hash + _encode_hashlock_inner_payload(inner_payload)
    raise TemplateCodecError("unsupported output template")


def _decode_output_payload(reader: _Reader, signer_hashes: list[bytes]) -> tuple[int, bytes]:
    output_kind = reader.u8()
    if output_kind == OUT_PKH_DIRECT:
        return TEMPLATE_PKH, reader.bytes(HASH_LEN_BYTES)
    if output_kind == OUT_PKH_SIGNER_REF:
        signer_index = reader.uvarint(max_value=U16_MAX)
        try:
            return TEMPLATE_PKH, signer_hashes[signer_index]
        except IndexError as exc:
            raise TemplateCodecError("PKH signer reference is out of range") from exc
    if output_kind == OUT_MULTISIG:
        header = reader.bytes(2)
        threshold = header[0]
        pubkey_count = header[1]
        if not 1 <= pubkey_count <= MULTISIG_MAX_KEYS:
            raise TemplateCodecError("multisig pubkey_count outside range")
        if not 1 <= threshold <= pubkey_count:
            raise TemplateCodecError("multisig threshold outside range")
        return TEMPLATE_MULTISIG, header + reader.bytes(pubkey_count * ED25519_PUBLIC_KEY_BYTES)
    if output_kind == OUT_HASHLOCK:
        lock_hash = reader.bytes(HASH_LEN_BYTES)
        return TEMPLATE_HASHLOCK, lock_hash + _decode_hashlock_inner_payload(reader)
    raise TemplateCodecError("unknown output kind")


def _encode_hashlock_inner_payload(payload: bytes) -> bytes:
    if len(payload) == HASH_LEN_BYTES:
        return bytes((OUT_PKH_DIRECT,)) + payload
    if len(payload) >= 2:
        threshold = payload[0]
        pubkey_count = payload[1]
        expected_len = 2 + pubkey_count * ED25519_PUBLIC_KEY_BYTES
        if (
            1 <= pubkey_count <= MULTISIG_MAX_KEYS
            and 1 <= threshold <= pubkey_count
            and len(payload) == expected_len
        ):
            return bytes((OUT_MULTISIG,)) + payload
    raise TemplateCodecError("unsupported hashlock inner template")


def _decode_hashlock_inner_payload(reader: _Reader) -> bytes:
    inner_kind = reader.u8()
    if inner_kind == OUT_PKH_DIRECT:
        return reader.bytes(HASH_LEN_BYTES)
    if inner_kind == OUT_MULTISIG:
        header = reader.bytes(2)
        threshold = header[0]
        pubkey_count = header[1]
        if not 1 <= pubkey_count <= MULTISIG_MAX_KEYS:
            raise TemplateCodecError("hashlock multisig pubkey_count outside range")
        if not 1 <= threshold <= pubkey_count:
            raise TemplateCodecError("hashlock multisig threshold outside range")
        return header + reader.bytes(pubkey_count * ED25519_PUBLIC_KEY_BYTES)
    raise TemplateCodecError("unknown hashlock inner template")


def _is_standard_ed25519_witness(witness: bytes) -> bool:
    return len(witness) == ED25519_SIGNATURE_BYTES + ED25519_PUBLIC_KEY_BYTES


def _uvarint(value: int) -> bytes:
    _require_uint("value", value, U64_MAX)
    output = bytearray()
    remaining = value
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            output.append(byte | 0x80)
        else:
            output.append(byte)
            return bytes(output)


def _require_bytes(name: str, value: bytes, *, max_len: int | None = None) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if max_len is not None and len(value) > max_len:
        raise ValueError(f"{name} exceeds max length")


def _require_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} outside uint range")


__all__ = [
    "CODEC_ID_BYTES",
    "CODEC_RAW",
    "CODEC_TEMPLATE_RANGE",
    "COMPRESSED_OBJECT_HEADER_BYTES",
    "MAX_TEMPLATE_COMPRESSED_BYTES",
    "TEMPLATE_CODEC_MAGIC",
    "TEMPLATE_RANGE_CODER_ADAPTIVE",
    "TemplateCodecError",
    "compress_tx",
    "decompress_tx",
]
