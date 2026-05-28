"""Optional deterministic learned residual codec for template-compressed transactions."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch

from tensorpow.codec.template import (
    CODEC_ID_BYTES,
    CODEC_TEMPLATE_RANGE,
    COMPRESSED_OBJECT_HEADER_BYTES,
    MAX_TEMPLATE_COMPRESSED_BYTES,
    TemplateCodecError,
)
from tensorpow.codec.template import (
    compress_tx as _compress_template_tx,
)
from tensorpow.codec.template import (
    decompress_tx as _decompress_template_tx,
)
from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes
from tensorpow.pow.kernel import configure_torch_determinism
from tensorpow.tx.transaction import MAX_TX_BYTES, Transaction

type LearnedBackend = Literal["auto", "cpu", "cuda", "mps"]
type StrPath = str | PathLike[str]

CODEC_LEARNED: Final[int] = 0x0004
LEARNED_CODEC_MAGIC: Final[bytes] = b"TPLC"
LEARNED_CODEC_WEIGHTS_PATH: Final[Path] = Path("data/learned_codec.npz")
LEARNED_CODEC_WEIGHTS_HASH: Final[bytes] = bytes(HASH_LEN_BYTES)
LEARNED_CODEC_DISABLED_HASH: Final[bytes] = LEARNED_CODEC_WEIGHTS_HASH
LEARNED_CODEC_EXTRA_COMPRESSION_PCT: Final[int] = 15

U8_BYTES: Final[int] = 1
U32_BYTES: Final[int] = 4
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF
INT8_MIN_VALUE: Final[int] = -128
INT8_MAX_VALUE: Final[int] = 127
BYTE_VALUES: Final[int] = 256
LEARNED_CODEC_INT8_ZERO_POINT: Final[int] = 128
INT8_ZERO_POINT: Final[int] = LEARNED_CODEC_INT8_ZERO_POINT

WEIGHTS_POSITION_PRIOR_KEY: Final[str] = "position_prior"
WEIGHTS_FALLBACK_PRIOR_KEY: Final[str] = "fallback_prior"
WEIGHTS_HASH_TAG: Final[bytes] = b"TensorPoW learned tx codec weights"

MAX_LEARNED_COMPRESSED_BYTES: Final[int] = (
    COMPRESSED_OBJECT_HEADER_BYTES
    + len(LEARNED_CODEC_MAGIC)
    + HASH_LEN_BYTES
    + (MAX_TEMPLATE_COMPRESSED_BYTES * 3)
    + 20
)

_VALID_BACKENDS: Final[frozenset[str]] = frozenset(("auto", "cpu", "cuda", "mps"))


class LearnedCodecError(ValueError):
    """Raised when learned-compressed transaction bytes or weights are invalid."""


@dataclass(frozen=True, slots=True)
class LearnedCodecWeights:
    """Frozen INT8 byte-prior weights for the learned transaction codec."""

    position_prior: tuple[int, ...]
    fallback_prior: int
    weights_hash: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.position_prior, tuple):
            raise TypeError("position_prior must be a tuple")
        if not self.position_prior:
            raise ValueError("position_prior must not be empty")
        if len(self.position_prior) > MAX_TEMPLATE_COMPRESSED_BYTES:
            raise ValueError("position_prior exceeds max template compressed length")
        for value in self.position_prior:
            _require_int8("position_prior value", value)
        _require_int8("fallback_prior", self.fallback_prior)
        _require_bytes_len("weights_hash", self.weights_hash, HASH_LEN_BYTES)


def load_weights(
    path: StrPath = LEARNED_CODEC_WEIGHTS_PATH,
    *,
    expected_weights_hash: bytes | None = None,
) -> LearnedCodecWeights:
    """Load frozen INT8 learned codec weights from a deterministic `.npz` file."""

    weights_path = Path(path)
    if not weights_path.is_file():
        raise LearnedCodecError(f"learned codec weights file not found: {weights_path}")

    try:
        with np.load(weights_path, allow_pickle=False) as archive:
            if WEIGHTS_POSITION_PRIOR_KEY not in archive.files:
                raise LearnedCodecError("learned codec weights missing position_prior")
            position_prior = archive[WEIGHTS_POSITION_PRIOR_KEY]
            fallback_prior = (
                archive[WEIGHTS_FALLBACK_PRIOR_KEY]
                if WEIGHTS_FALLBACK_PRIOR_KEY in archive.files
                else np.array(INT8_MIN_VALUE, dtype=np.int8)
            )
            weights = _weights_from_arrays(position_prior, fallback_prior)
            _enforce_weights_hash(weights, expected_weights_hash)
            return weights
    except OSError as exc:
        raise LearnedCodecError("learned codec weights file is unreadable") from exc
    except ValueError as exc:
        if isinstance(exc, LearnedCodecError):
            raise
        raise LearnedCodecError("learned codec weights file is malformed") from exc


def learned_codec_available(path: StrPath = LEARNED_CODEC_WEIGHTS_PATH) -> bool:
    """Return whether the frozen learned codec weights file exists locally."""

    return Path(path).is_file()


def compress_tx(
    tx: Transaction,
    weights: LearnedCodecWeights | None = None,
    *,
    weights_path: StrPath = LEARNED_CODEC_WEIGHTS_PATH,
    expected_weights_hash: bytes | None = None,
    backend: LearnedBackend = "cpu",
) -> bytes:
    """Return a canonical `CODEC_LEARNED` transaction object."""

    if not isinstance(tx, Transaction):
        raise TypeError("tx must be Transaction")
    codec_weights = _resolve_weights(
        weights,
        weights_path=weights_path,
        expected_weights_hash=expected_weights_hash,
    )

    template_object = _compress_template_tx(tx)
    body = _compress_template_object(template_object, codec_weights, backend=backend)
    return b"".join(
        (
            CODEC_LEARNED.to_bytes(CODEC_ID_BYTES, "little"),
            len(tx.to_bytes()).to_bytes(U32_BYTES, "little"),
            len(body).to_bytes(U32_BYTES, "little"),
            body,
        )
    )


def decompress_tx(
    data: bytes,
    weights: LearnedCodecWeights | None = None,
    *,
    weights_path: StrPath = LEARNED_CODEC_WEIGHTS_PATH,
    expected_weights_hash: bytes | None = None,
    backend: LearnedBackend = "cpu",
) -> Transaction:
    """Decode a canonical learned-compressed transaction object."""

    _require_bytes("data", data, max_len=MAX_LEARNED_COMPRESSED_BYTES)
    if len(data) < COMPRESSED_OBJECT_HEADER_BYTES:
        raise LearnedCodecError("learned compressed object header is truncated")
    codec_weights = _resolve_weights(
        weights,
        weights_path=weights_path,
        expected_weights_hash=expected_weights_hash,
    )

    reader = _Reader(data)
    codec_id = int.from_bytes(reader.bytes(CODEC_ID_BYTES), "little")
    if codec_id != CODEC_LEARNED:
        raise LearnedCodecError("unsupported learned transaction codec_id")
    uncompressed_len = int.from_bytes(reader.bytes(U32_BYTES), "little")
    compressed_len = int.from_bytes(reader.bytes(U32_BYTES), "little")
    if uncompressed_len > MAX_TX_BYTES:
        raise LearnedCodecError("uncompressed transaction length exceeds MAX_TX_BYTES")
    body = reader.bytes(compressed_len)
    reader.finish()

    template_object = _decompress_template_object(body, codec_weights, backend=backend)
    try:
        tx = _decompress_template_tx(template_object)
    except TemplateCodecError as exc:
        raise LearnedCodecError("learned codec reconstructed invalid template bytes") from exc

    if len(tx.to_bytes()) != uncompressed_len:
        raise LearnedCodecError("uncompressed transaction length mismatch")
    if (
        compress_tx(
            tx,
            codec_weights,
            expected_weights_hash=codec_weights.weights_hash,
            backend=backend,
        )
        != data
    ):
        raise LearnedCodecError("learned codec bytes are non-canonical")
    return tx


def predict_template_bytes(
    length: int,
    weights: LearnedCodecWeights,
    *,
    backend: LearnedBackend = "cpu",
) -> bytes:
    """Return the deterministic INT8 prior prediction for template-compressed bytes."""

    _require_uint("length", length, MAX_TEMPLATE_COMPRESSED_BYTES)
    if not isinstance(weights, LearnedCodecWeights):
        raise TypeError("weights must be LearnedCodecWeights")
    resolved_backend = _resolve_backend(backend)
    if length == 0:
        return b""

    configure_torch_determinism()
    try:
        device = torch.device(resolved_backend)
        prior = torch.tensor(weights.position_prior, dtype=torch.int32, device=device)
        predicted = torch.full(
            (length,),
            int(weights.fallback_prior),
            dtype=torch.int32,
            device=device,
        )
        learned_len = min(length, len(weights.position_prior))
        predicted[:learned_len] = prior[:learned_len]
        values = (predicted.to("cpu", dtype=torch.int32) + INT8_ZERO_POINT).tolist()
    except RuntimeError as exc:
        raise LearnedCodecError(
            f"{resolved_backend} backend cannot run learned codec deterministically"
        ) from exc

    return bytes(_require_byte_value("predicted byte", int(value)) for value in values)


def _compress_template_object(
    template_object: bytes,
    weights: LearnedCodecWeights,
    *,
    backend: LearnedBackend,
) -> bytes:
    _require_template_object(template_object)
    predicted = predict_template_bytes(len(template_object), weights, backend=backend)
    residuals = [
        (index, actual)
        for index, (actual, expected) in enumerate(zip(template_object, predicted, strict=True))
        if actual != expected
    ]

    body = bytearray(LEARNED_CODEC_MAGIC)
    body.extend(hash_bytes(template_object))
    body.extend(_uvarint(len(template_object)))
    body.extend(_uvarint(len(residuals)))

    previous_index = -1
    for index, actual in residuals:
        body.extend(_uvarint(index - previous_index - 1))
        body.append(actual)
        previous_index = index

    return bytes(body)


def _decompress_template_object(
    body: bytes,
    weights: LearnedCodecWeights,
    *,
    backend: LearnedBackend,
) -> bytes:
    _require_bytes("body", body, max_len=MAX_LEARNED_COMPRESSED_BYTES)
    reader = _Reader(body)
    if reader.bytes(len(LEARNED_CODEC_MAGIC)) != LEARNED_CODEC_MAGIC:
        raise LearnedCodecError("learned codec magic is invalid")

    expected_hash = reader.bytes(HASH_LEN_BYTES)
    template_len = reader.uvarint(max_value=MAX_TEMPLATE_COMPRESSED_BYTES)
    if template_len < COMPRESSED_OBJECT_HEADER_BYTES:
        raise LearnedCodecError("template object length is too short")
    residual_count = reader.uvarint(max_value=template_len)

    template_object = bytearray(predict_template_bytes(template_len, weights, backend=backend))
    position = -1
    for _ in range(residual_count):
        skip = reader.uvarint(max_value=template_len)
        position += skip + 1
        if position >= template_len:
            raise LearnedCodecError("residual position exceeds template object length")
        template_object[position] = reader.u8()
    reader.finish()

    decoded = bytes(template_object)
    if hash_bytes(decoded) != expected_hash:
        raise LearnedCodecError("learned residual hash mismatch")
    _require_template_object(decoded)
    return decoded


def _require_template_object(value: bytes) -> None:
    _require_bytes("template_object", value, max_len=MAX_TEMPLATE_COMPRESSED_BYTES)
    if len(value) < COMPRESSED_OBJECT_HEADER_BYTES:
        raise LearnedCodecError("template object header is truncated")
    codec_id = int.from_bytes(value[:CODEC_ID_BYTES], "little")
    if codec_id != CODEC_TEMPLATE_RANGE:
        raise LearnedCodecError("learned codec requires a template-range transaction object")


def _weights_from_arrays(
    position_prior: np.ndarray[tuple[int, ...], np.dtype[np.int8]],
    fallback_prior: np.ndarray[tuple[int, ...], np.dtype[np.int8]],
) -> LearnedCodecWeights:
    if position_prior.dtype != np.int8:
        raise LearnedCodecError("position_prior must have dtype int8")
    if position_prior.ndim != 1:
        raise LearnedCodecError("position_prior must be one-dimensional")
    if position_prior.size == 0:
        raise LearnedCodecError("position_prior must not be empty")
    if position_prior.size > MAX_TEMPLATE_COMPRESSED_BYTES:
        raise LearnedCodecError("position_prior exceeds max template compressed length")
    if fallback_prior.dtype != np.int8:
        raise LearnedCodecError("fallback_prior must have dtype int8")
    if fallback_prior.size != 1:
        raise LearnedCodecError("fallback_prior must be scalar int8")

    contiguous_prior = np.ascontiguousarray(position_prior)
    position_tuple = tuple(int(value) for value in contiguous_prior.tolist())
    fallback_value = int(np.ravel(fallback_prior)[0])
    canonical = b"".join(
        (
            WEIGHTS_HASH_TAG,
            _uvarint(len(position_tuple)),
            contiguous_prior.tobytes(order="C"),
            fallback_value.to_bytes(U8_BYTES, "little", signed=True),
        )
    )
    return LearnedCodecWeights(
        position_prior=position_tuple,
        fallback_prior=fallback_value,
        weights_hash=hash_bytes(canonical),
    )


def _resolve_weights(
    weights: LearnedCodecWeights | None,
    *,
    weights_path: StrPath,
    expected_weights_hash: bytes | None,
) -> LearnedCodecWeights:
    expected_hash = (
        LEARNED_CODEC_WEIGHTS_HASH if expected_weights_hash is None else expected_weights_hash
    )
    if weights is None:
        return load_weights(weights_path, expected_weights_hash=expected_hash)
    if not isinstance(weights, LearnedCodecWeights):
        raise TypeError("weights must be LearnedCodecWeights")
    _enforce_weights_hash(weights, expected_hash)
    return weights


def _enforce_weights_hash(
    weights: LearnedCodecWeights,
    expected_weights_hash: bytes | None,
) -> None:
    if expected_weights_hash is None:
        return
    _require_bytes_len("expected_weights_hash", expected_weights_hash, HASH_LEN_BYTES)
    if expected_weights_hash == LEARNED_CODEC_DISABLED_HASH:
        raise LearnedCodecError("learned codec weights are disabled by consensus hash")
    if weights.weights_hash != expected_weights_hash:
        raise LearnedCodecError("learned codec weights hash mismatch")


@dataclass(slots=True)
class _Reader:
    data: bytes
    offset: int = 0

    def __post_init__(self) -> None:
        _require_bytes("data", self.data)

    def bytes(self, length: int) -> bytes:
        _require_uint("length", length, MAX_LEARNED_COMPRESSED_BYTES)
        end = self.offset + length
        if end > len(self.data):
            raise LearnedCodecError("learned codec bytes are truncated")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def u8(self) -> int:
        return self.bytes(U8_BYTES)[0]

    def uvarint(self, *, max_value: int = U64_MAX) -> int:
        value, shift = 0, 0
        for index in range(10):
            byte = self.u8()
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                if index > 0 and value < (1 << (7 * index)):
                    raise LearnedCodecError("non-canonical varint")
                if value > max_value:
                    raise LearnedCodecError("varint exceeds maximum")
                return value
            shift += 7
        raise LearnedCodecError("varint is too long")

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise LearnedCodecError("trailing learned codec bytes")


def _resolve_backend(backend: LearnedBackend) -> Literal["cpu", "cuda", "mps"]:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"unknown learned codec backend {backend!r}")
    if backend == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if backend == "cuda" and not torch.cuda.is_available():
        raise LearnedCodecError("CUDA backend requested but CUDA is unavailable")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise LearnedCodecError("MPS backend requested but MPS is unavailable")
    return backend


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


def _require_int8(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not INT8_MIN_VALUE <= value <= INT8_MAX_VALUE:
        raise ValueError(f"{name} outside int8 range")


def _require_byte_value(name: str, value: int) -> int:
    if not 0 <= value < BYTE_VALUES:
        raise LearnedCodecError(f"{name} outside byte range")
    return value


def _require_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} outside uint range")


def _require_bytes(name: str, value: bytes, *, max_len: int | None = None) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if max_len is not None and len(value) > max_len:
        raise ValueError(f"{name} exceeds max length")


def _require_bytes_len(name: str, value: bytes, expected_len: int) -> None:
    _require_bytes(name, value)
    if len(value) != expected_len:
        raise ValueError(f"{name} must be {expected_len} bytes")


__all__ = [
    "CODEC_LEARNED",
    "INT8_ZERO_POINT",
    "LEARNED_CODEC_DISABLED_HASH",
    "LEARNED_CODEC_EXTRA_COMPRESSION_PCT",
    "LEARNED_CODEC_INT8_ZERO_POINT",
    "LEARNED_CODEC_MAGIC",
    "LEARNED_CODEC_WEIGHTS_HASH",
    "LEARNED_CODEC_WEIGHTS_PATH",
    "MAX_LEARNED_COMPRESSED_BYTES",
    "LearnedBackend",
    "LearnedCodecError",
    "LearnedCodecWeights",
    "compress_tx",
    "decompress_tx",
    "learned_codec_available",
    "load_weights",
    "predict_template_bytes",
]
