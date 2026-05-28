"""Tests for deterministic learned transaction residual compression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tensorpow.codec.learned import (
    CODEC_LEARNED,
    INT8_ZERO_POINT,
    LEARNED_CODEC_DISABLED_HASH,
    LEARNED_CODEC_INT8_ZERO_POINT,
    LEARNED_CODEC_MAGIC,
    LEARNED_CODEC_WEIGHTS_HASH,
    LearnedCodecError,
    LearnedCodecWeights,
    learned_codec_available,
    load_weights,
    predict_template_bytes,
)
from tensorpow.codec.learned import (
    compress_tx as compress_learned_tx,
)
from tensorpow.codec.learned import (
    decompress_tx as decompress_learned_tx,
)
from tensorpow.codec.template import (
    COMPRESSED_OBJECT_HEADER_BYTES,
)
from tensorpow.codec.template import (
    compress_tx as compress_template_tx,
)
from tensorpow.crypto.hash import hash_bytes
from tensorpow.crypto.signatures import ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES
from tensorpow.state.utxo import TEMPLATE_PKH, Outpoint
from tensorpow.tx.script import pubkey_hash
from tensorpow.tx.transaction import FORMAT_EPOCH, MAX_TX_BYTES, Input, Output, Transaction

ZERO_SIGNATURE = bytes(ED25519_SIGNATURE_BYTES)
ZERO_PUBLIC_KEY = bytes(ED25519_PUBLIC_KEY_BYTES)


@pytest.fixture
def zero_prior_weights(tmp_path: Path) -> LearnedCodecWeights:
    weights_path = tmp_path / "learned_codec.npz"
    np.savez(
        weights_path,
        position_prior=np.full(512, -INT8_ZERO_POINT, dtype=np.int8),
        fallback_prior=np.array(-INT8_ZERO_POINT, dtype=np.int8),
    )
    return load_weights(weights_path)


def test_learned_codec_round_trips_and_compresses_fixture(
    tmp_path: Path,
) -> None:
    tx = _zero_heavy_tx()
    template_object = compress_template_tx(tx)
    weights = _exact_template_weights(tmp_path, template_object)
    learned_object = compress_learned_tx(tx, weights, expected_weights_hash=weights.weights_hash)

    assert int.from_bytes(learned_object[:2], "little") == CODEC_LEARNED
    assert (
        decompress_learned_tx(
            learned_object,
            weights,
            expected_weights_hash=weights.weights_hash,
        ).to_bytes()
        == tx.to_bytes()
    )
    assert (
        compress_learned_tx(
            decompress_learned_tx(
                learned_object,
                weights,
                expected_weights_hash=weights.weights_hash,
            ),
            weights,
            expected_weights_hash=weights.weights_hash,
        )
        == learned_object
    )
    assert len(learned_object) * 100 <= len(template_object) * 85


def test_learned_codec_loads_int8_weights_and_predicts_deterministically(tmp_path: Path) -> None:
    weights_path = tmp_path / "learned_codec.npz"
    np.savez(
        weights_path,
        position_prior=np.array([-128, -127, 0, 127], dtype=np.int8),
        fallback_prior=np.array(-128, dtype=np.int8),
    )

    assert learned_codec_available(weights_path)
    assert not learned_codec_available(tmp_path / "missing.npz")
    weights = load_weights(weights_path)

    assert INT8_ZERO_POINT == LEARNED_CODEC_INT8_ZERO_POINT == 128
    assert LEARNED_CODEC_DISABLED_HASH == LEARNED_CODEC_WEIGHTS_HASH == bytes(32)
    assert weights.weights_hash != LEARNED_CODEC_DISABLED_HASH
    assert predict_template_bytes(6, weights) == bytes((0, 1, 128, 255, 0, 0))
    assert predict_template_bytes(6, weights) == predict_template_bytes(6, weights)


def test_learned_codec_enforces_expected_frozen_weights_hash(tmp_path: Path) -> None:
    weights_path = tmp_path / "learned_codec.npz"
    np.savez(
        weights_path,
        position_prior=np.full(512, -INT8_ZERO_POINT, dtype=np.int8),
        fallback_prior=np.array(-INT8_ZERO_POINT, dtype=np.int8),
    )
    tx = _zero_heavy_tx()
    weights = load_weights(weights_path)

    assert load_weights(weights_path, expected_weights_hash=weights.weights_hash) == weights
    encoded = compress_learned_tx(
        tx,
        weights_path=weights_path,
        expected_weights_hash=weights.weights_hash,
    )
    assert (
        decompress_learned_tx(
            encoded,
            weights_path=weights_path,
            expected_weights_hash=weights.weights_hash,
        )
        == tx
    )

    with pytest.raises(LearnedCodecError, match="disabled"):
        compress_learned_tx(tx, weights_path=weights_path)
    with pytest.raises(LearnedCodecError, match="disabled"):
        compress_learned_tx(tx, weights)
    with pytest.raises(LearnedCodecError, match="mismatch"):
        load_weights(weights_path, expected_weights_hash=hash_bytes(b"wrong weights"))


def test_learned_codec_is_deterministic_across_available_torch_backends(
    zero_prior_weights: LearnedCodecWeights,
) -> None:
    tx = _zero_heavy_tx()
    expected = compress_learned_tx(
        tx,
        zero_prior_weights,
        expected_weights_hash=zero_prior_weights.weights_hash,
        backend="cpu",
    )
    backends = ["auto", "cpu"]
    if torch.cuda.is_available():
        backends.append("cuda")
    if torch.backends.mps.is_available():
        backends.append("mps")

    for backend in backends:
        encoded = compress_learned_tx(
            tx,
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
            backend=backend,
        )
        assert encoded == expected
        assert (
            decompress_learned_tx(
                encoded,
                zero_prior_weights,
                expected_weights_hash=zero_prior_weights.weights_hash,
                backend=backend,
            )
            == tx
        )


def test_learned_codec_rejects_malformed_headers_and_lengths(
    zero_prior_weights: LearnedCodecWeights,
) -> None:
    encoded = compress_learned_tx(
        _zero_heavy_tx(),
        zero_prior_weights,
        expected_weights_hash=zero_prior_weights.weights_hash,
    )

    with pytest.raises(LearnedCodecError, match="header"):
        decompress_learned_tx(
            encoded[:5],
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )

    bad_codec = bytearray(encoded)
    bad_codec[:2] = (0).to_bytes(2, "little")
    with pytest.raises(LearnedCodecError, match="codec_id"):
        decompress_learned_tx(
            bytes(bad_codec),
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )

    bad_uncompressed_len = bytearray(encoded)
    bad_uncompressed_len[2:6] = (MAX_TX_BYTES + 1).to_bytes(4, "little")
    with pytest.raises(LearnedCodecError, match="MAX_TX_BYTES"):
        decompress_learned_tx(
            bytes(bad_uncompressed_len),
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )

    with pytest.raises(LearnedCodecError, match="truncated"):
        decompress_learned_tx(
            encoded[:-1],
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )
    with pytest.raises(LearnedCodecError, match="trailing"):
        decompress_learned_tx(
            encoded + b"\x00",
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )


def test_learned_codec_rejects_corrupt_and_noncanonical_bodies(
    zero_prior_weights: LearnedCodecWeights,
) -> None:
    tx = _zero_heavy_tx()
    encoded = compress_learned_tx(
        tx,
        zero_prior_weights,
        expected_weights_hash=zero_prior_weights.weights_hash,
    )
    body = encoded[COMPRESSED_OBJECT_HEADER_BYTES:]

    bad_magic = _object_with_body(encoded, b"BAD!" + body[len(LEARNED_CODEC_MAGIC) :])
    with pytest.raises(LearnedCodecError, match="magic"):
        decompress_learned_tx(
            bad_magic,
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )

    bad_hash = bytearray(body)
    bad_hash[len(LEARNED_CODEC_MAGIC)] ^= 1
    with pytest.raises(LearnedCodecError, match="hash"):
        decompress_learned_tx(
            _object_with_body(encoded, bytes(bad_hash)),
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )

    template_object = compress_template_tx(tx)
    bad_position_body = b"".join(
        (
            LEARNED_CODEC_MAGIC,
            hash_bytes(template_object),
            _uvarint(len(template_object)),
            _uvarint(1),
            _uvarint(len(template_object)),
            b"\x00",
        )
    )
    with pytest.raises(LearnedCodecError, match="position"):
        decompress_learned_tx(
            _object_with_body(encoded, bad_position_body),
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )

    noncanonical_body = _body_with_extra_zero_residual(template_object, zero_prior_weights)
    with pytest.raises(LearnedCodecError, match="non-canonical"):
        decompress_learned_tx(
            _object_with_body(encoded, noncanonical_body),
            zero_prior_weights,
            expected_weights_hash=zero_prior_weights.weights_hash,
        )


def test_learned_codec_rejects_malformed_weight_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    with pytest.raises(LearnedCodecError, match="not found"):
        load_weights(missing)

    wrong_dtype = tmp_path / "wrong_dtype.npz"
    np.savez(
        wrong_dtype,
        position_prior=np.zeros(4, dtype=np.int16),
        fallback_prior=np.array(-128, dtype=np.int8),
    )
    with pytest.raises(LearnedCodecError, match="dtype int8"):
        load_weights(wrong_dtype)

    missing_prior = tmp_path / "missing_prior.npz"
    np.savez(missing_prior, fallback_prior=np.array(-128, dtype=np.int8))
    with pytest.raises(LearnedCodecError, match="position_prior"):
        load_weights(missing_prior)


def _zero_heavy_tx() -> Transaction:
    return Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=0,
        lockheight=0,
        inputs=(
            Input(
                Outpoint(bytes(32), 0),
                witness=ZERO_SIGNATURE + ZERO_PUBLIC_KEY,
            ),
        ),
        outputs=(
            Output(
                amount_matoms=1,
                template_id=TEMPLATE_PKH,
                payload=pubkey_hash(ZERO_PUBLIC_KEY),
            ),
        ),
    )


def _exact_template_weights(tmp_path: Path, template_object: bytes) -> LearnedCodecWeights:
    weights_path = tmp_path / "exact_learned_codec.npz"
    np.savez(
        weights_path,
        position_prior=np.array(
            [byte - INT8_ZERO_POINT for byte in template_object],
            dtype=np.int8,
        ),
        fallback_prior=np.array(-INT8_ZERO_POINT, dtype=np.int8),
    )
    return load_weights(weights_path)


def _body_with_extra_zero_residual(
    template_object: bytes,
    weights: LearnedCodecWeights,
) -> bytes:
    predicted = predict_template_bytes(len(template_object), weights)
    residuals = [
        (index, actual)
        for index, (actual, expected) in enumerate(zip(template_object, predicted, strict=True))
        if actual != expected
    ]
    residuals.append((1, template_object[1]))
    residuals.sort()

    body = bytearray(LEARNED_CODEC_MAGIC)
    body.extend(hash_bytes(template_object))
    body.extend(_uvarint(len(template_object)))
    body.extend(_uvarint(len(residuals)))
    previous_index = -1
    for index, value in residuals:
        body.extend(_uvarint(index - previous_index - 1))
        body.append(value)
        previous_index = index
    return bytes(body)


def _object_with_body(original: bytes, body: bytes) -> bytes:
    return b"".join(
        (
            original[:2],
            original[2:6],
            len(body).to_bytes(4, "little"),
            body,
        )
    )


def _uvarint(value: int) -> bytes:
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
