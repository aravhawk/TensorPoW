"""Tests for deterministic template transaction compression."""

from __future__ import annotations

import pytest

import tensorpow.codec.template as template_codec
from tensorpow.codec.template import (
    CODEC_TEMPLATE_RANGE,
    COMPRESSED_OBJECT_HEADER_BYTES,
    TEMPLATE_CODEC_MAGIC,
    TEMPLATE_RANGE_CODER_ADAPTIVE,
    TemplateCodecError,
    compress_tx,
    decompress_tx,
)
from tensorpow.crypto.hash import hash_bytes
from tensorpow.state.utxo import TEMPLATE_HASHLOCK, TEMPLATE_MULTISIG, TEMPLATE_PKH, Outpoint
from tensorpow.tx.transaction import FORMAT_EPOCH, Input, Output, Transaction

MESSAGE_SIGNATURE = bytes.fromhex(
    "e29ead045bb9a506ae2cff64ebccf620afebbdfff03339ae433b8af371a2331ee"
    "4ea4e5086a835ae05420fd45f7566972849f6791dba94f05929b7645c68ce03"
)
PUBLIC_KEY_1 = bytes.fromhex("343010a1aba8774dd1e6f4f0c3349bae6824908a1e64cd638dc2ed1bc625af1d")
PUBLIC_KEY_2 = bytes.fromhex("ad5e8be40705840112d69a3fcd1603c2974a356878b4491744a6d62ae5c72a43")
PKH_1 = bytes.fromhex("ce0d4ef3ea76c782c6ca2b6368b11c3509324d43d58c76511d2bac9c7a32e650")
PKH_2 = bytes.fromhex("dfadc17a9f2c11c59bc3e629b8368bf0632868ec14f3fedbe5fbde109284cbf4")

REFERENCE_COMPRESSED_TX_HEX = (
    "010019010000b80000005450544301af000000000201f61fa0655b39d9040a791329c87d"
    "a1f95993f1a0cbebcbaa65c3ca5ee0492db9a4e2ac01cb20dd5dde15be280c29e591b"
    "51478945e9dc44c9797ed73c968b51cc658352356aa03e8acc0bd055d8b0b89e7e806"
    "ee31237c48d157104f24f3316a1ff4e7ada2305cdc7257302812bc3f0aacc76ce7038a"
    "bea2be6a60511a6a4f0810828d28b29fb08bc90cd7b5ad8924babde6833a5feb884850"
    "61ee9ee249f9a52ec334c2f16335d06d72e216"
)


def _outpoint(index: int = 7) -> Outpoint:
    return Outpoint(bytes(range(32)), index)


def _typical_pkh_tx() -> Transaction:
    return Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=0,
        lockheight=0,
        inputs=(Input(_outpoint(), witness=MESSAGE_SIGNATURE + PUBLIC_KEY_1),),
        outputs=(
            Output(123, TEMPLATE_PKH, payload=PKH_2),
            Output(45, TEMPLATE_PKH, payload=PKH_1),
        ),
    )


def test_template_codec_round_trip_reference_vector_and_compression_ratio() -> None:
    tx = _typical_pkh_tx()
    compressed = compress_tx(tx)

    assert compressed.hex() == REFERENCE_COMPRESSED_TX_HEX
    assert decompress_tx(compressed).to_bytes() == tx.to_bytes()
    assert compress_tx(decompress_tx(compressed)) == compressed
    assert len(compressed) <= len(tx.to_bytes()) * 70 // 100
    assert compressed[COMPRESSED_OBJECT_HEADER_BYTES : COMPRESSED_OBJECT_HEADER_BYTES + 4] == (
        TEMPLATE_CODEC_MAGIC
    )
    assert compressed[COMPRESSED_OBJECT_HEADER_BYTES + 4] == TEMPLATE_RANGE_CODER_ADAPTIVE


def test_template_codec_preserves_nonzero_locks_and_non_pkh_templates() -> None:
    multisig_payload = bytes((1, 2)) + PUBLIC_KEY_1 + PUBLIC_KEY_2
    hashlock_payload = hash_bytes(b"preimage") + multisig_payload
    tx = Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=10,
        lockheight=11,
        inputs=(Input(_outpoint(1), witness=b"raw witness"),),
        outputs=(
            Output(50, TEMPLATE_MULTISIG, locktime_ms=12, payload=multisig_payload),
            Output(51, TEMPLATE_HASHLOCK, lockheight=13, payload=hashlock_payload),
        ),
    )

    assert decompress_tx(compress_tx(tx)) == tx


def test_template_codec_preserves_hashlock_over_pkh_and_multisig() -> None:
    multisig_payload = bytes((2, 2)) + PUBLIC_KEY_1 + PUBLIC_KEY_2
    for inner_payload in (PKH_1, multisig_payload):
        tx = Transaction(
            version=FORMAT_EPOCH,
            sig_type=0,
            locktime_ms=0,
            lockheight=0,
            inputs=(Input(_outpoint(2), witness=b"raw witness"),),
            outputs=(
                Output(
                    52,
                    TEMPLATE_HASHLOCK,
                    payload=hash_bytes(b"preimage") + inner_payload,
                ),
            ),
        )

        assert decompress_tx(compress_tx(tx)) == tx


def test_template_codec_rejects_truncated_unknown_and_length_mismatched_objects() -> None:
    compressed = compress_tx(_typical_pkh_tx())

    with pytest.raises(TemplateCodecError, match="header"):
        decompress_tx(compressed[:5])
    bad_codec = bytearray(compressed)
    bad_codec[0:2] = (0).to_bytes(2, "little")
    with pytest.raises(TemplateCodecError, match="codec_id"):
        decompress_tx(bytes(bad_codec))
    bad_uncompressed_len = bytearray(compressed)
    bad_uncompressed_len[2:6] = (1).to_bytes(4, "little")
    with pytest.raises(TemplateCodecError, match="length mismatch"):
        decompress_tx(bytes(bad_uncompressed_len))
    with pytest.raises(TemplateCodecError, match="truncated"):
        decompress_tx(compressed[:-1])
    with pytest.raises(TemplateCodecError, match="trailing"):
        decompress_tx(compressed + b"\x00")


def test_template_codec_rejects_noncanonical_and_malformed_body_values() -> None:
    compressed = compress_tx(_typical_pkh_tx())
    body = compressed[COMPRESSED_OBJECT_HEADER_BYTES:]

    bad_magic = _object_with_body(compressed, b"BAD!" + body[4:])
    with pytest.raises(TemplateCodecError, match="magic"):
        decompress_tx(bad_magic)

    bad_coder = _object_with_body(
        compressed,
        TEMPLATE_CODEC_MAGIC + b"\xff" + body[len(TEMPLATE_CODEC_MAGIC) + 1 :],
    )
    with pytest.raises(TemplateCodecError, match="range coder"):
        decompress_tx(bad_coder)

    template_payload = _template_payload_from_object(compressed)

    bad_flags = _object_with_template_payload(compressed, b"\x80" + template_payload[1:])
    with pytest.raises(TemplateCodecError, match="flags"):
        decompress_tx(bad_flags)

    bad_input_count = _object_with_template_payload(
        compressed,
        template_payload[:1] + b"\x81\x00" + template_payload[2:],
    )
    with pytest.raises(TemplateCodecError, match="varint"):
        decompress_tx(bad_input_count)

    bad_ref = bytearray(template_payload)
    bad_ref[-2] = 5
    with pytest.raises(TemplateCodecError, match="reference"):
        decompress_tx(_object_with_template_payload(compressed, bytes(bad_ref)))


def test_template_codec_rejects_noncanonical_direct_signer_hash_output() -> None:
    tx = _typical_pkh_tx()
    compressed = compress_tx(tx)
    template_payload = _template_payload_from_object(compressed)
    noncanonical_payload = template_payload[:-3] + bytes((0,)) + PKH_1 + template_payload[-1:]

    with pytest.raises(TemplateCodecError, match="non-canonical"):
        decompress_tx(_object_with_template_payload(compressed, noncanonical_payload))


def test_template_codec_wraps_transaction_construction_errors() -> None:
    compressed = compress_tx(_typical_pkh_tx())
    zero_amount_payload = b"".join(
        (
            b"\x00",
            b"\x00",
            b"\x01",
            b"\x00",
            bytes((template_codec.OUT_PKH_DIRECT,)),
            PKH_1,
            b"\x00",
        )
    )

    with pytest.raises(TemplateCodecError, match="payload"):
        decompress_tx(_object_with_template_payload(compressed, zero_amount_payload))


def test_template_codec_header_is_consistent_with_lengths() -> None:
    tx = _typical_pkh_tx()
    compressed = compress_tx(tx)

    assert int.from_bytes(compressed[:2], "little") == CODEC_TEMPLATE_RANGE
    assert int.from_bytes(compressed[2:6], "little") == len(tx.to_bytes())
    assert int.from_bytes(compressed[6:10], "little") == len(
        compressed[COMPRESSED_OBJECT_HEADER_BYTES:]
    )


def _object_with_body(original: bytes, body: bytes) -> bytes:
    return b"".join(
        (
            original[:2],
            original[2:6],
            len(body).to_bytes(4, "little"),
            body,
        )
    )


def _template_payload_from_object(compressed: bytes) -> bytes:
    body = compressed[COMPRESSED_OBJECT_HEADER_BYTES:]
    assert body[: len(TEMPLATE_CODEC_MAGIC)] == TEMPLATE_CODEC_MAGIC
    assert body[len(TEMPLATE_CODEC_MAGIC)] == TEMPLATE_RANGE_CODER_ADAPTIVE
    payload_len_offset = len(TEMPLATE_CODEC_MAGIC) + 1
    payload_len = int.from_bytes(
        body[payload_len_offset : payload_len_offset + 4],
        "little",
    )
    return template_codec._range_decode(body[payload_len_offset + 4 :], payload_len)


def _object_with_template_payload(original: bytes, template_payload: bytes) -> bytes:
    body = b"".join(
        (
            TEMPLATE_CODEC_MAGIC,
            bytes((TEMPLATE_RANGE_CODER_ADAPTIVE,)),
            len(template_payload).to_bytes(4, "little"),
            template_codec._range_encode(template_payload),
        )
    )
    return _object_with_body(original, body)
