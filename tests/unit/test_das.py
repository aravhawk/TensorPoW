"""Tests for DAS cellization, sampling, and proof verification."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tensorpow.crypto.hash import hash_bytes
from tensorpow.net.das import (
    DAS_CELL_BYTES,
    DAS_RS_EXTENSION_FACTOR,
    DAS_SAMPLE_SUCCESS_THRESHOLD_PCT,
    DAS_SAMPLES_PER_FRUIT,
    DAS_WITHHOLDING_DETECTION_PCT,
    DAS_WITHHOLDING_PCT,
    DOMAIN_DAS_SAMPLE,
    DASMerkleProof,
    DASSampleRequest,
    availability_confidence_pct,
    create_sample_proof,
    decode_sample_request,
    decode_sample_response,
    encode_payload,
    encode_sample_request,
    encode_sample_response,
    is_available,
    select_sample_requests,
    verify_sample,
)


def test_encode_payload_is_bit_exact_and_reed_solomon_extended() -> None:
    payload = bytes(range(DAS_CELL_BYTES)) + b"TensorPoW DAS vector"
    encoding = encode_payload(payload)

    assert DOMAIN_DAS_SAMPLE == 0x70
    assert DAS_RS_EXTENSION_FACTOR == 2
    assert encoding.commitment.payload_length == 276
    assert encoding.commitment.data_side == 2
    assert encoding.commitment.extended_side == 4
    assert len(encoding.cells) == 16
    assert (
        encoding.commitment.cell_root.hex()
        == "e50af87408105860efc94a5b66bf823f86d18227afa55a630e56ac86c60f22ee"
    )
    assert (
        encoding.commitment.root.hex()
        == "0cc9487c8e3dda70b4deb04117c8d5912462f0ca73f3a4e7c0e124c596a3216b"
    )

    assert encoding.cells[0] == bytes(range(DAS_CELL_BYTES))
    assert encoding.cells[1] == b"TensorPoW DAS vector".ljust(DAS_CELL_BYTES, b"\x00")
    assert encoding.cells[2][:16].hex() == "fca8bc9cad8de2a4c15ffaf2d143b082"
    assert encoding.cells[2][-16:].hex() == "eaede4e3f6f1f8ffd2d5dcdbcec9c0c7"
    assert encoding.cells[-1][:16].hex() == "d792da52ae269f9263299fb703798a4a"
    assert encoding.cells[-1][-16:].hex() == "5c4874600c182430fce8d4c0acb88490"


def test_sample_selection_and_proof_verification_are_deterministic() -> None:
    payload = bytes(range(DAS_CELL_BYTES)) + b"TensorPoW DAS vector"
    fruit_hash = hash_bytes(b"das-test-fruit")
    encoding = encode_payload(payload)

    requests = select_sample_requests(fruit_hash, encoding.commitment)
    assert [(request.sample_index, request.row, request.column) for request in requests] == [
        (0, 0, 3),
        (1, 2, 2),
        (2, 0, 1),
        (3, 1, 3),
        (4, 0, 0),
        (5, 3, 3),
        (6, 1, 0),
        (7, 2, 2),
        (8, 1, 0),
        (9, 2, 3),
    ]

    proof = create_sample_proof(encoding, requests[3])
    assert verify_sample(requests[3], encoding.commitment, proof)


def test_das_wire_request_and_response_codecs_round_trip() -> None:
    payload = bytes(range(DAS_CELL_BYTES)) + b"TensorPoW DAS vector"
    fruit_hash = hash_bytes(b"das-wire-fruit")
    encoding = encode_payload(payload)
    request = select_sample_requests(fruit_hash, encoding.commitment)[0]
    proof = create_sample_proof(encoding, request)

    request_bytes = encode_sample_request(request)
    response_bytes = encode_sample_response(proof)

    assert decode_sample_request(request_bytes) == request
    decoded_proof = decode_sample_response(response_bytes)
    assert decoded_proof == proof
    assert verify_sample(request, encoding.commitment, decoded_proof)


def test_das_wire_codecs_reject_malformed_payloads() -> None:
    encoding = encode_payload(bytes(range(DAS_CELL_BYTES)))
    request = select_sample_requests(hash_bytes(b"das-wire-bad"), encoding.commitment)[0]
    proof = create_sample_proof(encoding, request)
    response_bytes = encode_sample_response(proof)

    with pytest.raises(ValueError, match="request length"):
        decode_sample_request(encode_sample_request(request)[:-1])
    with pytest.raises(ValueError, match="trailing"):
        decode_sample_response(response_bytes + b"\x00")
    with pytest.raises(ValueError, match="truncated"):
        decode_sample_response(response_bytes[:-1])

    bad_row_count = bytearray(response_bytes)
    bad_row_count[8:12] = (0).to_bytes(4, "little")
    with pytest.raises(ValueError, match="positive"):
        decode_sample_response(bytes(bad_row_count))

    with pytest.raises(ValueError, match="uint32"):
        DASSampleRequest(
            fruit_hash=hash_bytes(b"das-overflow"),
            sample_index=0,
            row=2**32,
            column=0,
        )


def test_tampered_samples_and_proofs_are_rejected() -> None:
    encoding = encode_payload(bytes(range(DAS_CELL_BYTES)) + b"TensorPoW DAS vector")
    request = select_sample_requests(hash_bytes(b"das-test-fruit"), encoding.commitment)[3]
    proof = create_sample_proof(encoding, request)

    tampered_cell = (
        bytes([proof.row_cells[request.column][0] ^ 0x01]) + proof.row_cells[request.column][1:]
    )
    row_cells = list(proof.row_cells)
    row_cells[request.column] = tampered_cell
    assert not verify_sample(
        request, encoding.commitment, replace(proof, row_cells=tuple(row_cells))
    )

    first_proof = proof.row_proofs[0]
    bad_sibling = replace(first_proof.siblings[0], digest=bytes([0xFF]) * 32)
    bad_row_proofs = list(proof.row_proofs)
    bad_row_proofs[0] = DASMerkleProof(
        leaf_index=first_proof.leaf_index,
        leaf_count=first_proof.leaf_count,
        siblings=(bad_sibling, *first_proof.siblings[1:]),
    )
    assert not verify_sample(
        request,
        encoding.commitment,
        replace(proof, row_proofs=tuple(bad_row_proofs)),
    )

    bad_request = replace(request, row=(request.row + 1) % encoding.commitment.extended_side)
    assert not verify_sample(bad_request, encoding.commitment, proof)


def test_light_node_availability_decision_uses_threshold() -> None:
    assert DAS_SAMPLE_SUCCESS_THRESHOLD_PCT == 75
    assert DAS_SAMPLES_PER_FRUIT == 10
    assert is_available((True,) * 8 + (False,) * 2)
    assert not is_available((True,) * 7 + (False,) * 3)
    assert not is_available(())


def test_withholding_detection_confidence_exceeds_protocol_target() -> None:
    assert DAS_WITHHOLDING_PCT == 50
    assert availability_confidence_pct() == 99.90234375
    assert availability_confidence_pct() > DAS_WITHHOLDING_DETECTION_PCT

    encoding = encode_payload(bytes(range(DAS_CELL_BYTES)) * 5)
    side = encoding.commitment.extended_side
    withheld_offsets = set(range(len(encoding.cells) // 2))

    trials = 1024
    detected = 0
    for trial in range(trials):
        fruit_hash = hash_bytes(b"das-withholding-regression" + trial.to_bytes(4, "little"))
        requests = select_sample_requests(fruit_hash, encoding.commitment)
        if any((request.row * side + request.column) in withheld_offsets for request in requests):
            detected += 1

    detection_pct = detected * 100 / trials
    assert detection_pct > DAS_WITHHOLDING_DETECTION_PCT
