"""Tests for UTXO state commitments."""

from __future__ import annotations

import pytest

from tensorpow.state.utxo import (
    MAX_SUPPLY_MATOMS,
    TEMPLATE_PKH,
    TX_OUTPUT_PAYLOAD_MAX_BYTES,
    UTXO,
    Outpoint,
    UTXOInclusionProof,
    UTXONonInclusionProof,
    UTXOSet,
)

OUTPOINT_BYTES_HEX = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f07000000"
OUTPOINT_KEY_HEX = "d967219c6014d6c162e37f730ec06319764dd381371004d9bdf688aa44a8ebb1"
UTXO_BYTES_HEX = (
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f07000000"
    "7b000000000000000000090909090909090909090909090909090909090909090909090909"
    "09090909090a000000000000000b0000000000000007007061796c6f6164"
)
UTXO_VALUE_HASH_HEX = "a0e7c9777010e5ee06b7f4cade2730178d56164b8f1ec4a647e7aef5ba3fec7d"
ROOT_ONE_HEX = "1ad4df4dec67df2040be56e7328df18efb6ef196f095573a44b4b549e775c23f"
ROOT_TWO_HEX = "6199b344560647cb2f6c89b256a9fbbd6595edca33a2f9706197af49c19bb5df"


def _outpoint() -> Outpoint:
    return Outpoint(bytes(range(32)), 7)


def _utxo() -> UTXO:
    return UTXO(
        outpoint=_outpoint(),
        amount_matoms=123,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=bytes([9]) * 32,
        locktime_ms=10,
        lockheight=11,
        payload=b"payload",
    )


def _other_utxo() -> UTXO:
    return UTXO(
        outpoint=Outpoint(bytes([1]) * 32, 0),
        amount_matoms=456,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=bytes([8]) * 32,
    )


def test_outpoint_and_utxo_serialization_match_reference_vectors() -> None:
    outpoint = _outpoint()
    utxo = _utxo()

    assert outpoint.to_bytes().hex() == OUTPOINT_BYTES_HEX
    assert Outpoint.from_bytes(outpoint.to_bytes()) == outpoint
    assert outpoint.key().hex() == OUTPOINT_KEY_HEX
    assert utxo.to_bytes().hex() == UTXO_BYTES_HEX
    assert UTXO.from_bytes(utxo.to_bytes(), expected_outpoint_key=outpoint.key()) == utxo
    assert utxo.value_hash().hex() == UTXO_VALUE_HASH_HEX


def test_utxo_set_add_remove_roots_and_order_independence() -> None:
    utxo = _utxo()
    other = _other_utxo()
    utxo_set = UTXOSet()

    assert not utxo_set.contains(utxo.outpoint)
    utxo_set.add(utxo)
    assert utxo_set.contains(utxo.outpoint)
    assert utxo_set.get(utxo.outpoint) == utxo
    assert utxo_set.merkle_root().hex() == ROOT_ONE_HEX
    utxo_set.add(other)
    assert utxo_set.merkle_root().hex() == ROOT_TWO_HEX
    assert UTXOSet([other, utxo]).merkle_root() == utxo_set.merkle_root()
    assert utxo_set.remove(utxo.outpoint) == utxo
    assert not utxo_set.contains(utxo.outpoint)


def test_utxo_inclusion_and_non_inclusion_proofs_verify_and_tampering_fails() -> None:
    utxo = _utxo()
    utxo_set = UTXOSet([utxo, _other_utxo()])
    root = utxo_set.merkle_root()
    inclusion = utxo_set.inclusion_proof(utxo.outpoint)
    missing = Outpoint(bytes([2]) * 32, 2)
    absence = utxo_set.non_inclusion_proof(missing)

    assert UTXOSet.verify_proof(utxo.outpoint, inclusion, root)
    assert UTXOSet.verify_proof(missing, absence, root)
    assert absence.siblings
    assert not UTXOSet.verify_proof(utxo.outpoint, inclusion, bytes(32))
    bad_inclusion = UTXOInclusionProof(
        outpoint=utxo.outpoint,
        utxo=UTXO(
            outpoint=utxo.outpoint,
            amount_matoms=124,
            template_id=TEMPLATE_PKH,
            owner_pubkey_hash=bytes([9]) * 32,
        ),
        siblings=inclusion.siblings,
    )
    assert not UTXOSet.verify_proof(utxo.outpoint, bad_inclusion, root)
    bad_absence = UTXONonInclusionProof(
        outpoint=missing,
        empty_depth=absence.empty_depth,
        siblings=(*absence.siblings[:-1], bytes([255]) * 32),
    )
    assert not UTXOSet.verify_proof(missing, bad_absence, root)


def test_non_inclusion_proofs_cover_empty_sets() -> None:
    utxo_set = UTXOSet()
    missing = _outpoint()
    proof = utxo_set.non_inclusion_proof(missing)

    assert proof.empty_depth == 0
    assert proof.siblings == ()
    assert UTXOSet.verify_proof(missing, proof, utxo_set.merkle_root())


def test_utxo_rejects_malformed_values_and_expected_key_mismatch() -> None:
    with pytest.raises(ValueError):
        Outpoint.from_bytes(b"short")
    with pytest.raises(ValueError, match="nonzero"):
        UTXO(_outpoint(), 0, TEMPLATE_PKH, bytes(32))
    with pytest.raises(ValueError, match="MAX_SUPPLY"):
        UTXO(_outpoint(), MAX_SUPPLY_MATOMS + 1, TEMPLATE_PKH, bytes(32))
    with pytest.raises(ValueError, match="template"):
        UTXO(_outpoint(), 1, 99, bytes(32))
    with pytest.raises(ValueError, match="payload"):
        UTXO(
            _outpoint(),
            1,
            TEMPLATE_PKH,
            bytes(32),
            payload=b"x" * (TX_OUTPUT_PAYLOAD_MAX_BYTES + 1),
        )
    with pytest.raises(ValueError, match="expected key"):
        UTXO.from_bytes(_utxo().to_bytes(), expected_outpoint_key=bytes(32))
    with pytest.raises(ValueError, match="length"):
        UTXO.from_bytes(_utxo().to_bytes() + b"\x00")


def test_utxo_set_rejects_duplicates_missing_removes_and_bad_proof_shapes() -> None:
    utxo = _utxo()
    utxo_set = UTXOSet([utxo])

    with pytest.raises(KeyError, match="already exists"):
        utxo_set.add(utxo)
    with pytest.raises(KeyError, match="not present"):
        utxo_set.remove(_other_utxo().outpoint)
    with pytest.raises(KeyError, match="not present"):
        utxo_set.inclusion_proof(_other_utxo().outpoint)
    with pytest.raises(KeyError, match="present"):
        utxo_set.non_inclusion_proof(utxo.outpoint)
    with pytest.raises(TypeError):
        UTXOSet.verify_proof(utxo.outpoint, object(), utxo_set.merkle_root())  # type: ignore[arg-type]
