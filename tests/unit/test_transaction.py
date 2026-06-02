"""Tests for canonical transaction serialization."""

from __future__ import annotations

import pytest

from tensorpow.crypto.hash import DOMAIN_TX_SIGHASH, domain_hash
from tensorpow.crypto.signatures import SIG_TYPE_ML_DSA_RESERVED
from tensorpow.state.utxo import TEMPLATE_HASHLOCK, TEMPLATE_MULTISIG, TEMPLATE_PKH, Outpoint
from tensorpow.tx.transaction import (
    FORMAT_EPOCH,
    MAX_TX_BYTES,
    TX_SEQUENCE_FINAL,
    TX_WITNESS_MAX_BYTES,
    Input,
    Output,
    Transaction,
    TxDecodeError,
)

TX_BYTES_HEX = (
    "000000010000000000000002000000000000000100000102030405060708090a0b0c0d0e"
    "0f101112131415161718191a1b1c1d1e1f07000000ffffffff0200010201007b000000"
    "0000000000000a000000000000000b0000000000000020000909090909090909090909"
    "090909090909090909090909090909090909090909"
)
TX_ID_HEX = "3082fbd5faaf071a6924ea9e78dd1bd75abf46af59ca5c3c7dbcbd1fbab261ed"
SIGHASH_HEX = "1f40bf3886422f28971cb9bb6cfc4b74b817880bc6e67e868360faa80ad9e1a0"
INPUT_BYTES_HEX = (
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f07000000ffffffff02000102"
)
OUTPUT_BYTES_HEX = (
    "7b0000000000000000000a000000000000000b0000000000000020000909090909090909"
    "090909090909090909090909090909090909090909090909"
)


def _outpoint() -> Outpoint:
    return Outpoint(bytes(range(32)), 7)


def _input(witness: bytes = b"\x01\x02") -> Input:
    return Input(previous_outpoint=_outpoint(), witness=witness)


def _output(payload: bytes = bytes([9]) * 32) -> Output:
    return Output(
        amount_matoms=123,
        template_id=TEMPLATE_PKH,
        locktime_ms=10,
        lockheight=11,
        payload=payload,
    )


def _tx() -> Transaction:
    return Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=1,
        lockheight=2,
        inputs=(_input(),),
        outputs=(_output(),),
    )


def test_transaction_input_and_output_match_reference_vectors() -> None:
    tx = _tx()
    input_ = _input()
    output = _output()

    assert input_.to_bytes().hex() == INPUT_BYTES_HEX
    assert Input.from_bytes(input_.to_bytes()) == input_
    assert output.to_bytes().hex() == OUTPUT_BYTES_HEX
    assert Output.from_bytes(output.to_bytes()) == output
    assert tx.to_bytes().hex() == TX_BYTES_HEX
    assert Transaction.from_bytes(tx.to_bytes()) == tx
    assert tx.tx_id().hex() == TX_ID_HEX
    assert tx.sighash(0).hex() == SIGHASH_HEX


def test_coinbase_round_trip_allows_zero_inputs_but_requires_outputs() -> None:
    tx = Transaction.coinbase((_output(),))

    assert tx.inputs == ()
    assert Transaction.from_bytes(tx.to_bytes()) == tx
    with pytest.raises(ValueError, match="output_count"):
        Transaction.coinbase(())


def test_sighash_uses_empty_witnesses_and_checks_index() -> None:
    tx = _tx()
    changed_witness = Transaction(
        version=tx.version,
        sig_type=tx.sig_type,
        locktime_ms=tx.locktime_ms,
        lockheight=tx.lockheight,
        inputs=(_input(witness=b"different"),),
        outputs=tx.outputs,
    )

    assert changed_witness.tx_id() != tx.tx_id()
    assert changed_witness.sighash(0) == tx.sighash(0)
    assert tx.without_witnesses().inputs[0].witness == b""
    with pytest.raises(IndexError):
        tx.sighash(1)


def test_sighash_input_index_is_uint32_le() -> None:
    tx = Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=1,
        lockheight=2,
        inputs=(
            _input(),
            Input(
                previous_outpoint=Outpoint(bytes([1]) * 32, 1),
                witness=b"\x03\x04",
            ),
        ),
        outputs=(_output(),),
    )

    empty_witness_bytes = tx.without_witnesses().to_bytes()
    assert tx.sighash(1) == domain_hash(
        DOMAIN_TX_SIGHASH,
        empty_witness_bytes + (1).to_bytes(4, "little"),
    )
    assert tx.sighash(1) != domain_hash(
        DOMAIN_TX_SIGHASH,
        empty_witness_bytes + (1).to_bytes(2, "little"),
    )


def test_transaction_rejects_malformed_truncated_and_noncanonical_bytes() -> None:
    raw = bytearray(_tx().to_bytes())

    with pytest.raises(TxDecodeError, match="truncated"):
        Transaction.from_bytes(bytes(raw[:-1]))
    with pytest.raises(TxDecodeError, match="trailing"):
        Transaction.from_bytes(bytes(raw) + b"\x00")

    bad_version = bytearray(raw)
    bad_version[0] = 1
    with pytest.raises(TxDecodeError, match="FORMAT_EPOCH"):
        Transaction.from_bytes(bytes(bad_version))

    bad_sig_type = bytearray(raw)
    bad_sig_type[2] = SIG_TYPE_ML_DSA_RESERVED
    with pytest.raises(TxDecodeError, match="signature type"):
        Transaction.from_bytes(bytes(bad_sig_type))

    sequence_offset = 2 + 1 + 8 + 8 + 2 + 32 + 4
    bad_sequence = bytearray(raw)
    bad_sequence[sequence_offset] = 0
    with pytest.raises(TxDecodeError, match="TX_SEQUENCE_FINAL"):
        Transaction.from_bytes(bytes(bad_sequence))


def test_transaction_rejects_zero_outputs_and_bad_component_shapes() -> None:
    zero_outputs = (
        FORMAT_EPOCH.to_bytes(2, "little")
        + b"\x00"
        + (0).to_bytes(8, "little")
        + (0).to_bytes(8, "little")
        + (0).to_bytes(2, "little")
        + (0).to_bytes(2, "little")
    )
    with pytest.raises(TxDecodeError, match="output_count"):
        Transaction.from_bytes(zero_outputs)
    with pytest.raises(TxDecodeError, match="trailing"):
        Input.from_bytes(_input().to_bytes() + b"\x00")
    with pytest.raises(TxDecodeError, match="trailing"):
        Output.from_bytes(_output().to_bytes() + b"\x00")
    with pytest.raises(ValueError, match="witness"):
        Input(_outpoint(), TX_SEQUENCE_FINAL, b"x" * (TX_WITNESS_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="template"):
        Output(1, 99, payload=bytes(32))
    with pytest.raises(ValueError, match="PKH"):
        Output(1, TEMPLATE_PKH, payload=bytes(31))
    with pytest.raises(ValueError, match="multisig"):
        Output(1, TEMPLATE_MULTISIG, payload=bytes(32))
    with pytest.raises(ValueError, match="hashlock"):
        Output(1, TEMPLATE_HASHLOCK, payload=bytes(32))


def test_transaction_size_limit_is_enforced_on_construct_and_decode() -> None:
    oversized_outputs = tuple(Output(1, TEMPLATE_PKH, payload=bytes(32)) for _ in range(140))

    with pytest.raises(ValueError, match=str(MAX_TX_BYTES)):
        Transaction(FORMAT_EPOCH, 0, 0, 0, (), oversized_outputs)
    with pytest.raises(TxDecodeError, match="max length"):
        Transaction.from_bytes(bytes(MAX_TX_BYTES + 1))
