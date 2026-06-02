"""Tests for standard transaction scripts and templates."""

from __future__ import annotations

import pytest

from tensorpow.crypto.hash import hash_bytes
from tensorpow.crypto.signatures import (
    ED25519_SIGNATURE_BYTES,
    SIG_TYPE_ML_DSA_RESERVED,
    sign,
)
from tensorpow.state.utxo import TEMPLATE_HASHLOCK, TEMPLATE_MULTISIG, TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.tx.script import (
    MULTISIG_MAX_KEYS,
    OP_CHECKLOCKHEIGHT,
    OP_CHECKLOCKTIME,
    OP_CHECKMULTISIG,
    OP_CHECKSIG,
    OP_DUP,
    OP_HASH256,
    OP_HASHLOCK,
    OP_PUSH,
    OP_VERIFY,
    SCRIPT_MAX_BYTES,
    SCRIPT_MAX_ELEMENT_BYTES,
    SCRIPT_MAX_OPS,
    SCRIPT_MAX_STACK_ITEMS,
    ScriptContext,
    ScriptError,
    check_locks,
    encode_push,
    execute_script,
    pubkey_hash,
    validate_template_payload,
    verify_script,
    verify_template,
    verify_utxo_spend,
)

MESSAGE = bytes([5]) * 32
PUB1 = bytes.fromhex("343010a1aba8774dd1e6f4f0c3349bae6824908a1e64cd638dc2ed1bc625af1d")
PRIV1 = bytes.fromhex("cd4f7f79a2b8168f5cbeccb55d415492fd3504e52ed4fe7b02ea404fede9a40b")
PKH1 = bytes.fromhex("ce0d4ef3ea76c782c6ca2b6368b11c3509324d43d58c76511d2bac9c7a32e650")
SIG1 = bytes.fromhex(
    "e29ead045bb9a506ae2cff64ebccf620afebbdfff03339ae433b8af371a2331ee"
    "4ea4e5086a835ae05420fd45f7566972849f6791dba94f05929b7645c68ce03"
)
PUB2 = bytes.fromhex("ad5e8be40705840112d69a3fcd1603c2974a356878b4491744a6d62ae5c72a43")
PRIV2 = bytes.fromhex("df0f6ff90da5acd52c9b9abc7d095c50da6283c116d9698723c2eb2014fde303")
PKH2 = bytes.fromhex("dfadc17a9f2c11c59bc3e629b8368bf0632868ec14f3fedbe5fbde109284cbf4")
SIG2 = bytes.fromhex(
    "82238d48f307b8b7bacfa938fbdb8bd48fcd6487604da309a29eb86a053f7462"
    "74a850557b8249bfbd41f6ce5af31fd65e682f80ac049b2fd2798aec785d4902"
)


def test_pkh_template_validates_signature_and_public_key_hash() -> None:
    witness = SIG1 + PUB1

    assert pubkey_hash(PUB1) == PKH1
    assert verify_template(TEMPLATE_PKH, PKH1, witness, MESSAGE)
    assert verify_template(TEMPLATE_PKH, PKH1, sign(MESSAGE, PRIV1) + PUB1, MESSAGE)
    assert not verify_template(TEMPLATE_PKH, PKH2, witness, MESSAGE)
    assert not verify_template(TEMPLATE_PKH, PKH1, bytes([SIG1[0] ^ 1]) + SIG1[1:] + PUB1, MESSAGE)
    assert not verify_template(
        TEMPLATE_PKH, PKH1, witness, MESSAGE, sig_type=SIG_TYPE_ML_DSA_RESERVED
    )


def test_multisig_template_accepts_ordered_threshold_signatures() -> None:
    payload = bytes((2, 2)) + PUB1 + PUB2
    witness = SIG1 + SIG2

    validate_template_payload(TEMPLATE_MULTISIG, payload)
    assert verify_template(TEMPLATE_MULTISIG, payload, witness, MESSAGE)
    assert not verify_template(TEMPLATE_MULTISIG, payload, SIG2 + SIG1, MESSAGE)
    assert not verify_template(TEMPLATE_MULTISIG, payload, SIG1, MESSAGE)
    duplicate_payload = bytes((2, 2)) + PUB1 + PUB1
    with pytest.raises(ScriptError, match="distinct"):
        validate_template_payload(TEMPLATE_MULTISIG, duplicate_payload)
    assert not verify_template(TEMPLATE_MULTISIG, duplicate_payload, SIG1 + SIG1, MESSAGE)

    with pytest.raises(ScriptError, match="threshold"):
        validate_template_payload(TEMPLATE_MULTISIG, bytes((0, 1)) + PUB1)
    with pytest.raises(ScriptError, match="pubkey_count"):
        validate_template_payload(TEMPLATE_MULTISIG, bytes((1, MULTISIG_MAX_KEYS + 1)))


def test_hashlock_template_requires_preimage_then_inner_pkh_witness() -> None:
    preimage = b"tensorpow"
    payload = hash_bytes(preimage) + PKH1
    witness = len(preimage).to_bytes(2, "little") + preimage + SIG1 + PUB1

    assert verify_template(TEMPLATE_HASHLOCK, payload, witness, MESSAGE)
    assert not verify_template(
        TEMPLATE_HASHLOCK, payload, witness.replace(preimage, b"wrongpow"), MESSAGE
    )
    assert not verify_template(TEMPLATE_HASHLOCK, payload, witness[:-1], MESSAGE)

    with pytest.raises(ScriptError, match="hashlock"):
        validate_template_payload(TEMPLATE_HASHLOCK, bytes(32))


def test_script_vm_executes_core_opcodes_and_lock_checks() -> None:
    context = ScriptContext(message=MESSAGE, current_time_ms=100, current_height=9)
    script = (
        encode_push(SIG1)
        + encode_push(PUB1)
        + bytes((OP_CHECKSIG, OP_VERIFY))
        + encode_push(b"x")
        + encode_push(hash_bytes(b"x"))
        + bytes((OP_HASHLOCK, OP_VERIFY))
        + encode_push((100).to_bytes(8, "little"))
        + bytes((OP_CHECKLOCKTIME, OP_VERIFY))
        + encode_push((9).to_bytes(8, "little"))
        + bytes((OP_CHECKLOCKHEIGHT,))
    )

    assert verify_script(script, context)
    assert execute_script(script, context)[-1] == b"\x01"
    assert execute_script(encode_push(b"x") + bytes((OP_HASH256,)), context) == (hash_bytes(b"x"),)
    with pytest.raises(ScriptError, match="locktime"):
        check_locks(locktime_ms=101, lockheight=0, current_time_ms=100, current_height=9)
    with pytest.raises(ScriptError, match="lockheight"):
        check_locks(locktime_ms=0, lockheight=10, current_time_ms=100, current_height=9)


def test_script_vm_requires_canonical_boolean_values() -> None:
    context = ScriptContext(message=MESSAGE)

    assert verify_script(encode_push(b"\x01"), context)
    assert not verify_script(encode_push(b""), context)
    assert not verify_script(encode_push(b"\x02"), context)

    with pytest.raises(ScriptError, match="canonical"):
        execute_script(encode_push(b"\x02") + bytes((OP_VERIFY,)), context)
    with pytest.raises(ScriptError, match="canonical"):
        execute_script(encode_push(b"\x00") + bytes((OP_VERIFY,)), context)


def test_script_vm_rejects_malformed_programs_and_stack_shapes() -> None:
    context = ScriptContext(message=MESSAGE)

    with pytest.raises(ScriptError, match="underflow"):
        execute_script(bytes((OP_DUP,)), context)
    with pytest.raises(ScriptError, match="unknown"):
        execute_script(b"\xff", context)
    with pytest.raises(ScriptError, match="signature type"):
        ScriptContext(b"", sig_type=SIG_TYPE_ML_DSA_RESERVED)
    with pytest.raises(ScriptError, match="truncated push length"):
        execute_script(bytes((OP_PUSH, 1)), context)
    with pytest.raises(ScriptError, match="truncated push data"):
        execute_script(bytes((OP_PUSH, 2, 0, 1)), context)
    with pytest.raises(ScriptError, match="max element"):
        encode_push(b"x" * (SCRIPT_MAX_ELEMENT_BYTES + 1))
    with pytest.raises(ScriptError, match="max size"):
        execute_script(b"\x00" * (SCRIPT_MAX_BYTES + 1), context)
    with pytest.raises(ScriptError, match="op count"):
        execute_script(encode_push(b"") * (SCRIPT_MAX_OPS + 1), context)
    with pytest.raises(ScriptError, match="stack"):
        execute_script(b"", context, initial_stack=(b"",) * (SCRIPT_MAX_STACK_ITEMS + 1))
    with pytest.raises(ScriptError, match="signature"):
        execute_script(
            encode_push(b"\x01") + bytes((OP_CHECKMULTISIG,)),
            context,
            initial_stack=(b"x" * (ED25519_SIGNATURE_BYTES - 1), b"\x01", PUB1),
        )
    duplicate_multisig = (
        encode_push(SIG1)
        + encode_push(SIG1)
        + encode_push(b"\x02")
        + encode_push(PUB1)
        + encode_push(PUB1)
        + encode_push(b"\x02")
        + bytes((OP_CHECKMULTISIG,))
    )
    with pytest.raises(ScriptError, match="distinct"):
        execute_script(duplicate_multisig, context)


def test_utxo_template_spend_uses_utxo_payload_and_locks() -> None:
    utxo = UTXO(
        outpoint=Outpoint(bytes(32), 0),
        amount_matoms=1,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=PKH1,
        locktime_ms=100,
        lockheight=10,
    )

    assert verify_utxo_spend(
        utxo,
        SIG1 + PUB1,
        MESSAGE,
        current_time_ms=100,
        current_height=10,
    )
    assert not verify_utxo_spend(
        utxo,
        SIG1 + PUB1,
        MESSAGE,
        current_time_ms=99,
        current_height=10,
    )
