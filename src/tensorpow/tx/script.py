"""Deterministic minimal script interpreter and standard templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tensorpow.crypto.hash import DOMAIN_ADDRESS, HASH_LEN_BYTES, domain_hash, hash_bytes
from tensorpow.crypto.signatures import (
    ED25519_PUBLIC_KEY_BYTES,
    ED25519_SIGNATURE_BYTES,
    SIG_TYPE_ED25519,
    verify_by_sig_type,
)
from tensorpow.state.utxo import (
    TEMPLATE_HASHLOCK,
    TEMPLATE_MULTISIG,
    TEMPLATE_PKH,
    UTXO,
)
from tensorpow.tx.transaction import TX_WITNESS_MAX_BYTES, U8_MAX, U16_BYTES, U64_MAX

SCRIPT_MAX_BYTES: Final[int] = 1024
SCRIPT_MAX_OPS: Final[int] = 256
SCRIPT_MAX_STACK_ITEMS: Final[int] = 1024
SCRIPT_MAX_ELEMENT_BYTES: Final[int] = 520
MULTISIG_MAX_KEYS: Final[int] = 15

OP_PUSH: Final[int] = 0x01
OP_VERIFY: Final[int] = 0x69
OP_DUP: Final[int] = 0x76
OP_HASH256: Final[int] = 0xAA
OP_CHECKSIG: Final[int] = 0xAC
OP_CHECKMULTISIG: Final[int] = 0xAE
OP_HASHLOCK: Final[int] = 0xB1
OP_CHECKLOCKTIME: Final[int] = 0xB2
OP_CHECKLOCKHEIGHT: Final[int] = 0xB3


class ScriptError(ValueError):
    """Raised when script execution or template validation fails."""


@dataclass(frozen=True, slots=True)
class ScriptContext:
    """Consensus values visible to script execution."""

    message: bytes
    sig_type: int = SIG_TYPE_ED25519
    current_time_ms: int = 0
    current_height: int = 0

    def __post_init__(self) -> None:
        _require_bytes("message", self.message)
        _require_sig_type(self.sig_type)
        _require_u64("current_time_ms", self.current_time_ms)
        _require_u64("current_height", self.current_height)


def execute_script(
    script: bytes,
    context: ScriptContext,
    *,
    initial_stack: tuple[bytes, ...] = (),
) -> tuple[bytes, ...]:
    """Run a bounded stack script and return the final stack."""

    _require_script(script)
    if not isinstance(context, ScriptContext):
        raise TypeError("context must be ScriptContext")
    stack = _initial_stack(initial_stack)

    offset = 0
    op_count = 0
    while offset < len(script):
        op_count += 1
        if op_count > SCRIPT_MAX_OPS:
            raise ScriptError("script exceeds max op count")

        opcode = script[offset]
        offset += 1
        if opcode == OP_PUSH:
            item, offset = _read_push(script, offset)
            _push(stack, item)
        elif opcode == OP_DUP:
            if not stack:
                raise ScriptError("OP_DUP stack underflow")
            _push(stack, stack[-1])
        elif opcode == OP_HASH256:
            _push(stack, hash_bytes(_pop(stack, "OP_HASH256")))
        elif opcode == OP_CHECKSIG:
            public_key = _pop(stack, "OP_CHECKSIG")
            signature = _pop(stack, "OP_CHECKSIG")
            _push_bool(stack, _verify_signature(context, signature, public_key))
        elif opcode == OP_CHECKMULTISIG:
            _push_bool(stack, _execute_checkmultisig(stack, context))
        elif opcode == OP_HASHLOCK:
            expected_hash = _pop(stack, "OP_HASHLOCK")
            preimage = _pop(stack, "OP_HASHLOCK")
            if len(expected_hash) != HASH_LEN_BYTES:
                raise ScriptError("hashlock expected hash must be 32 bytes")
            _push_bool(stack, hash_bytes(preimage) == expected_hash)
        elif opcode == OP_CHECKLOCKTIME:
            required = _decode_lock(_pop(stack, "OP_CHECKLOCKTIME"), "locktime_ms")
            _push_bool(stack, required == 0 or context.current_time_ms >= required)
        elif opcode == OP_CHECKLOCKHEIGHT:
            required = _decode_lock(_pop(stack, "OP_CHECKLOCKHEIGHT"), "lockheight")
            _push_bool(stack, required == 0 or context.current_height >= required)
        elif opcode == OP_VERIFY:
            if not _truthy(_pop(stack, "OP_VERIFY")):
                raise ScriptError("OP_VERIFY failed")
        else:
            raise ScriptError(f"unknown opcode 0x{opcode:02x}")

    return tuple(stack)


def verify_script(
    script: bytes,
    context: ScriptContext,
    *,
    initial_stack: tuple[bytes, ...] = (),
) -> bool:
    """Return True when a script leaves a truthy top stack item."""

    try:
        stack = execute_script(script, context, initial_stack=initial_stack)
    except (ScriptError, TypeError, ValueError):
        return False
    return bool(stack) and _truthy(stack[-1])


def encode_push(item: bytes) -> bytes:
    """Encode one canonical `OP_PUSH || uint16_len_le || item` operation."""

    _require_stack_item("item", item)
    return bytes((OP_PUSH,)) + len(item).to_bytes(U16_BYTES, "little") + item


def pubkey_hash(public_key: bytes) -> bytes:
    """Return the active address-domain public-key hash."""

    _require_bytes("public_key", public_key)
    if len(public_key) != ED25519_PUBLIC_KEY_BYTES:
        raise ValueError(f"public_key must be {ED25519_PUBLIC_KEY_BYTES} bytes")
    return domain_hash(DOMAIN_ADDRESS, public_key)


def validate_template_payload(template_id: int, payload: bytes) -> None:
    """Validate the canonical payload shape for an active output template."""

    _require_template_id(template_id)
    _require_bytes("payload", payload)
    if template_id == TEMPLATE_PKH:
        if len(payload) != HASH_LEN_BYTES:
            raise ScriptError("PKH payload must be 32 bytes")
    elif template_id == TEMPLATE_MULTISIG:
        _parse_multisig_payload(payload)
    elif template_id == TEMPLATE_HASHLOCK:
        if len(payload) <= HASH_LEN_BYTES:
            raise ScriptError("hashlock payload must include hash plus inner template payload")
        inner_template = _infer_inner_template_id(payload[HASH_LEN_BYTES:])
        validate_template_payload(inner_template, payload[HASH_LEN_BYTES:])


def verify_template(
    template_id: int,
    payload: bytes,
    witness: bytes,
    message: bytes,
    *,
    sig_type: int = SIG_TYPE_ED25519,
    locktime_ms: int = 0,
    lockheight: int = 0,
    current_time_ms: int = 0,
    current_height: int = 0,
) -> bool:
    """Verify a witness against a standard template payload."""

    try:
        validate_template_payload(template_id, payload)
        _require_bytes("witness", witness, max_len=TX_WITNESS_MAX_BYTES)
        _require_bytes("message", message)
        _require_sig_type(sig_type)
        check_locks(
            locktime_ms=locktime_ms,
            lockheight=lockheight,
            current_time_ms=current_time_ms,
            current_height=current_height,
        )
        if template_id == TEMPLATE_PKH:
            return _verify_pkh(payload, witness, message, sig_type)
        if template_id == TEMPLATE_MULTISIG:
            return _verify_multisig_template(payload, witness, message, sig_type)
        if template_id == TEMPLATE_HASHLOCK:
            return _verify_hashlock(payload, witness, message, sig_type)
    except (ScriptError, TypeError, ValueError):
        return False
    return False


def verify_utxo_spend(
    utxo: UTXO,
    witness: bytes,
    message: bytes,
    *,
    sig_type: int = SIG_TYPE_ED25519,
    current_time_ms: int = 0,
    current_height: int = 0,
) -> bool:
    """Verify a witness against a UTXO's template and absolute locks."""

    if not isinstance(utxo, UTXO):
        raise TypeError("utxo must be UTXO")
    payload = utxo.owner_pubkey_hash if utxo.template_id == TEMPLATE_PKH else utxo.payload
    return verify_template(
        utxo.template_id,
        payload,
        witness,
        message,
        sig_type=sig_type,
        locktime_ms=utxo.locktime_ms,
        lockheight=utxo.lockheight,
        current_time_ms=current_time_ms,
        current_height=current_height,
    )


def check_locks(
    *,
    locktime_ms: int,
    lockheight: int,
    current_time_ms: int,
    current_height: int,
) -> None:
    """Reject immature absolute wall-time or DAG-height locks."""

    _require_u64("locktime_ms", locktime_ms)
    _require_u64("lockheight", lockheight)
    _require_u64("current_time_ms", current_time_ms)
    _require_u64("current_height", current_height)
    if locktime_ms != 0 and current_time_ms < locktime_ms:
        raise ScriptError("locktime is not mature")
    if lockheight != 0 and current_height < lockheight:
        raise ScriptError("lockheight is not mature")


def _verify_pkh(payload: bytes, witness: bytes, message: bytes, sig_type: int) -> bool:
    if len(witness) != ED25519_SIGNATURE_BYTES + ED25519_PUBLIC_KEY_BYTES:
        return False
    signature = witness[:ED25519_SIGNATURE_BYTES]
    public_key = witness[ED25519_SIGNATURE_BYTES:]
    return pubkey_hash(public_key) == payload and _verify_signature_bytes(
        sig_type,
        message,
        signature,
        public_key,
    )


def _verify_multisig_template(
    payload: bytes,
    witness: bytes,
    message: bytes,
    sig_type: int,
) -> bool:
    threshold, public_keys = _parse_multisig_payload(payload)
    if len(witness) != threshold * ED25519_SIGNATURE_BYTES:
        return False
    signatures = tuple(
        witness[offset : offset + ED25519_SIGNATURE_BYTES]
        for offset in range(0, len(witness), ED25519_SIGNATURE_BYTES)
    )
    return _verify_ordered_multisig(sig_type, message, signatures, public_keys)


def _verify_hashlock(payload: bytes, witness: bytes, message: bytes, sig_type: int) -> bool:
    if len(witness) < U16_BYTES:
        return False
    preimage_len = int.from_bytes(witness[:U16_BYTES], "little")
    if preimage_len > SCRIPT_MAX_ELEMENT_BYTES:
        return False
    preimage_end = U16_BYTES + preimage_len
    if preimage_end > len(witness):
        return False

    expected_hash = payload[:HASH_LEN_BYTES]
    inner_payload = payload[HASH_LEN_BYTES:]
    preimage = witness[U16_BYTES:preimage_end]
    inner_witness = witness[preimage_end:]
    if hash_bytes(preimage) != expected_hash:
        return False
    inner_template = _infer_inner_template_id(inner_payload)
    return verify_template(inner_template, inner_payload, inner_witness, message, sig_type=sig_type)


def _execute_checkmultisig(stack: list[bytes], context: ScriptContext) -> bool:
    pubkey_count = _decode_small_int(_pop(stack, "OP_CHECKMULTISIG"), "pubkey_count")
    if not 1 <= pubkey_count <= MULTISIG_MAX_KEYS:
        raise ScriptError("pubkey_count outside multisig range")

    public_keys = tuple(
        reversed(tuple(_pop(stack, "OP_CHECKMULTISIG") for _ in range(pubkey_count)))
    )
    for public_key in public_keys:
        if len(public_key) != ED25519_PUBLIC_KEY_BYTES:
            raise ScriptError("multisig public key must be 32 bytes")

    threshold = _decode_small_int(_pop(stack, "OP_CHECKMULTISIG"), "threshold")
    if not 1 <= threshold <= pubkey_count:
        raise ScriptError("threshold outside multisig range")

    signatures = tuple(reversed(tuple(_pop(stack, "OP_CHECKMULTISIG") for _ in range(threshold))))
    for signature in signatures:
        if len(signature) != ED25519_SIGNATURE_BYTES:
            raise ScriptError("multisig signature must be 64 bytes")

    return _verify_ordered_multisig(context.sig_type, context.message, signatures, public_keys)


def _parse_multisig_payload(payload: bytes) -> tuple[int, tuple[bytes, ...]]:
    if len(payload) < 2:
        raise ScriptError("multisig payload is truncated")
    threshold = payload[0]
    pubkey_count = payload[1]
    if not 1 <= pubkey_count <= MULTISIG_MAX_KEYS:
        raise ScriptError("pubkey_count outside multisig range")
    if not 1 <= threshold <= pubkey_count:
        raise ScriptError("threshold outside multisig range")
    expected_len = 2 + pubkey_count * ED25519_PUBLIC_KEY_BYTES
    if len(payload) != expected_len:
        raise ScriptError("multisig payload length mismatch")
    public_keys = tuple(
        payload[offset : offset + ED25519_PUBLIC_KEY_BYTES]
        for offset in range(2, len(payload), ED25519_PUBLIC_KEY_BYTES)
    )
    return threshold, public_keys


def _infer_inner_template_id(payload: bytes) -> int:
    if len(payload) == HASH_LEN_BYTES:
        return TEMPLATE_PKH
    _parse_multisig_payload(payload)
    return TEMPLATE_MULTISIG


def _verify_ordered_multisig(
    sig_type: int,
    message: bytes,
    signatures: tuple[bytes, ...],
    public_keys: tuple[bytes, ...],
) -> bool:
    public_key_index = 0
    for signature in signatures:
        matched = False
        while public_key_index < len(public_keys):
            public_key = public_keys[public_key_index]
            public_key_index += 1
            if _verify_signature_bytes(sig_type, message, signature, public_key):
                matched = True
                break
        if not matched:
            return False
    return True


def _verify_signature(context: ScriptContext, signature: bytes, public_key: bytes) -> bool:
    return _verify_signature_bytes(context.sig_type, context.message, signature, public_key)


def _verify_signature_bytes(
    sig_type: int,
    message: bytes,
    signature: bytes,
    public_key: bytes,
) -> bool:
    if len(signature) != ED25519_SIGNATURE_BYTES:
        return False
    if len(public_key) != ED25519_PUBLIC_KEY_BYTES:
        return False
    return verify_by_sig_type(sig_type, message, signature, public_key)


def _read_push(script: bytes, offset: int) -> tuple[bytes, int]:
    if offset + U16_BYTES > len(script):
        raise ScriptError("truncated push length")
    length = int.from_bytes(script[offset : offset + U16_BYTES], "little")
    offset += U16_BYTES
    if length > SCRIPT_MAX_ELEMENT_BYTES:
        raise ScriptError("push exceeds max element size")
    end = offset + length
    if end > len(script):
        raise ScriptError("truncated push data")
    return script[offset:end], end


def _initial_stack(initial_stack: tuple[bytes, ...]) -> list[bytes]:
    if not isinstance(initial_stack, tuple):
        raise TypeError("initial_stack must be a tuple")
    stack = []
    for item in initial_stack:
        _require_stack_item("stack item", item)
        stack.append(item)
    if len(stack) > SCRIPT_MAX_STACK_ITEMS:
        raise ScriptError("stack exceeds max item count")
    return stack


def _push(stack: list[bytes], item: bytes) -> None:
    _require_stack_item("stack item", item)
    if len(stack) >= SCRIPT_MAX_STACK_ITEMS:
        raise ScriptError("stack exceeds max item count")
    stack.append(item)


def _push_bool(stack: list[bytes], value: bool) -> None:
    _push(stack, b"\x01" if value else b"")


def _pop(stack: list[bytes], opcode_name: str) -> bytes:
    if not stack:
        raise ScriptError(f"{opcode_name} stack underflow")
    return stack.pop()


def _decode_small_int(item: bytes, name: str) -> int:
    if len(item) != 1:
        raise ScriptError(f"{name} must be one byte")
    return item[0]


def _decode_lock(item: bytes, name: str) -> int:
    if len(item) != 8:
        raise ScriptError(f"{name} must be uint64 little-endian bytes")
    return int.from_bytes(item, "little")


def _truthy(item: bytes) -> bool:
    return any(byte != 0 for byte in item)


def _require_template_id(template_id: int) -> None:
    _require_u8("template_id", template_id)
    if template_id not in (TEMPLATE_PKH, TEMPLATE_MULTISIG, TEMPLATE_HASHLOCK):
        raise ScriptError("template_id must be an active output template")


def _require_sig_type(sig_type: int) -> None:
    _require_u8("sig_type", sig_type)
    if sig_type != SIG_TYPE_ED25519:
        raise ScriptError("sig_type must be an active signature type")


def _require_u8(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= U8_MAX:
        raise ValueError(f"{name} outside uint range")


def _require_u64(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= U64_MAX:
        raise ValueError(f"{name} outside uint range")


def _require_bytes(name: str, value: bytes, *, max_len: int | None = None) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if max_len is not None and len(value) > max_len:
        raise ValueError(f"{name} exceeds max length")


def _require_stack_item(name: str, value: bytes) -> None:
    _require_bytes(name, value)
    if len(value) > SCRIPT_MAX_ELEMENT_BYTES:
        raise ScriptError(f"{name} exceeds max element size")


def _require_script(script: bytes) -> None:
    _require_bytes("script", script)
    if len(script) > SCRIPT_MAX_BYTES:
        raise ScriptError("script exceeds max size")


__all__ = [
    "MULTISIG_MAX_KEYS",
    "OP_CHECKLOCKHEIGHT",
    "OP_CHECKLOCKTIME",
    "OP_CHECKMULTISIG",
    "OP_CHECKSIG",
    "OP_DUP",
    "OP_HASH256",
    "OP_HASHLOCK",
    "OP_PUSH",
    "OP_VERIFY",
    "SCRIPT_MAX_BYTES",
    "SCRIPT_MAX_ELEMENT_BYTES",
    "SCRIPT_MAX_OPS",
    "SCRIPT_MAX_STACK_ITEMS",
    "ScriptContext",
    "ScriptError",
    "check_locks",
    "encode_push",
    "execute_script",
    "pubkey_hash",
    "validate_template_payload",
    "verify_script",
    "verify_template",
    "verify_utxo_spend",
]
