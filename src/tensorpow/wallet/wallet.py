"""Encrypted wallet keystores and standard PKH transaction signing."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from tensorpow.crypto.address import address_to_pubkey_hash, pubkey_to_address, validate_address
from tensorpow.crypto.signatures import ED25519_PRIVATE_KEY_BYTES, SIG_TYPE_ED25519, Keypair, sign
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.tx.script import pubkey_hash, verify_template
from tensorpow.tx.transaction import FORMAT_EPOCH, Input, Output, Transaction

KEYSTORE_FORMAT: Final[str] = "tensorpow-wallet-keystore"
KEYSTORE_AAD: Final[bytes] = b"tensorpow-wallet-keystore"
KEYSTORE_CIPHER: Final[str] = "aes-256-gcm"
KEYSTORE_KDF: Final[str] = "scrypt"

KEYSTORE_SALT_BYTES: Final[int] = 16
KEYSTORE_NONCE_BYTES: Final[int] = 12
KEYSTORE_KEY_BYTES: Final[int] = 32
SCRYPT_N: Final[int] = 2**14
SCRYPT_R: Final[int] = 8
SCRYPT_P: Final[int] = 1

U32_MAX: Final[int] = 0xFFFFFFFF
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF


class WalletError(ValueError):
    """Raised when wallet input, keystore data, or spend construction is invalid."""


@dataclass(frozen=True, slots=True)
class WalletMetadata:
    """Public wallet metadata stored alongside an encrypted seed."""

    public_key: bytes
    address: str

    def __post_init__(self) -> None:
        _require_exact_bytes("public_key", self.public_key, 32)
        if not validate_address(self.address):
            raise WalletError("wallet address is invalid")
        if pubkey_to_address(self.public_key) != self.address:
            raise WalletError("wallet address does not match public key")


@dataclass(frozen=True, slots=True)
class Wallet:
    """Single-key Ed25519 wallet for standard PKH outputs."""

    private_key: bytes = field(repr=False)
    public_key: bytes

    def __post_init__(self) -> None:
        _require_exact_bytes("private_key", self.private_key, ED25519_PRIVATE_KEY_BYTES)
        _require_exact_bytes("public_key", self.public_key, 32)
        if _public_key_from_seed(self.private_key) != self.public_key:
            raise WalletError("public key does not match private seed")

    @classmethod
    def generate(cls) -> Wallet:
        """Create a new random wallet."""

        keypair = Keypair.generate()
        return cls(private_key=keypair.private_key, public_key=keypair.public_key)

    @classmethod
    def from_seed(cls, seed: bytes) -> Wallet:
        """Recover a wallet from a raw 32-byte Ed25519 private-key seed."""

        _require_exact_bytes("seed", seed, ED25519_PRIVATE_KEY_BYTES)
        return cls(private_key=seed, public_key=_public_key_from_seed(seed))

    @classmethod
    def create(cls) -> Wallet:
        """Create a new random wallet."""

        return cls.generate()

    @classmethod
    def recover(cls, seed_hex: str) -> Wallet:
        """Recover a wallet from a 32-byte hex seed."""

        return cls.from_seed(decode_seed_hex(seed_hex))

    @classmethod
    def load(cls, path: str | Path, password: str | bytes) -> Wallet:
        """Load and decrypt a keystore."""

        return load_keystore(path, password)

    @property
    def address(self) -> str:
        """Return the wallet's Tensorcoin address."""

        return pubkey_to_address(self.public_key)

    def address_text(self) -> str:
        """Return the wallet address for call-style compatibility."""

        return self.address

    @property
    def seed_hex(self) -> str:
        """Return the raw recovery seed as lowercase hex."""

        return self.private_key.hex()

    def seed_text(self) -> str:
        """Return the raw recovery seed as lowercase hex."""

        return self.seed_hex

    @property
    def owner_pubkey_hash(self) -> bytes:
        """Return the public-key hash owned by this wallet."""

        return pubkey_hash(self.public_key)

    def pubkey_hash(self) -> bytes:
        """Return the public-key hash owned by this wallet."""

        return self.owner_pubkey_hash

    @property
    def metadata(self) -> WalletMetadata:
        """Return public wallet metadata."""

        return WalletMetadata(public_key=self.public_key, address=self.address)

    def owned_utxos(self, utxos: Iterable[UTXO]) -> tuple[UTXO, ...]:
        """Return standard PKH UTXOs spendable by this wallet in canonical order."""

        owned = []
        for utxo in _require_utxos(utxos):
            if (
                utxo.template_id == TEMPLATE_PKH
                and utxo.owner_pubkey_hash == self.owner_pubkey_hash
            ):
                owned.append(utxo)
        return tuple(sorted(owned, key=lambda utxo: utxo.outpoint.to_bytes()))

    def balance_matoms(self, utxos: Iterable[UTXO]) -> int:
        """Return the wallet's confirmed standard-PKH balance in matoms."""

        return sum(utxo.amount_matoms for utxo in self.owned_utxos(utxos))

    def balance(self, utxos: Iterable[UTXO]) -> int:
        """Return the wallet's confirmed standard-PKH balance in matoms."""

        return self.balance_matoms(utxos)

    def save(self, path: str | Path, password: str | bytes, *, overwrite: bool = False) -> Path:
        """Save this wallet to an encrypted keystore and return the path."""

        save_keystore(self, path, password, overwrite=overwrite)
        return Path(path)

    def create_signed_transaction(
        self,
        *,
        utxos: Iterable[UTXO],
        recipient_address: str,
        amount_matoms: int,
        fee_matoms: int = 0,
        change_address: str | None = None,
        locktime_ms: int = 0,
        lockheight: int = 0,
    ) -> Transaction:
        """Build and sign a standard PKH transaction with deterministic coin selection."""

        _require_positive_u64("amount_matoms", amount_matoms)
        _require_u64("fee_matoms", fee_matoms)
        _require_u64("locktime_ms", locktime_ms)
        _require_u64("lockheight", lockheight)
        recipient_hash = _address_hash("recipient_address", recipient_address)
        change_hash = _address_hash("change_address", change_address or self.address)
        required = amount_matoms + fee_matoms
        if required > U64_MAX:
            raise WalletError("amount plus fee outside uint64 range")

        selected: list[UTXO] = []
        selected_amount = 0
        for utxo in self.owned_utxos(utxos):
            selected.append(utxo)
            selected_amount += utxo.amount_matoms
            if selected_amount >= required:
                break
        if selected_amount < required:
            raise WalletError("insufficient spendable balance")

        outputs: list[Output] = [Output(amount_matoms, TEMPLATE_PKH, payload=recipient_hash)]
        change_matoms = selected_amount - required
        if change_matoms:
            outputs.append(Output(change_matoms, TEMPLATE_PKH, payload=change_hash))

        unsigned = Transaction(
            version=FORMAT_EPOCH,
            sig_type=SIG_TYPE_ED25519,
            locktime_ms=locktime_ms,
            lockheight=lockheight,
            inputs=tuple(Input(utxo.outpoint) for utxo in selected),
            outputs=tuple(outputs),
        )
        return self.sign_transaction(unsigned)

    def sign_transaction(self, unsigned_tx: Transaction) -> Transaction:
        """Return ``unsigned_tx`` with every input signed by this wallet."""

        if not isinstance(unsigned_tx, Transaction):
            raise TypeError("unsigned_tx must be Transaction")
        if not unsigned_tx.inputs:
            raise WalletError("transaction has no inputs to sign")
        base_tx = unsigned_tx.without_witnesses()
        signed_inputs = []
        for index, input_ in enumerate(base_tx.inputs):
            signature = sign(base_tx.sighash(index), self.private_key)
            witness = signature + self.public_key
            if not verify_template(
                TEMPLATE_PKH,
                self.owner_pubkey_hash,
                witness,
                base_tx.sighash(index),
                sig_type=base_tx.sig_type,
            ):
                raise WalletError("wallet signature failed verification")
            signed_inputs.append(
                Input(
                    previous_outpoint=input_.previous_outpoint,
                    sequence=input_.sequence,
                    witness=witness,
                )
            )
        return Transaction(
            version=base_tx.version,
            sig_type=base_tx.sig_type,
            locktime_ms=base_tx.locktime_ms,
            lockheight=base_tx.lockheight,
            inputs=tuple(signed_inputs),
            outputs=base_tx.outputs,
        )

    def build_transaction(
        self,
        available_utxos: Iterable[UTXO],
        recipient_address: str,
        *,
        amount_matoms: int,
        fee_matoms: int,
    ) -> Transaction:
        """Compatibility wrapper for standard signed transaction construction."""

        return self.create_signed_transaction(
            utxos=available_utxos,
            recipient_address=recipient_address,
            amount_matoms=amount_matoms,
            fee_matoms=fee_matoms,
        )


def create_keystore(
    path: str | Path,
    password: str | bytes,
    *,
    overwrite: bool = False,
) -> Wallet:
    """Generate a new wallet, write its encrypted keystore, and return it."""

    wallet = Wallet.generate()
    save_keystore(wallet, path, password, overwrite=overwrite)
    return wallet


def create_wallet(path: str | Path, password: str | bytes, *, overwrite: bool = False) -> Wallet:
    """Create and save a new wallet."""

    return create_keystore(path, password, overwrite=overwrite)


def load_wallet(path: str | Path, password: str | bytes) -> Wallet:
    """Load and decrypt a keystore."""

    return load_keystore(path, password)


def recover_wallet(
    path: str | Path,
    password: str | bytes,
    seed_hex: str,
    *,
    overwrite: bool = False,
) -> Wallet:
    """Recover a wallet from hex seed and save it."""

    return import_seed_to_keystore(decode_seed_hex(seed_hex), path, password, overwrite=overwrite)


def import_seed_to_keystore(
    seed: bytes,
    path: str | Path,
    password: str | bytes,
    *,
    overwrite: bool = False,
) -> Wallet:
    """Recover a wallet from ``seed`` and write it as an encrypted keystore."""

    wallet = Wallet.from_seed(seed)
    save_keystore(wallet, path, password, overwrite=overwrite)
    return wallet


def save_keystore(
    wallet: Wallet,
    path: str | Path,
    password: str | bytes,
    *,
    overwrite: bool = False,
) -> None:
    """Write ``wallet`` to ``path`` with password-based authenticated encryption."""

    if not isinstance(wallet, Wallet):
        raise TypeError("wallet must be Wallet")
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError("wallet keystore already exists")
    salt = os.urandom(KEYSTORE_SALT_BYTES)
    nonce = os.urandom(KEYSTORE_NONCE_BYTES)
    key = _derive_keystore_key(_password_bytes(password), salt)
    plaintext = _json_bytes({"seed": wallet.seed_hex})
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, KEYSTORE_AAD)
    document = {
        "address": wallet.address,
        "cipher": KEYSTORE_CIPHER,
        "ciphertext": ciphertext.hex(),
        "format": KEYSTORE_FORMAT,
        "kdf": KEYSTORE_KDF,
        "kdf_params": {
            "n": SCRYPT_N,
            "p": SCRYPT_P,
            "r": SCRYPT_R,
            "salt": salt.hex(),
        },
        "nonce": nonce.hex(),
        "public_key": wallet.public_key.hex(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with suppress(OSError):
        target.chmod(0o600)


def load_keystore(path: str | Path, password: str | bytes) -> Wallet:
    """Decrypt and validate a wallet keystore."""

    document = _load_json_mapping(path)
    metadata = _metadata_from_keystore(document)
    if _expect_str(document, "format") != KEYSTORE_FORMAT:
        raise WalletError("unsupported wallet keystore format")
    if _expect_str(document, "cipher") != KEYSTORE_CIPHER:
        raise WalletError("unsupported wallet keystore cipher")
    if _expect_str(document, "kdf") != KEYSTORE_KDF:
        raise WalletError("unsupported wallet keystore KDF")

    kdf_params = _expect_mapping(document, "kdf_params")
    salt = _decode_hex("salt", _expect_str(kdf_params, "salt"), KEYSTORE_SALT_BYTES)
    nonce = _decode_hex("nonce", _expect_str(document, "nonce"), KEYSTORE_NONCE_BYTES)
    ciphertext = _decode_hex("ciphertext", _expect_str(document, "ciphertext"), None)
    key = _derive_keystore_key(
        _password_bytes(password),
        salt,
        n=_expect_int(kdf_params, "n"),
        r=_expect_int(kdf_params, "r"),
        p=_expect_int(kdf_params, "p"),
    )
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, KEYSTORE_AAD)
    except InvalidTag as exc:
        raise WalletError("wallet password is incorrect or keystore was tampered with") from exc

    payload = _parse_plaintext(plaintext)
    wallet = Wallet.from_seed(_decode_hex("seed", _expect_str(payload, "seed"), 32))
    if wallet.metadata != metadata:
        raise WalletError("keystore metadata does not match encrypted seed")
    return wallet


def read_keystore_metadata(path: str | Path) -> WalletMetadata:
    """Read validated public metadata from a wallet keystore without decrypting it."""

    return _metadata_from_keystore(_load_json_mapping(path))


def balance_for_address(utxos: Iterable[UTXO], address: str) -> int:
    """Return the standard-PKH balance for ``address`` across ``utxos``."""

    owner_hash = _address_hash("address", address)
    return sum(
        utxo.amount_matoms
        for utxo in _require_utxos(utxos)
        if utxo.template_id == TEMPLATE_PKH and utxo.owner_pubkey_hash == owner_hash
    )


def utxo_to_json(utxo: UTXO) -> dict[str, object]:
    """Encode a UTXO into the wallet CLI JSON shape."""

    if not isinstance(utxo, UTXO):
        raise TypeError("utxo must be UTXO")
    return {
        "amount_matoms": utxo.amount_matoms,
        "lockheight": utxo.lockheight,
        "locktime_ms": utxo.locktime_ms,
        "output_index": utxo.outpoint.output_index,
        "owner_pubkey_hash": utxo.owner_pubkey_hash.hex(),
        "payload": utxo.payload.hex(),
        "template_id": utxo.template_id,
        "tx_id": utxo.outpoint.tx_id.hex(),
    }


def utxo_from_json(value: Mapping[str, object]) -> UTXO:
    """Decode a UTXO from the wallet CLI JSON shape."""

    if not isinstance(value, Mapping):
        raise WalletError("UTXO entry must be an object")
    tx_id = _decode_hex("tx_id", _expect_str(value, "tx_id"), 32)
    output_index = _expect_int(value, "output_index")
    if output_index > U32_MAX:
        raise WalletError("output_index outside uint32 range")
    return UTXO(
        outpoint=Outpoint(tx_id, output_index),
        amount_matoms=_expect_int(value, "amount_matoms"),
        template_id=_expect_int(value, "template_id"),
        owner_pubkey_hash=_decode_hex(
            "owner_pubkey_hash",
            _expect_str(value, "owner_pubkey_hash"),
            32,
        ),
        locktime_ms=_expect_int(value, "locktime_ms", default=0),
        lockheight=_expect_int(value, "lockheight", default=0),
        payload=_decode_hex("payload", _expect_str(value, "payload", default=""), None),
    )


def load_utxos_json(path: str | Path) -> tuple[UTXO, ...]:
    """Load a deterministic list of UTXOs from a JSON file."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WalletError(f"failed to read UTXO JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise WalletError("UTXO JSON must contain a list")
    return tuple(utxo_from_json(cast(Mapping[str, object], item)) for item in raw)


def utxos_from_json(path: str | Path) -> tuple[UTXO, ...]:
    """Load UTXOs from a JSON file."""

    return load_utxos_json(path)


def utxos_to_json(utxos: Iterable[UTXO]) -> str:
    """Serialize UTXOs as deterministic JSON."""

    return json.dumps([utxo_to_json(utxo) for utxo in _require_utxos(utxos)], sort_keys=True)


def decode_seed_hex(value: str) -> bytes:
    """Decode a 32-byte wallet seed from lowercase or uppercase hex."""

    return _decode_hex("seed", value, ED25519_PRIVATE_KEY_BYTES)


def _public_key_from_seed(seed: bytes) -> bytes:
    return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()


def _metadata_from_keystore(document: Mapping[str, object]) -> WalletMetadata:
    return WalletMetadata(
        public_key=_decode_hex("public_key", _expect_str(document, "public_key"), 32),
        address=_expect_str(document, "address"),
    )


def _parse_plaintext(plaintext: bytes) -> Mapping[str, object]:
    try:
        raw = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WalletError("keystore plaintext is malformed") from exc
    if not isinstance(raw, Mapping):
        raise WalletError("keystore plaintext must be an object")
    return cast(Mapping[str, object], raw)


def _load_json_mapping(path: str | Path) -> Mapping[str, object]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WalletError(f"failed to read wallet keystore: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise WalletError("wallet keystore must be a JSON object")
    return cast(Mapping[str, object], raw)


def _derive_keystore_key(
    password: bytes,
    salt: bytes,
    *,
    n: int = SCRYPT_N,
    r: int = SCRYPT_R,
    p: int = SCRYPT_P,
) -> bytes:
    _require_exact_bytes("salt", salt, KEYSTORE_SALT_BYTES)
    _require_positive_int("n", n)
    _require_positive_int("r", r)
    _require_positive_int("p", p)
    return Scrypt(salt=salt, length=KEYSTORE_KEY_BYTES, n=n, r=r, p=p).derive(password)


def _password_bytes(password: str | bytes) -> bytes:
    if isinstance(password, str):
        encoded = password.encode("utf-8")
    elif isinstance(password, bytes):
        encoded = password
    else:
        raise TypeError("password must be str or bytes")
    if not encoded:
        raise WalletError("wallet password must not be empty")
    return encoded


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _address_hash(name: str, address: str) -> bytes:
    try:
        return address_to_pubkey_hash(address)
    except (TypeError, ValueError) as exc:
        raise WalletError(f"{name} is invalid") from exc


def _decode_hex(name: str, value: str, expected_len: int | None) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise WalletError(f"{name} must be hex") from exc
    if expected_len is not None and len(decoded) != expected_len:
        raise WalletError(f"{name} must decode to {expected_len} bytes")
    return decoded


def _expect_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise WalletError(f"{key} must be an object")
    return cast(Mapping[str, object], value)


def _expect_str(mapping: Mapping[str, object], key: str, *, default: str | None = None) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str):
        raise WalletError(f"{key} must be a string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise WalletError(f"{key} must be an integer")
    if value < 0:
        raise WalletError(f"{key} must be nonnegative")
    return value


def _require_utxos(utxos: object) -> tuple[UTXO, ...]:
    if isinstance(utxos, bytes | str) or not isinstance(utxos, Iterable):
        raise TypeError("utxos must be an iterable of UTXO values")
    result = tuple(utxos)
    for utxo in result:
        if not isinstance(utxo, UTXO):
            raise TypeError("utxos must contain UTXO values")
    return result


def _require_exact_bytes(name: str, value: bytes, expected_len: int) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != expected_len:
        raise WalletError(f"{name} must be {expected_len} bytes")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value <= 0:
        raise WalletError(f"{name} must be positive")


def _require_u64(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= U64_MAX:
        raise WalletError(f"{name} outside uint64 range")


def _require_positive_u64(name: str, value: int) -> None:
    _require_u64(name, value)
    if value == 0:
        raise WalletError(f"{name} must be nonzero")


__all__ = [
    "KEYSTORE_CIPHER",
    "KEYSTORE_FORMAT",
    "KEYSTORE_KDF",
    "SCRYPT_N",
    "SCRYPT_P",
    "SCRYPT_R",
    "Wallet",
    "WalletError",
    "WalletMetadata",
    "balance_for_address",
    "create_keystore",
    "create_wallet",
    "decode_seed_hex",
    "import_seed_to_keystore",
    "load_keystore",
    "load_utxos_json",
    "load_wallet",
    "read_keystore_metadata",
    "recover_wallet",
    "save_keystore",
    "utxo_from_json",
    "utxo_to_json",
    "utxos_from_json",
    "utxos_to_json",
]
