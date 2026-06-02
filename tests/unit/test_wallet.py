"""Tests for encrypted wallets and transaction construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorpow.state.utxo import MAX_SUPPLY_MATOMS, TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.tx.script import verify_utxo_spend
from tensorpow.tx.transaction import Transaction
from tensorpow.wallet import (
    SCRYPT_MAX_N,
    SCRYPT_MAX_P,
    SCRYPT_MAX_R,
    Wallet,
    WalletError,
    load_wallet,
    recover_wallet,
    utxos_to_json,
)


def test_wallet_keystore_round_trip_and_recovery(tmp_path: Path) -> None:
    wallet = Wallet.create()
    path = wallet.save(tmp_path / "wallet.json", "correct horse")

    loaded = load_wallet(path, "correct horse")
    recovered = recover_wallet(tmp_path / "recovered.json", "correct horse", wallet.seed_hex)

    assert loaded.private_key == wallet.private_key
    assert loaded.address == wallet.address
    assert recovered.address == wallet.address
    with pytest.raises(WalletError, match="password"):
        load_wallet(path, "wrong password")
    with pytest.raises(WalletError, match="hex"):
        Wallet.recover("not-hex")


@pytest.mark.parametrize(
    ("param", "value", "message"),
    (
        ("n", SCRYPT_MAX_N * 2, "scrypt n"),
        ("r", SCRYPT_MAX_R + 1, "scrypt r"),
        ("p", SCRYPT_MAX_P + 1, "scrypt p"),
    ),
)
def test_wallet_keystore_rejects_excessive_scrypt_params(
    tmp_path: Path,
    param: str,
    value: int,
    message: str,
) -> None:
    wallet = Wallet.create()
    path = wallet.save(tmp_path / f"{param}.json", "correct horse")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["kdf_params"][param] = value
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(WalletError, match=message):
        load_wallet(path, "correct horse")


def test_wallet_builds_signed_pkh_transaction_and_change() -> None:
    wallet = Wallet.recover("11" * 32)
    recipient = Wallet.recover("22" * 32)
    utxos = (_owned_utxo(wallet, 100, 0),)

    tx = wallet.build_transaction(
        utxos,
        recipient.address,
        amount_matoms=40,
        fee_matoms=3,
    )

    assert tx.outputs[0].amount_matoms == 40
    assert tx.outputs[1].amount_matoms == 57
    assert tx.outputs[0].payload == recipient.pubkey_hash()
    assert tx.outputs[1].payload == wallet.pubkey_hash()
    assert len(tx.inputs[0].witness) == 96
    assert verify_utxo_spend(utxos[0], tx.inputs[0].witness, tx.sighash(0), sig_type=tx.sig_type)
    assert Transaction.from_bytes(tx.to_bytes()) == tx


def test_wallet_balance_and_insufficient_funds() -> None:
    wallet = Wallet.recover("11" * 32)
    other = Wallet.recover("22" * 32)
    utxos = (_owned_utxo(wallet, 7, 0), _owned_utxo(other, 9, 1))

    assert wallet.balance(utxos) == 7
    with pytest.raises(WalletError, match="insufficient"):
        wallet.build_transaction(utxos, other.address, amount_matoms=8, fee_matoms=1)


def test_wallet_filters_locked_and_immature_utxos() -> None:
    wallet = Wallet.recover("11" * 32)
    recipient = Wallet.recover("22" * 32)
    unlocked = _owned_utxo(wallet, 7, 0)
    time_locked = _owned_utxo(wallet, 11, 1, locktime_ms=100)
    height_locked = _owned_utxo(wallet, 13, 2, lockheight=10)
    utxos = (time_locked, height_locked, unlocked)

    assert wallet.owned_utxos(utxos) == (unlocked,)
    assert wallet.balance(utxos) == 7
    assert wallet.balance(utxos, current_time_ms=100, current_height=10) == 31
    with pytest.raises(WalletError, match="insufficient"):
        wallet.build_transaction(utxos, recipient.address, amount_matoms=8, fee_matoms=0)

    tx = wallet.build_transaction(
        utxos,
        recipient.address,
        amount_matoms=8,
        fee_matoms=0,
        current_time_ms=100,
        current_height=10,
    )
    assert {input_.previous_outpoint for input_ in tx.inputs} == {
        unlocked.outpoint,
        time_locked.outpoint,
    }


def test_wallet_wraps_invalid_transaction_construction_errors() -> None:
    wallet = Wallet.recover("11" * 32)
    recipient = Wallet.recover("22" * 32)

    with pytest.raises(WalletError, match="MAX_SUPPLY_MATOMS"):
        wallet.build_transaction(
            (
                _owned_utxo(wallet, MAX_SUPPLY_MATOMS, 0),
                _owned_utxo(wallet, 1, 1),
            ),
            recipient.address,
            amount_matoms=MAX_SUPPLY_MATOMS + 1,
            fee_matoms=0,
        )

    many_small_utxos = tuple(_owned_utxo(wallet, 1, index) for index in range(80))
    with pytest.raises(WalletError, match="transaction exceeds"):
        wallet.build_transaction(
            many_small_utxos,
            recipient.address,
            amount_matoms=len(many_small_utxos),
            fee_matoms=0,
        )


def test_utxo_json_helpers(tmp_path: Path) -> None:
    wallet = Wallet.recover("11" * 32)
    utxo = _owned_utxo(wallet, 7, 0)
    path = tmp_path / "utxos.json"
    path.write_text(utxos_to_json((utxo,)), encoding="utf-8")

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == [
        {
            "amount_matoms": 7,
            "lockheight": 0,
            "locktime_ms": 0,
            "output_index": 0,
            "owner_pubkey_hash": wallet.pubkey_hash().hex(),
            "payload": wallet.pubkey_hash().hex(),
            "template_id": TEMPLATE_PKH,
            "tx_id": bytes([1] * 32).hex(),
        }
    ]


def _owned_utxo(
    wallet: Wallet,
    amount: int,
    index: int,
    *,
    locktime_ms: int = 0,
    lockheight: int = 0,
) -> UTXO:
    outpoint = Outpoint(bytes([index + 1]) * 32, index)
    return UTXO(
        outpoint=outpoint,
        amount_matoms=amount,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=wallet.pubkey_hash(),
        locktime_ms=locktime_ms,
        lockheight=lockheight,
        payload=wallet.pubkey_hash(),
    )
