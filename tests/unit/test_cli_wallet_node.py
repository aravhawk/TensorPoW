"""Tests for wallet and node CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tensorpow.cli.mine import main as mine_main
from tensorpow.cli.node import main as node_main
from tensorpow.cli.wallet import DEFAULT_SEND_FEE_MATOMS
from tensorpow.cli.wallet import main as wallet_main
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.tx.transaction import Transaction
from tensorpow.wallet import Wallet, utxos_to_json


def test_wallet_cli_create_address_export_import_balance_and_send(
    tmp_path: Path,
    capsys: object,
) -> None:
    wallet_path = tmp_path / "wallet.json"

    assert wallet_main(["create", "--wallet", str(wallet_path), "--password", "pw"]) == 0
    created = _read_stdout(capsys)
    assert created["address"].startswith("tsc1")
    assert wallet_main(["create", "--wallet", str(wallet_path), "--password", "pw"]) == 1
    assert _read_stderr(capsys).startswith("error: wallet keystore already exists")

    assert wallet_main(["address", "--wallet", str(wallet_path), "--password", "pw"]) == 0
    assert _read_stdout(capsys)["address"] == created["address"]

    assert wallet_main(["export", "--wallet", str(wallet_path), "--password", "pw"]) == 0
    seed = _read_stdout(capsys)["seed"]
    imported_path = tmp_path / "imported.json"
    assert (
        wallet_main(
            ["import", "--wallet", str(imported_path), "--password", "pw", "--seed", str(seed)]
        )
        == 0
    )
    assert _read_stdout(capsys)["address"] == created["address"]

    wallet = Wallet.load(wallet_path, "pw")
    recipient = Wallet.recover("22" * 32)
    utxo = UTXO(
        outpoint=Outpoint(bytes.fromhex("11" * 32), 0),
        amount_matoms=50,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=wallet.pubkey_hash(),
        payload=wallet.pubkey_hash(),
    )
    utxo_path = tmp_path / "utxos.json"
    utxo_path.write_text(utxos_to_json((utxo,)), encoding="utf-8")

    assert (
        wallet_main(
            [
                "balance",
                "--wallet",
                str(wallet_path),
                "--password",
                "pw",
                "--utxos",
                str(utxo_path),
            ]
        )
        == 0
    )
    assert _read_stdout(capsys)["balance_matoms"] == 50

    assert (
        wallet_main(
            [
                "send",
                "--wallet",
                str(wallet_path),
                "--password",
                "pw",
                "--utxos",
                str(utxo_path),
                "--to",
                recipient.address,
                "--amount",
                "20",
                "--fee",
                "1",
            ]
        )
        == 0
    )
    sent = _read_stdout(capsys)
    assert len(sent["tx_id"]) == 64
    assert len(sent["tx"]) > 64

    assert (
        wallet_main(
            [
                "send",
                "--wallet",
                str(wallet_path),
                "--password",
                "pw",
                "--utxos",
                str(utxo_path),
                "--to",
                recipient.address,
                "--amount",
                "20",
            ]
        )
        == 0
    )
    default_fee_tx = Transaction.from_bytes(bytes.fromhex(str(_read_stdout(capsys)["tx"])))
    assert utxo.amount_matoms - sum(output.amount_matoms for output in default_fee_tx.outputs) == (
        DEFAULT_SEND_FEE_MATOMS
    )


def test_node_cli_init_start_status_peers_stop(tmp_path: Path, capsys: object) -> None:
    data_dir = tmp_path / "data"

    assert node_main(["init", "--data-dir", str(data_dir)]) == 0
    assert _read_stdout(capsys)["data_dir"] == str(data_dir)

    assert node_main(["start", "--data-dir", str(data_dir)]) == 0
    assert _read_stdout(capsys)["running"]
    assert node_main(["status", "--data-dir", str(data_dir)]) == 0
    assert _read_stdout(capsys)["running"]
    assert node_main(["peers", "--data-dir", str(data_dir)]) == 0
    assert _read_stdout(capsys)["peer_count"] == 0
    assert node_main(["stop", "--data-dir", str(data_dir)]) == 0
    assert not _read_stdout(capsys)["running"]


def test_mine_cli_finds_nonce_and_rejects_bad_targets(capsys: object) -> None:
    assert (
        mine_main(
            [
                "--target",
                "ff" * 32,
                "--backend",
                "cpu",
                "--start-nonce",
                "7",
                "--max-nonce",
                "7",
            ]
        )
        == 0
    )
    found = _read_stdout(capsys)
    assert found["found"] is True
    assert found["backend"] == "cpu"
    assert found["nonce"] == 7
    assert found["attempts"] == 1
    assert len(str(found["digest"])) == 64

    assert mine_main(["--target", "not-hex", "--backend", "cpu"]) == 1
    assert _read_stderr(capsys) == "error: target must be hex"
    assert mine_main(["--target", "00", "--backend", "cpu"]) == 1
    assert _read_stderr(capsys) == "error: target must decode to 32 bytes"
    assert (
        mine_main(
            [
                "--target",
                "ff" * 32,
                "--backend",
                "cpu",
                "--start-nonce",
                "9",
                "--max-nonce",
                "8",
            ]
        )
        == 1
    )
    assert _read_stderr(capsys) == "error: max_nonce outside uint64 range or before start_nonce"


def _read_stdout(capsys: object) -> dict[str, Any]:
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert isinstance(payload, dict)
    return dict(payload)


def _read_stderr(capsys: object) -> str:
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    return str(captured.err).strip()
