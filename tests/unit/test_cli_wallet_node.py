"""Tests for wallet and node CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tensorpow.cli.node as node_cli
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPW_TEST_PASSWORD", "pw")
    wallet_path = tmp_path / "wallet.json"

    assert (
        wallet_main(["create", "--wallet", str(wallet_path), "--password-env", "TPW_TEST_PASSWORD"])
        == 0
    )
    created = _read_stdout(capsys)
    assert created["address"].startswith("tsc1")
    assert (
        wallet_main(["create", "--wallet", str(wallet_path), "--password-env", "TPW_TEST_PASSWORD"])
        == 1
    )
    assert _read_stderr(capsys).startswith("error: wallet keystore already exists")

    assert (
        wallet_main(
            ["address", "--wallet", str(wallet_path), "--password-env", "TPW_TEST_PASSWORD"]
        )
        == 0
    )
    assert _read_stdout(capsys)["address"] == created["address"]

    export_args = ["export", "--wallet", str(wallet_path), "--password-env", "TPW_TEST_PASSWORD"]
    assert wallet_main(export_args) == 0
    assert "seed" not in _read_stdout(capsys)
    assert wallet_main([*export_args, "--show-seed", "--yes"]) == 0
    seed = _read_stdout(capsys)["seed"]
    monkeypatch.setenv("TPW_TEST_SEED", str(seed))
    imported_path = tmp_path / "imported.json"
    assert (
        wallet_main(
            [
                "import",
                "--wallet",
                str(imported_path),
                "--password-env",
                "TPW_TEST_PASSWORD",
                "--seed-env",
                "TPW_TEST_SEED",
            ]
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
                "--password-env",
                "TPW_TEST_PASSWORD",
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
                "--password-env",
                "TPW_TEST_PASSWORD",
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
                "--password-env",
                "TPW_TEST_PASSWORD",
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
    stopped = _read_stdout(capsys)
    assert not stopped["running"]
    assert stopped["was_running"]


def test_node_stop_never_signals_an_unrelated_process_from_a_stale_pidfile(
    tmp_path: Path,
    capsys: object,
) -> None:
    """L-10: a stale pidfile whose PID now belongs to another process is not killed."""

    import subprocess
    import sys

    data_dir = tmp_path / "data"
    assert node_main(["init", "--data-dir", str(data_dir)]) == 0
    _read_stdout(capsys)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (data_dir / node_cli.PID_FILENAME).write_text(f"{bystander.pid}\n", encoding="utf-8")
        assert node_main(["status", "--data-dir", str(data_dir)]) == 0
        assert not _read_stdout(capsys)["running"]

        (data_dir / node_cli.PID_FILENAME).write_text(f"{bystander.pid}\n", encoding="utf-8")
        assert node_main(["stop", "--data-dir", str(data_dir)]) == 0
        assert _read_stdout(capsys)["was_running"] is False
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait(timeout=5)


def test_node_pid_file_acquisition_refuses_active_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    active_pid = 101
    competing_pid = 202
    monkeypatch.setattr(node_cli, "_pid_is_running", lambda pid: pid == active_pid)

    node_cli._write_pid(data_dir, active_pid)
    with pytest.raises(node_cli.NodeCliError, match="node is already running"):
        node_cli._write_pid(data_dir, competing_pid)

    assert node_cli._read_pid(data_dir) == active_pid
    node_cli._remove_pid_if_matches(data_dir, competing_pid)
    assert node_cli._read_pid(data_dir) == active_pid


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


def test_wallet_cli_never_accepts_secrets_on_argv(tmp_path: Path) -> None:
    """M-7: --password / --seed values on the command line are refused."""

    wallet_path = tmp_path / "wallet.json"
    with pytest.raises(SystemExit):
        wallet_main(["create", "--wallet", str(wallet_path), "--password", "pw"])
    with pytest.raises(SystemExit):
        wallet_main(["import", "--wallet", str(wallet_path), "--seed", "11" * 32])
    assert not wallet_path.exists()


def test_wallet_cli_reads_password_and_seed_from_stdin(
    tmp_path: Path,
    capsys: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    wallet_path = tmp_path / "wallet.json"
    monkeypatch.setattr("sys.stdin", io.StringIO("stdin-pw\n"))
    assert wallet_main(["create", "--wallet", str(wallet_path), "--password-stdin"]) == 0
    created = _read_stdout(capsys)

    imported = tmp_path / "imported.json"
    monkeypatch.setattr("sys.stdin", io.StringIO("stdin-pw\n" + "33" * 32 + "\n"))
    assert (
        wallet_main(["import", "--wallet", str(imported), "--password-stdin", "--seed-stdin"]) == 0
    )
    assert _read_stdout(capsys)["address"] != created["address"]

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert wallet_main(["address", "--wallet", str(wallet_path)]) == 1
    assert "no TTY" in _read_stderr(capsys)


def test_wallet_cli_seed_export_requires_confirmation(
    tmp_path: Path,
    capsys: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import stat

    monkeypatch.setenv("TPW_TEST_PASSWORD", "pw")
    wallet_path = tmp_path / "wallet.json"
    base = ["--wallet", str(wallet_path), "--password-env", "TPW_TEST_PASSWORD"]
    assert wallet_main(["create", *base]) == 0
    _read_stdout(capsys)

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert wallet_main(["export", *base, "--show-seed"]) == 1
    assert "requires --yes" in _read_stderr(capsys)

    out = tmp_path / "export.json"
    assert wallet_main(["export", *base, "--show-seed", "--yes", "--out", str(out)]) == 0
    printed = _read_stdout(capsys)
    assert "seed" not in printed
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert len(json.loads(out.read_text(encoding="utf-8"))["seed"]) == 64
