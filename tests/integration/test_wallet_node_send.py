"""Wallet-to-node local send integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from tensorpow.cli.wallet import main as wallet_main
from tensorpow.node import TensorPowConfig, TensorPowNode
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.tx.transaction import Transaction
from tensorpow.wallet import Wallet, utxos_to_json


def test_wallet_cli_send_is_accepted_by_local_node(tmp_path: Path, capsys: object) -> None:
    wallet = Wallet.recover("11" * 32)
    recipient = Wallet.recover("22" * 32)
    wallet_path = wallet.save(tmp_path / "wallet.json", "pw")
    node = TensorPowNode(TensorPowConfig(data_dir=tmp_path / "node"))
    try:
        funded_utxo = UTXO(
            outpoint=Outpoint(bytes.fromhex("aa" * 32), 0),
            amount_matoms=100,
            template_id=TEMPLATE_PKH,
            owner_pubkey_hash=wallet.pubkey_hash(),
            payload=wallet.pubkey_hash(),
        )
        node.utxo_set.add(funded_utxo)
        utxo_path = tmp_path / "utxos.json"
        utxo_path.write_text(utxos_to_json((funded_utxo,)), encoding="utf-8")

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
                    "40",
                    "--fee",
                    "2",
                ]
            )
            == 0
        )
        sent = _read_stdout(capsys)
        tx = Transaction.from_bytes(bytes.fromhex(str(sent["tx"])))

        node_result, mempool_result = node.process_raw_tx(tx.to_bytes())

        assert node_result.accepted
        assert node_result.object_hash == tx.tx_id()
        assert mempool_result is not None
        assert mempool_result.accepted
        assert node.get_tx(tx.tx_id()) == tx
    finally:
        node.close()


def _read_stdout(capsys: object) -> dict[str, object]:
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert isinstance(payload, dict)
    return payload
