"""Command-line wallet controls for TensorPoW."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, cast

from tensorpow.tx import Transaction
from tensorpow.wallet import (
    WalletError,
    create_wallet,
    load_utxos_json,
    load_wallet,
    recover_wallet,
)

type _Command = Callable[[argparse.Namespace], int]

DEFAULT_SEND_FEE_MATOMS: Final[int] = 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``tensorpow-wallet`` CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = cast(_Command, args.handler)
    try:
        return handler(args)
    except (OSError, TypeError, ValueError, WalletError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensorpow-wallet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    _add_wallet_args(create)
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(handler=_cmd_create)

    address = subparsers.add_parser("address")
    _add_wallet_args(address)
    address.set_defaults(handler=_cmd_address)

    balance = subparsers.add_parser("balance")
    _add_wallet_args(balance)
    balance.add_argument("--utxos", type=Path, required=True)
    balance.set_defaults(handler=_cmd_balance)

    send = subparsers.add_parser("send")
    _add_wallet_args(send)
    send.add_argument("--utxos", type=Path, required=True)
    send.add_argument("--to", required=True)
    send.add_argument("--amount", type=int, required=True)
    send.add_argument("--fee", type=int, default=DEFAULT_SEND_FEE_MATOMS)
    send.add_argument("--change-address")
    send.add_argument("--out", type=Path)
    send.set_defaults(handler=_cmd_send)

    import_ = subparsers.add_parser("import")
    _add_wallet_args(import_)
    import_.add_argument("--seed", required=True)
    import_.add_argument("--overwrite", action="store_true")
    import_.set_defaults(handler=_cmd_import)

    export = subparsers.add_parser("export")
    _add_wallet_args(export)
    export.set_defaults(handler=_cmd_export)
    return parser


def _add_wallet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wallet", type=Path, required=True)
    parser.add_argument("--password", required=True)


def _cmd_create(args: argparse.Namespace) -> int:
    wallet = create_wallet(args.wallet, args.password, overwrite=args.overwrite)
    _print_json({"address": wallet.address, "wallet": str(Path(args.wallet))})
    return 0


def _cmd_address(args: argparse.Namespace) -> int:
    wallet = load_wallet(args.wallet, args.password)
    _print_json({"address": wallet.address, "wallet": str(Path(args.wallet))})
    return 0


def _cmd_balance(args: argparse.Namespace) -> int:
    wallet = load_wallet(args.wallet, args.password)
    utxos = load_utxos_json(args.utxos)
    _print_json(
        {
            "address": wallet.address,
            "balance_matoms": wallet.balance_matoms(utxos),
            "owned_utxo_count": len(wallet.owned_utxos(utxos)),
        }
    )
    return 0


def _cmd_send(args: argparse.Namespace) -> int:
    wallet = load_wallet(args.wallet, args.password)
    tx = wallet.create_signed_transaction(
        utxos=load_utxos_json(args.utxos),
        recipient_address=args.to,
        amount_matoms=args.amount,
        fee_matoms=args.fee,
        change_address=args.change_address,
    )
    raw_tx = tx.to_bytes()
    if args.out is not None:
        _write_raw_tx(Path(args.out), raw_tx)
    _print_json({"tx": raw_tx.hex(), "tx_id": tx.tx_id().hex()})
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    wallet = recover_wallet(args.wallet, args.password, args.seed, overwrite=args.overwrite)
    _print_json({"address": wallet.address, "wallet": str(Path(args.wallet))})
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    wallet = load_wallet(args.wallet, args.password)
    _print_json({"address": wallet.address, "seed": wallet.seed_hex})
    return 0


def _write_raw_tx(path: Path, raw_tx: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_tx.hex() + "\n", encoding="utf-8")
    if Transaction.from_bytes(raw_tx).to_bytes() != raw_tx:
        raise WalletError("transaction serialization round trip failed")


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
