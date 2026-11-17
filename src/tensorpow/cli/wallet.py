"""Command-line wallet controls for TensorPoW."""

from __future__ import annotations

import argparse
import getpass
import json
import os
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
EXPORT_CONFIRMATION: Final[str] = "show seed"


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
    parser = argparse.ArgumentParser(prog="tensorpow-wallet", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", allow_abbrev=False)
    _add_wallet_args(create)
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(handler=_cmd_create)

    address = subparsers.add_parser("address", allow_abbrev=False)
    _add_wallet_args(address)
    address.set_defaults(handler=_cmd_address)

    balance = subparsers.add_parser("balance", allow_abbrev=False)
    _add_wallet_args(balance)
    balance.add_argument("--utxos", type=Path, required=True)
    balance.set_defaults(handler=_cmd_balance)

    send = subparsers.add_parser("send", allow_abbrev=False)
    _add_wallet_args(send)
    send.add_argument("--utxos", type=Path, required=True)
    send.add_argument("--to", required=True)
    send.add_argument("--amount", type=int, required=True)
    send.add_argument("--fee", type=int, default=DEFAULT_SEND_FEE_MATOMS)
    send.add_argument("--change-address")
    send.add_argument("--out", type=Path)
    send.set_defaults(handler=_cmd_send)

    import_ = subparsers.add_parser("import", allow_abbrev=False)
    _add_wallet_args(import_)
    seed_source = import_.add_mutually_exclusive_group()
    seed_source.add_argument(
        "--seed-env",
        metavar="NAME",
        help="read the hex seed from environment variable NAME",
    )
    seed_source.add_argument(
        "--seed-stdin",
        action="store_true",
        help="read the hex seed from the first line of stdin (or the line after the password)",
    )
    import_.add_argument("--overwrite", action="store_true")
    import_.set_defaults(handler=_cmd_import)

    export = subparsers.add_parser("export", allow_abbrev=False)
    _add_wallet_args(export)
    export.add_argument(
        "--show-seed",
        action="store_true",
        help="include the secret seed (requires interactive confirmation or --yes)",
    )
    export.add_argument("--yes", action="store_true", help="confirm --show-seed without a prompt")
    export.add_argument("--out", type=Path, help="write the export to a 0600 file, not stdout")
    export.set_defaults(handler=_cmd_export)
    return parser


def _add_wallet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wallet", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--password-env",
        metavar="NAME",
        help="read the keystore password from environment variable NAME",
    )
    source.add_argument(
        "--password-stdin",
        action="store_true",
        help="read the keystore password from the first line of stdin",
    )


def _password(args: argparse.Namespace, *, confirm: bool = False) -> str:
    """Read the keystore password without ever taking it from argv."""

    if args.password_env:
        value = os.environ.get(args.password_env)
        if not value:
            raise WalletError(f"environment variable {args.password_env} is not set")
        return value
    if args.password_stdin:
        return _stdin_line("password")
    if not sys.stdin.isatty():
        raise WalletError("no TTY: use --password-stdin or --password-env NAME")
    value = getpass.getpass("Wallet password: ")
    if confirm and getpass.getpass("Repeat password: ") != value:
        raise WalletError("passwords do not match")
    return value


def _seed(args: argparse.Namespace) -> str:
    if args.seed_env:
        value = os.environ.get(args.seed_env)
        if not value:
            raise WalletError(f"environment variable {args.seed_env} is not set")
        return value
    if args.seed_stdin:
        return _stdin_line("seed")
    if not sys.stdin.isatty():
        raise WalletError("no TTY: use --seed-stdin or --seed-env NAME")
    return getpass.getpass("Wallet seed (hex): ")


def _stdin_line(label: str) -> str:
    line = sys.stdin.readline()
    if not line:
        raise WalletError(f"expected the {label} on stdin")
    value = line.rstrip("\r\n")
    if not value:
        raise WalletError(f"the {label} read from stdin is empty")
    return value


def _cmd_create(args: argparse.Namespace) -> int:
    wallet = create_wallet(args.wallet, _password(args, confirm=True), overwrite=args.overwrite)
    _print_json({"address": wallet.address, "wallet": str(Path(args.wallet))})
    return 0


def _cmd_address(args: argparse.Namespace) -> int:
    wallet = load_wallet(args.wallet, _password(args))
    _print_json({"address": wallet.address, "wallet": str(Path(args.wallet))})
    return 0


def _cmd_balance(args: argparse.Namespace) -> int:
    wallet = load_wallet(args.wallet, _password(args))
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
    wallet = load_wallet(args.wallet, _password(args))
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
    password = _password(args, confirm=True)
    wallet = recover_wallet(args.wallet, password, _seed(args), overwrite=args.overwrite)
    _print_json({"address": wallet.address, "wallet": str(Path(args.wallet))})
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    wallet = load_wallet(args.wallet, _password(args))
    payload: dict[str, object] = {"address": wallet.address}
    if args.show_seed:
        if not args.yes:
            if not sys.stdin.isatty():
                raise WalletError("--show-seed without a TTY requires --yes")
            answer = input(f"This prints the wallet seed. Type '{EXPORT_CONFIRMATION}': ")
            if answer.strip() != EXPORT_CONFIRMATION:
                raise WalletError("seed export not confirmed")
        payload["seed"] = wallet.seed_hex
    if args.out is not None:
        _write_private_json(Path(args.out), payload)
        _print_json({"address": wallet.address, "out": str(args.out)})
        return 0
    _print_json(payload)
    return 0


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_raw_tx(path: Path, raw_tx: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_tx.hex() + "\n", encoding="utf-8")
    if Transaction.from_bytes(raw_tx).to_bytes() != raw_tx:
        raise WalletError("transaction serialization round trip failed")


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
