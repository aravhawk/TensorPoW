"""Command-line TensorPoW mining smoke tool."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from threading import Event

from tensorpow.crypto.hash import HASH_LEN_BYTES
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH, FruitPowHeader
from tensorpow.pow.kernel import FRUIT_TARGET_LE
from tensorpow.pow.miner import mine


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``tensorpow-mine`` CLI."""

    parser = argparse.ArgumentParser(prog="tensorpow-mine")
    parser.add_argument(
        "--target", default=FRUIT_TARGET_LE.hex(), help="32-byte little-endian target hex"
    )
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--start-nonce", type=int, default=0)
    parser.add_argument("--max-nonce", type=int)
    args = parser.parse_args(argv)

    try:
        target = _decode_target(args.target)
        template = FruitPowHeader(
            version=FORMAT_EPOCH,
            sig_type_supported=SIG_TYPE_ED25519_BIT,
            effective_parent_hashes=(GENESIS_PARENT_HASH,),
            latest_anchor=GENESIS_PARENT_HASH,
            tx_merkle_root=GENESIS_PARENT_HASH,
            timestamp_ms=0,
            shard_id=0,
            nonce=args.start_nonce,
        )
        result = mine(
            template,
            target,
            Event(),
            backend=args.backend,
            start_nonce=args.start_nonce,
            max_nonce=args.max_nonce,
        )
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result is None:
        print(json.dumps({"found": False}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "attempts": result.attempts,
                "backend": result.backend,
                "digest": result.digest.hex(),
                "elapsed_seconds": result.elapsed_seconds,
                "found": True,
                "nonce": result.nonce,
            },
            sort_keys=True,
        )
    )
    return 0


def _decode_target(value: str) -> bytes:
    try:
        target = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("target must be hex") from exc
    if len(target) != HASH_LEN_BYTES:
        raise ValueError(f"target must decode to {HASH_LEN_BYTES} bytes")
    return target


if __name__ == "__main__":
    raise SystemExit(main())
