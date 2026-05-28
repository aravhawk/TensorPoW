"""Train the deterministic INT8 learned transaction codec prior."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Final

import numpy as np

from tensorpow.codec.learned import (
    INT8_ZERO_POINT,
    LEARNED_CODEC_WEIGHTS_PATH,
)
from tensorpow.codec.template import compress_tx
from tensorpow.crypto.hash import hash_bytes
from tensorpow.crypto.signatures import ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES
from tensorpow.state.utxo import TEMPLATE_HASHLOCK, TEMPLATE_MULTISIG, TEMPLATE_PKH, Outpoint
from tensorpow.tx.script import pubkey_hash
from tensorpow.tx.transaction import FORMAT_EPOCH, Input, Output, Transaction

DEFAULT_SAMPLES: Final[int] = 4096
DEFAULT_MAX_MODEL_LEN: Final[int] = 1024
DEFAULT_SEED: Final[int] = 55


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TensorPoW learned tx codec INT8 position priors."
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--output",
        type=Path,
        default=LEARNED_CODEC_WEIGHTS_PATH,
        help="Output .npz path for frozen codec weights.",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.max_model_len <= 0:
        raise SystemExit("--max-model-len must be positive")

    rng = random.Random(args.seed)
    position_counts = [Counter[int]() for _ in range(args.max_model_len)]
    fallback_counts: Counter[int] = Counter()

    for _ in range(args.samples):
        encoded = compress_tx(_simulate_tx(rng))
        fallback_counts.update(encoded)
        for position, value in enumerate(encoded[: args.max_model_len]):
            position_counts[position][value] += 1

    fallback_byte = _most_common_byte(fallback_counts)
    prior_bytes = bytes(
        _most_common_byte(counts, fallback=fallback_byte) for counts in position_counts
    )
    position_prior = np.frombuffer(
        bytes((value - INT8_ZERO_POINT) % 256 for value in prior_bytes),
        dtype=np.int8,
    ).copy()
    fallback_prior = np.array(fallback_byte - INT8_ZERO_POINT, dtype=np.int8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        position_prior=position_prior,
        fallback_prior=fallback_prior,
        sample_count=np.array(args.samples, dtype=np.uint32),
        seed=np.array(args.seed, dtype=np.uint64),
    )
    print(f"wrote {args.output} ({args.samples} simulated transactions)")


def _simulate_tx(rng: random.Random) -> Transaction:
    input_count = 1 + (rng.randrange(10) == 0)
    inputs = tuple(_simulate_input(rng, index) for index in range(input_count))
    signer_hashes = [
        pubkey_hash(input_.witness[ED25519_SIGNATURE_BYTES:])
        for input_ in inputs
        if len(input_.witness) == ED25519_SIGNATURE_BYTES + ED25519_PUBLIC_KEY_BYTES
    ]
    outputs = [_simulate_pkh_output(rng, signer_hashes)]
    if rng.randrange(3) == 0:
        outputs.append(_simulate_pkh_output(rng, signer_hashes))
    if rng.randrange(8) == 0:
        outputs.append(_simulate_multisig_output(rng))
    if rng.randrange(12) == 0:
        outputs.append(_simulate_hashlock_output(rng))

    return Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=0 if rng.randrange(20) else rng.randrange(1, 1_000_000),
        lockheight=0 if rng.randrange(20) else rng.randrange(1, 100_000),
        inputs=inputs,
        outputs=tuple(outputs),
    )


def _simulate_input(rng: random.Random, index: int) -> Input:
    tx_id = bytes(32) if rng.randrange(6) == 0 else _rng_bytes(rng, 32)
    outpoint = Outpoint(tx_id, index if rng.randrange(4) else rng.randrange(8))
    signature = _rng_bytes(rng, ED25519_SIGNATURE_BYTES)
    public_key = _rng_bytes(rng, ED25519_PUBLIC_KEY_BYTES)
    return Input(outpoint, witness=signature + public_key)


def _simulate_pkh_output(rng: random.Random, signer_hashes: list[bytes]) -> Output:
    if signer_hashes and rng.randrange(2) == 0:
        payload = signer_hashes[rng.randrange(len(signer_hashes))]
    else:
        payload = hash_bytes(_rng_bytes(rng, 16))
    return Output(
        amount_matoms=1 + rng.randrange(10_000_000),
        template_id=TEMPLATE_PKH,
        payload=payload,
    )


def _simulate_multisig_output(rng: random.Random) -> Output:
    pubkey_count = 2 + rng.randrange(3)
    threshold = 1 + rng.randrange(pubkey_count)
    payload = bytes((threshold, pubkey_count)) + b"".join(
        _rng_bytes(rng, ED25519_PUBLIC_KEY_BYTES) for _ in range(pubkey_count)
    )
    return Output(
        amount_matoms=1 + rng.randrange(10_000_000),
        template_id=TEMPLATE_MULTISIG,
        payload=payload,
    )


def _simulate_hashlock_output(rng: random.Random) -> Output:
    payload = hash_bytes(_rng_bytes(rng, 16)) + hash_bytes(_rng_bytes(rng, 16))
    return Output(
        amount_matoms=1 + rng.randrange(10_000_000),
        template_id=TEMPLATE_HASHLOCK,
        payload=payload,
    )


def _rng_bytes(rng: random.Random, length: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(length))


def _most_common_byte(counts: Counter[int], *, fallback: int = 0) -> int:
    if not counts:
        return fallback
    return min((-count, value) for value, count in counts.items())[1]


if __name__ == "__main__":
    main()
