#!/usr/bin/env python
"""Benchmark deterministic TensorPoW matrix work."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from time import perf_counter

import torch

from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.pow.challenge import (
    FORMAT_EPOCH,
    GENESIS_PARENT_HASH,
    FruitPowHeader,
    build_challenge_matrices,
    with_nonce,
)
from tensorpow.pow.kernel import (
    FRUIT_WORK_OPS,
    POW_ACCUM_BYTES,
    POW_MATRIX_BYTES,
    POW_OPS_PER_MATMUL,
    matmul_int8,
    pow_digest,
    resolve_backend,
)
from tensorpow.pow.verify import pow_digest_for_header


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TensorPoW matmul throughput")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    header = FruitPowHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        effective_parent_hashes=(GENESIS_PARENT_HASH,),
        latest_anchor=GENESIS_PARENT_HASH,
        tx_merkle_root=GENESIS_PARENT_HASH,
        timestamp_ms=0,
        shard_id=0,
        nonce=0,
    )
    left, right = build_challenge_matrices(header)
    resolved_backend = resolve_backend(args.backend)

    kernel_timings = []
    kernel_digest = b""
    for _ in range(args.warmup):
        kernel_digest = pow_digest(matmul_int8(left, right, backend=args.backend))

    for _ in range(args.iterations):
        start = perf_counter()
        output = matmul_int8(left, right, backend=args.backend)
        kernel_digest = pow_digest(output)
        kernel_timings.append(perf_counter() - start)

    full_attempt_timings = []
    full_attempt_digest = b""
    for nonce in range(args.warmup):
        full_attempt_digest = pow_digest_for_header(with_nonce(header, nonce), backend=args.backend)

    for nonce in range(args.warmup, args.warmup + args.iterations):
        start = perf_counter()
        full_attempt_digest = pow_digest_for_header(with_nonce(header, nonce), backend=args.backend)
        full_attempt_timings.append(perf_counter() - start)

    best_seconds = min(kernel_timings)
    best_full_attempt_seconds = min(full_attempt_timings)
    nonces_per_second_best = 1 / best_full_attempt_seconds
    dim = left.shape[0]
    ops = POW_OPS_PER_MATMUL
    expected_attempts_per_fruit = FRUIT_WORK_OPS / ops
    input_bytes = 2 * POW_MATRIX_BYTES
    output_bytes = dim * dim * POW_ACCUM_BYTES
    memory_floor_bytes = input_bytes + output_bytes + output_bytes
    result = {
        "backend": args.backend,
        "best_seconds": best_seconds,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "estimated_fruits_per_second_best": nonces_per_second_best / expected_attempts_per_fruit,
        "expected_attempts_per_fruit": expected_attempts_per_fruit,
        "fruit_work_ops": FRUIT_WORK_OPS,
        "full_attempt_last_digest": full_attempt_digest.hex(),
        "full_attempt_best_seconds": best_full_attempt_seconds,
        "full_attempt_mean_seconds": statistics.fmean(full_attempt_timings),
        "full_attempt_timings_seconds": full_attempt_timings,
        "gops": ops / best_seconds / 1_000_000_000,
        "iterations": args.iterations,
        "kernel_mean_seconds": statistics.fmean(kernel_timings),
        "kernel_reference_digest": kernel_digest.hex(),
        "memory_floor_bytes": memory_floor_bytes,
        "mps_available": torch.backends.mps.is_available(),
        "nonces_per_second_best": nonces_per_second_best,
        "ops_per_matmul": ops,
        "platform": platform.platform(),
        "resolved_backend": resolved_backend,
        "kernel_timings_seconds": kernel_timings,
        "tops": ops / best_seconds / 1_000_000_000_000,
        "torch_version": torch.__version__,
        "warmup": args.warmup,
    }

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
