#!/usr/bin/env python
"""Deterministic TensorPoW consensus soak runner."""

from __future__ import annotations

import argparse
import json
import math
import platform as platform_module
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, sleep
from typing import Final, Literal, cast

from blake3 import blake3

from tensorpow.chain.blocks import (
    Anchor,
    FeeFloorEntry,
    Fruit,
    anchor_reward_root,
    fee_floor_set_root,
    fruit_set_root,
    parent_candidate_root,
    tx_merkle_root,
)
from tensorpow.chain.headers import AnchorHeader, FruitHeader
from tensorpow.consensus.anchor_daa import ANCHOR_INTERVAL_MS
from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT
from tensorpow.genesis import GENESIS_CHAIN_ID_TESTNET, GenesisInputs, build_genesis_artifact
from tensorpow.mempool import ROOT_SHARD_ID, ShardTree
from tensorpow.node import NodeResult, TensorPowConfig, TensorPowNode
from tensorpow.pow.challenge import FORMAT_EPOCH, GENESIS_PARENT_HASH
from tensorpow.pow.kernel import Backend, resolve_backend
from tensorpow.pow.verify import pow_digest_for_header
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO
from tensorpow.tx.transaction import Output, Transaction
from tensorpow.wallet import Wallet

SOAK_TEST_DURATION_HOURS: Final[int] = 6
DEFAULT_DURATION_SECONDS: Final[float] = SOAK_TEST_DURATION_HOURS * 60 * 60
DEFAULT_MAX_STEPS: Final[int | None] = None
DEFAULT_ANCHOR_INTERVAL: Final[int] = 3
DURATION_STEP_CADENCE_SECONDS: Final[float] = 12.0
TEST_MODE_STEPS: Final[int] = 3
BASE_TIMESTAMP_MS: Final[int] = 1_779_841_408_000
WALLET_COUNT: Final[int] = 4
VALID_BACKENDS: Final[frozenset[str]] = frozenset(("auto", "cpu", "cuda", "mps"))
CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    (
        "anchor_interval",
        "data_dir",
        "duration_hours",
        "duration_seconds",
        "max_steps",
        "platforms",
        "report_path",
        "seed",
    )
)
COMPARED_FIELDS: Final[tuple[str, ...]] = (
    "final_state_root",
    "utxo_outputs_digest",
    "consensus_bytes_digest",
    "utxo_count",
    "tx_count",
    "fruit_count",
    "anchor_count",
)
FINAL_COMPARE_REQUIRED_PLATFORMS: Final[dict[str, Literal["cpu", "cuda", "mps"]]] = {
    "mac-arm-cpu": "cpu",
    "mac-arm-mps": "mps",
    "ubuntu-x86-cpu": "cpu",
    "ubuntu-x86-cuda": "cuda",
}
FINAL_COMPARE_ELAPSED_TOLERANCE_SECONDS: Final[float] = 2.0


class ConfigError(ValueError):
    """Raised when a soak config is malformed or unsafe to run."""


class ReportError(ValueError):
    """Raised when a saved soak report is malformed."""


class DeterminismMismatchError(RuntimeError):
    """Raised when platform labels produce different consensus results."""


@dataclass(frozen=True, slots=True)
class PlatformRun:
    """One platform label/backend pair in a soak comparison."""

    label: str
    backend: Backend = "auto"

    def __post_init__(self) -> None:
        _require_label(self.label)
        _parse_backend(self.backend)

    def to_json_obj(self) -> dict[str, str]:
        return {"backend": self.backend, "label": self.label}


@dataclass(frozen=True, slots=True)
class SoakConfig:
    """Validated deterministic workload configuration."""

    duration_seconds: float = DEFAULT_DURATION_SECONDS
    max_steps: int | None = DEFAULT_MAX_STEPS
    platforms: tuple[PlatformRun, ...] = field(
        default_factory=lambda: (PlatformRun("local-auto", "auto"),)
    )
    seed: str = "tensorpow-soak"
    anchor_interval: int = DEFAULT_ANCHOR_INTERVAL
    data_dir: Path | None = None
    report_path: Path | None = None

    def __post_init__(self) -> None:
        duration_seconds = _nonnegative_float("duration_seconds", self.duration_seconds)
        object.__setattr__(self, "duration_seconds", duration_seconds)

        if self.max_steps is not None:
            _require_positive_int("max_steps", self.max_steps)
        if duration_seconds == 0 and self.max_steps is None:
            raise ConfigError("duration_seconds must be positive when max_steps is omitted")

        if not isinstance(self.platforms, tuple) or not self.platforms:
            raise ConfigError("platforms must contain at least one platform")
        labels = tuple(platform.label for platform in self.platforms)
        if len(set(labels)) != len(labels):
            raise ConfigError("platform labels must be unique")
        for platform in self.platforms:
            if not isinstance(platform, PlatformRun):
                raise ConfigError("platforms must contain PlatformRun objects")

        _require_nonempty_str("seed", self.seed)
        _require_positive_int("anchor_interval", self.anchor_interval)
        if self.data_dir is not None:
            object.__setattr__(self, "data_dir", Path(self.data_dir))
        if self.report_path is not None:
            object.__setattr__(self, "report_path", Path(self.report_path))

    def to_json_obj(self) -> dict[str, object]:
        return {
            "anchor_interval": self.anchor_interval,
            "data_dir": None if self.data_dir is None else str(self.data_dir),
            "duration_seconds": self.duration_seconds,
            "max_steps": self.max_steps,
            "platforms": [platform.to_json_obj() for platform in self.platforms],
            "report_path": None if self.report_path is None else str(self.report_path),
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class SoakResult:
    """Consensus result from one platform label."""

    label: str
    backend: Backend
    resolved_backend: Literal["cpu", "cuda", "mps"]
    platform: str
    python_version: str
    steps: int
    elapsed_seconds: float
    final_state_root: str
    utxo_outputs_digest: str
    consensus_bytes_digest: str
    last_pow_digest: str
    utxo_count: int
    tx_count: int
    fruit_count: int
    anchor_count: int
    consensus_object_count: int

    def compared_values(self) -> dict[str, str | int]:
        return {field: getattr(self, field) for field in COMPARED_FIELDS}

    def to_json_obj(self) -> dict[str, object]:
        return {
            "anchor_count": self.anchor_count,
            "backend": self.backend,
            "consensus_bytes_digest": self.consensus_bytes_digest,
            "consensus_object_count": self.consensus_object_count,
            "elapsed_seconds": self.elapsed_seconds,
            "final_state_root": self.final_state_root,
            "fruit_count": self.fruit_count,
            "label": self.label,
            "last_pow_digest": self.last_pow_digest,
            "platform": self.platform,
            "python_version": self.python_version,
            "resolved_backend": self.resolved_backend,
            "steps": self.steps,
            "tx_count": self.tx_count,
            "utxo_count": self.utxo_count,
            "utxo_outputs_digest": self.utxo_outputs_digest,
        }


@dataclass(frozen=True, slots=True)
class SoakReport:
    """Complete multi-label soak report."""

    config: SoakConfig
    results: tuple[SoakResult, ...]
    comparison: dict[str, object]

    def to_json_obj(self) -> dict[str, object]:
        return {
            "comparison": self.comparison,
            "config": self.config.to_json_obj(),
            "results": [result.to_json_obj() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class _LoadedReport:
    path: Path
    config: SoakConfig
    results: tuple[SoakResult, ...]


class _Transcript:
    """Streaming digest of consensus-critical bytes in workload order."""

    def __init__(self) -> None:
        self._hasher = blake3()
        self.count = 0

    def record(self, kind: str, data: bytes) -> None:
        _require_nonempty_str("kind", kind)
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        kind_bytes = kind.encode("ascii")
        self._hasher.update(len(kind_bytes).to_bytes(1, "little"))
        self._hasher.update(kind_bytes)
        self._hasher.update(len(data).to_bytes(8, "little"))
        self._hasher.update(data)
        self.count += 1

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def run_soak(config: SoakConfig) -> SoakReport:
    """Run the deterministic workload for every platform and compare results."""

    if not isinstance(config, SoakConfig):
        raise TypeError("config must be SoakConfig")
    with ThreadPoolExecutor(max_workers=len(config.platforms)) as executor:
        results = tuple(
            executor.map(lambda platform: _run_platform(config, platform), config.platforms)
        )
    comparison = compare_results(results)
    return SoakReport(config=config, results=results, comparison=comparison)


def compare_results(results: Sequence[SoakResult]) -> dict[str, object]:
    """Compare final state, UTXO outputs, and byte transcripts across labels."""

    if not results:
        raise ValueError("results must not be empty")
    labels = tuple(result.label for result in results)
    if len(set(labels)) != len(labels):
        raise ValueError("result labels must be unique")
    baseline = results[0]
    mismatches: list[dict[str, object]] = []
    for result in results[1:]:
        for compared_field in COMPARED_FIELDS:
            baseline_value = getattr(baseline, compared_field)
            result_value = getattr(result, compared_field)
            if result_value != baseline_value:
                mismatches.append(
                    {
                        "baseline": baseline.label,
                        "baseline_value": baseline_value,
                        "field": compared_field,
                        "label": result.label,
                        "value": result_value,
                    }
                )
    if mismatches:
        raise DeterminismMismatchError(json.dumps(mismatches, sort_keys=True))
    return {
        "baseline_label": baseline.label,
        "compared_fields": list(COMPARED_FIELDS),
        "label_count": len(results),
        "match": True,
    }


def compare_report_files(paths: Sequence[str | Path]) -> dict[str, object]:
    """Load final six-hour soak reports and compare every platform result."""

    if not paths:
        raise ReportError("at least one report path is required")
    all_results: list[SoakResult] = []
    loaded_reports: list[_LoadedReport] = []
    report_paths: list[str] = []
    for path in paths:
        loaded = _load_report(path)
        loaded_reports.append(loaded)
        report_paths.append(str(loaded.path))
        all_results.extend(loaded.results)
    _validate_final_compare_reports(tuple(loaded_reports), tuple(all_results))
    comparison = compare_results(tuple(all_results))
    return {
        "comparison": comparison,
        "labels": [result.label for result in all_results],
        "report_count": len(report_paths),
        "report_paths": report_paths,
        "result_count": len(all_results),
    }


def load_report_results(path: str | Path) -> tuple[SoakResult, ...]:
    """Load validated platform results from a saved soak JSON report."""

    return _load_report(path).results


def _load_report(path: str | Path) -> _LoadedReport:
    """Load one saved soak JSON report with its validated config."""

    report_path = Path(path)
    try:
        raw: object = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"report does not exist: {report_path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"report must be valid JSON: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise ReportError("report must be a JSON object")
    config_payload = raw.get("config")
    if not isinstance(config_payload, Mapping):
        raise ReportError("report config must be a JSON object")
    try:
        config = config_from_mapping(config_payload)
    except ConfigError as exc:
        raise ReportError(f"report config is invalid: {exc}") from exc
    comparison = raw.get("comparison")
    if not isinstance(comparison, Mapping) or comparison.get("match") is not True:
        raise ReportError("report comparison must be present and matched")
    results = raw.get("results")
    if not isinstance(results, list) or not results:
        raise ReportError("report results must be a non-empty list")
    loaded_results = tuple(_result_from_mapping(item) for item in results)
    configured_labels = tuple(platform.label for platform in config.platforms)
    result_labels = tuple(result.label for result in loaded_results)
    if result_labels != configured_labels:
        raise ReportError("report result labels must match config platform labels")
    for platform, result in zip(config.platforms, loaded_results, strict=True):
        if result.backend != platform.backend:
            raise ReportError("report result backend must match config platform backend")
    return _LoadedReport(path=report_path, config=config, results=loaded_results)


def _validate_final_compare_reports(
    reports: tuple[_LoadedReport, ...],
    results: tuple[SoakResult, ...],
) -> None:
    """Validate that report comparison proves the required final soak shape."""

    if not reports:
        raise ReportError("at least one report is required")
    expected_labels = set(FINAL_COMPARE_REQUIRED_PLATFORMS)
    actual_labels = {result.label for result in results}
    if actual_labels != expected_labels:
        missing = sorted(expected_labels - actual_labels)
        extra = sorted(actual_labels - expected_labels)
        raise ReportError(
            "final reports must contain exactly the required labels; "
            f"missing={missing} extra={extra}"
        )

    for report in reports:
        if report.config.max_steps is not None:
            raise ReportError("final reports must be duration-based, not max_steps-bounded")
        if report.config.duration_seconds < DEFAULT_DURATION_SECONDS:
            raise ReportError("final reports must run for at least six hours")
        expected_steps = _step_budget(report.config)
        if expected_steps % report.config.anchor_interval != 0:
            raise ReportError("final report step budget must end on an anchor boundary")
        for result in report.results:
            expected_backend = FINAL_COMPARE_REQUIRED_PLATFORMS[result.label]
            if result.backend != expected_backend or result.resolved_backend != expected_backend:
                raise ReportError(f"{result.label} must use resolved {expected_backend} backend")
            if result.steps != expected_steps:
                raise ReportError(f"{result.label} steps do not match report duration")
            if (
                result.elapsed_seconds + FINAL_COMPARE_ELAPSED_TOLERANCE_SECONDS
                < report.config.duration_seconds
            ):
                raise ReportError(f"{result.label} elapsed_seconds is below report duration")
            if result.fruit_count != result.steps:
                raise ReportError(f"{result.label} fruit_count must equal steps")
            if result.anchor_count != result.steps // report.config.anchor_interval:
                raise ReportError(f"{result.label} anchor_count must match fully anchored workload")


def load_config(path: str | Path) -> SoakConfig:
    """Load and validate a JSON soak config file."""

    config_path = Path(path)
    try:
        raw: object = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config must be valid JSON: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("config must be a JSON object")
    return config_from_mapping(raw)


def config_from_mapping(raw: Mapping[str, object]) -> SoakConfig:
    """Build a validated config from a decoded JSON object."""

    unknown = set(raw) - CONFIG_KEYS
    if unknown:
        raise ConfigError(f"unknown config key: {sorted(unknown)[0]}")
    if "duration_seconds" in raw and "duration_hours" in raw:
        raise ConfigError("duration_seconds and duration_hours are mutually exclusive")

    duration_seconds = DEFAULT_DURATION_SECONDS
    if "duration_seconds" in raw:
        duration_seconds = _nonnegative_float("duration_seconds", raw["duration_seconds"])
    elif "duration_hours" in raw:
        duration_hours = _nonnegative_float("duration_hours", raw["duration_hours"])
        duration_seconds = duration_hours * 60 * 60

    max_steps = _optional_positive_int("max_steps", raw.get("max_steps"))
    platforms = (
        _parse_platforms(raw["platforms"])
        if "platforms" in raw
        else (PlatformRun("local-auto", "auto"),)
    )
    seed = _optional_str("seed", raw.get("seed"), "tensorpow-soak")
    anchor_interval = _optional_positive_int(
        "anchor_interval",
        raw.get("anchor_interval", DEFAULT_ANCHOR_INTERVAL),
    )
    if anchor_interval is None:
        raise ConfigError("anchor_interval must be positive")
    data_dir = _optional_path("data_dir", raw.get("data_dir"))
    report_path = _optional_path("report_path", raw.get("report_path"))
    return SoakConfig(
        duration_seconds=duration_seconds,
        max_steps=max_steps,
        platforms=platforms,
        seed=seed,
        anchor_interval=anchor_interval,
        data_dir=data_dir,
        report_path=report_path,
    )


def config_from_args(args: argparse.Namespace) -> SoakConfig:
    """Build a config from CLI arguments, applying explicit overrides."""

    config_path = cast(Path | None, args.config)
    config = load_config(config_path) if config_path is not None else SoakConfig()

    platforms_arg = cast(list[str] | None, args.platform)
    platforms = config.platforms
    if platforms_arg is not None:
        platforms = tuple(_parse_platform_spec(spec) for spec in platforms_arg)

    duration_seconds = config.duration_seconds
    duration_seconds_arg = cast(float | None, args.duration_seconds)
    duration_hours_arg = cast(float | None, args.duration_hours)
    if duration_seconds_arg is not None and duration_hours_arg is not None:
        raise ConfigError("--duration-seconds and --duration-hours are mutually exclusive")
    if duration_seconds_arg is not None:
        duration_seconds = _nonnegative_float("duration_seconds", duration_seconds_arg)
    elif duration_hours_arg is not None:
        duration_seconds = _nonnegative_float("duration_hours", duration_hours_arg) * 60 * 60

    max_steps = config.max_steps
    max_steps_arg = cast(int | None, args.max_steps)
    if max_steps_arg is not None:
        max_steps = max_steps_arg

    anchor_interval = config.anchor_interval
    anchor_interval_arg = cast(int | None, args.anchor_interval)
    if anchor_interval_arg is not None:
        anchor_interval = anchor_interval_arg

    seed = config.seed
    seed_arg = cast(str | None, args.seed)
    if seed_arg is not None:
        seed = seed_arg

    data_dir = config.data_dir
    data_dir_arg = cast(Path | None, args.data_dir)
    if data_dir_arg is not None:
        data_dir = data_dir_arg

    report_path = config.report_path
    report_arg = cast(Path | None, args.report)
    if report_arg is not None:
        report_path = report_arg

    if cast(bool, args.test_mode):
        if max_steps is None:
            max_steps = TEST_MODE_STEPS
        if duration_seconds == DEFAULT_DURATION_SECONDS:
            duration_seconds = 0

    return SoakConfig(
        duration_seconds=duration_seconds,
        max_steps=max_steps,
        platforms=platforms,
        seed=seed,
        anchor_interval=anchor_interval,
        data_dir=data_dir,
        report_path=report_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TensorPoW deterministic soak workloads")
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument(
        "--compare-report",
        action="append",
        type=Path,
        help="compare one or more saved soak JSON reports instead of running a workload",
    )
    parser.add_argument(
        "--platform",
        action="append",
        help="platform label with optional backend, e.g. linux-x86-cuda:cuda",
    )
    parser.add_argument("--duration-hours", type=float, help="soak duration in hours")
    parser.add_argument("--duration-seconds", type=float, help="soak duration in seconds")
    parser.add_argument("--max-steps", type=int, help="stop after this many workload steps")
    parser.add_argument("--anchor-interval", type=int, help="create an anchor every N steps")
    parser.add_argument("--seed", help="deterministic workload seed")
    parser.add_argument("--data-dir", type=Path, help="base directory for temporary node stores")
    parser.add_argument("--report", type=Path, help="write the JSON report to this path")
    parser.add_argument("--test-mode", action="store_true", help="short bounded CI workload")
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        compare_paths = cast(list[Path] | None, args.compare_report)
        if compare_paths is not None:
            payload = compare_report_files(compare_paths)
            config = None
        else:
            config = config_from_args(args)
            payload = run_soak(config).to_json_obj()
    except ConfigError as exc:
        parser.error(str(exc))
    except ReportError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    except DeterminismMismatchError as exc:
        print(f"determinism mismatch: {exc}", file=sys.stderr)
        return 1

    rendered = (
        json.dumps(payload, sort_keys=True)
        if args.json
        else json.dumps(payload, indent=2, sort_keys=True)
    )
    if config is not None and config.report_path is not None:
        config.report_path.parent.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _run_platform(config: SoakConfig, platform: PlatformRun) -> SoakResult:
    resolved_backend = resolve_backend(platform.backend)
    start = monotonic()
    transcript = _Transcript()
    data_parent = config.data_dir
    if data_parent is not None:
        data_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f"tensorpow-soak-{_safe_label(platform.label)}-",
        dir=None if data_parent is None else str(data_parent),
    ) as temp_dir:
        genesis = _genesis_anchor(config.seed)
        node = TensorPowNode(
            TensorPowConfig(
                data_dir=Path(temp_dir) / "node",
                expected_genesis_hash=genesis.block_hash(),
            ),
            pow_verifier=_soak_pow_verifier,
        )
        try:
            result = _run_node_workload(
                config=config,
                genesis=genesis,
                node=node,
                platform=platform,
                resolved_backend=resolved_backend,
                transcript=transcript,
                start=start,
            )
        finally:
            node.close()
    return result


def _run_node_workload(
    *,
    config: SoakConfig,
    genesis: Anchor,
    node: TensorPowNode,
    platform: PlatformRun,
    resolved_backend: Literal["cpu", "cuda", "mps"],
    transcript: _Transcript,
    start: float,
) -> SoakResult:
    wallets = tuple(_wallet(config.seed, index) for index in range(WALLET_COUNT))
    _require_accepted("genesis", node.process_anchor(genesis))
    last_fruit_hash = GENESIS_PARENT_HASH
    last_anchor_hash = genesis.block_hash()
    last_pow_digest = _record_anchor(transcript, genesis, platform.backend)
    pending_fruit_hashes: list[bytes] = []
    steps = 0
    tx_count = 0
    fruit_count = 0
    anchor_count = 0
    step_budget = _step_budget(config)

    while steps < step_budget:
        coinbase = _coinbase_transaction(wallets[steps % WALLET_COUNT], steps)
        spend = _spend_transaction(node, wallets, steps)
        transactions = (coinbase,) if spend is None else (coinbase, spend)
        fruit = _fruit(
            transactions=transactions,
            parent_selected=last_fruit_hash,
            latest_anchor=last_anchor_hash,
            step=steps,
            anchor_interval=config.anchor_interval,
        )
        _require_accepted("fruit", node.process_fruit(fruit))
        last_fruit_hash = fruit.block_hash()
        pending_fruit_hashes.append(last_fruit_hash)
        fruit_count += 1
        tx_count += len(transactions)
        last_pow_digest = _record_fruit(transcript, fruit, platform.backend)
        transcript.record("utxo-root", node.utxo_set.merkle_root())

        if (steps + 1) % config.anchor_interval == 0:
            anchor = _anchor(
                covered_fruit_hashes=tuple(sorted(pending_fruit_hashes)),
                parent_candidate_hashes=(last_fruit_hash,),
                parent_anchor=last_anchor_hash,
                step=steps,
                anchor_interval=config.anchor_interval,
                fee_floor_matoms_per_kb=1 + steps,
            )
            _require_accepted("anchor", node.process_anchor(anchor))
            last_anchor_hash = anchor.block_hash()
            pending_fruit_hashes.clear()
            anchor_count += 1
            last_pow_digest = _record_anchor(transcript, anchor, platform.backend)

        steps += 1
        _pace_duration_run(start, config.duration_seconds, steps, step_budget)

    utxos = node.utxo_set.utxos()
    final_state_root = node.utxo_set.merkle_root()
    utxo_outputs_digest = _utxo_outputs_digest(utxos)
    transcript.record("final-utxo-root", final_state_root)
    for utxo in utxos:
        transcript.record("final-utxo", utxo.to_bytes())

    return SoakResult(
        label=platform.label,
        backend=platform.backend,
        resolved_backend=resolved_backend,
        platform=platform_module.platform(),
        python_version=platform_module.python_version(),
        steps=steps,
        elapsed_seconds=monotonic() - start,
        final_state_root=final_state_root.hex(),
        utxo_outputs_digest=utxo_outputs_digest,
        consensus_bytes_digest=transcript.hexdigest(),
        last_pow_digest=last_pow_digest.hex(),
        utxo_count=len(utxos),
        tx_count=tx_count,
        fruit_count=fruit_count,
        anchor_count=anchor_count,
        consensus_object_count=transcript.count,
    )


def _step_budget(config: SoakConfig) -> int:
    if config.max_steps is not None:
        return config.max_steps
    if config.duration_seconds <= 0:
        raise ConfigError("duration_seconds must be positive when max_steps is omitted")
    return max(1, math.ceil(config.duration_seconds / DURATION_STEP_CADENCE_SECONDS))


def _pace_duration_run(
    start: float,
    duration_seconds: float,
    completed_steps: int,
    total_steps: int,
) -> None:
    if duration_seconds <= 0:
        return
    target_elapsed = duration_seconds * completed_steps / total_steps
    remaining = start + target_elapsed - monotonic()
    if remaining > 0:
        sleep(remaining)


def _wallet(seed: str, index: int) -> Wallet:
    seed_bytes = hash_bytes(f"{seed}/wallet/{index}".encode())
    return Wallet.from_seed(seed_bytes)


def _genesis_anchor(seed: str) -> Anchor:
    return build_genesis_artifact(
        GenesisInputs.create(
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            whitepaper_hash=hash_bytes(f"{seed}/genesis/whitepaper".encode()),
            bitcoin_block_hash=hash_bytes(f"{seed}/genesis/bitcoin".encode()),
            ethereum_block_hash=hash_bytes(f"{seed}/genesis/ethereum".encode()),
            founder_pubkey_hash=hash_bytes(f"{seed}/genesis/founder".encode()),
        )
    ).anchor


def _coinbase_transaction(wallet: Wallet, step: int) -> Transaction:
    amount = 100_000 + step
    return Transaction.coinbase(
        (
            Output(
                amount_matoms=amount,
                template_id=TEMPLATE_PKH,
                payload=wallet.owner_pubkey_hash,
            ),
        )
    )


def _spend_transaction(
    node: TensorPowNode,
    wallets: tuple[Wallet, ...],
    step: int,
) -> Transaction | None:
    if step == 0:
        return None
    source = wallets[(step - 1) % len(wallets)]
    recipient = wallets[(step + 1) % len(wallets)]
    current_height = node._anchor_height()  # deterministic soak harness visibility
    spendable = tuple(
        utxo
        for utxo in source.owned_utxos(node.utxo_set.utxos())
        if utxo.lockheight <= current_height
    )
    if not spendable:
        return None
    fee = 1 + step
    spend_cap = spendable[0].amount_matoms - fee
    if spend_cap <= 0:
        return None
    amount = min(spend_cap, 100 + step * 3)
    return source.create_signed_transaction(
        utxos=spendable,
        recipient_address=recipient.address,
        amount_matoms=amount,
        fee_matoms=fee,
    )


def _fruit(
    *,
    transactions: tuple[Transaction, ...],
    parent_selected: bytes,
    latest_anchor: bytes,
    step: int,
    anchor_interval: int,
) -> Fruit:
    tx_bytes = tuple(tx.to_bytes() for tx in transactions)
    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=parent_selected,
        parent_bitmap=b"",
        latest_anchor=latest_anchor,
        tx_merkle_root=tx_merkle_root(tx_bytes),
        timestamp_ms=_fruit_timestamp_ms(step, anchor_interval),
        shard_id=ROOT_SHARD_ID,
        nonce=step,
    )
    return Fruit(header=header, transactions=tx_bytes)


def _anchor(
    *,
    covered_fruit_hashes: tuple[bytes, ...],
    parent_candidate_hashes: tuple[bytes, ...],
    parent_anchor: bytes,
    step: int,
    anchor_interval: int,
    fee_floor_matoms_per_kb: int,
) -> Anchor:
    tree = ShardTree()
    fee_entries = (FeeFloorEntry(ROOT_SHARD_ID, fee_floor_matoms_per_kb),)
    header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=parent_anchor,
        fruit_set_root=fruit_set_root(covered_fruit_hashes),
        parent_candidate_root=parent_candidate_root(parent_candidate_hashes),
        shard_tree_state_root=tree.state_root(),
        fee_floor_set_root=fee_floor_set_root(fee_entries),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=_anchor_timestamp_ms(step, anchor_interval),
        nonce=step,
    )
    return Anchor(
        header=header,
        covered_fruit_hashes=covered_fruit_hashes,
        parent_candidate_hashes=parent_candidate_hashes,
        shard_tree_bytes=tree.serialize(),
        fee_floor_entries=fee_entries,
    )


def _fruit_timestamp_ms(step: int, anchor_interval: int) -> int:
    _require_nonnegative_int("step", step)
    _require_positive_int("anchor_interval", anchor_interval)
    return BASE_TIMESTAMP_MS + step * ANCHOR_INTERVAL_MS // anchor_interval


def _anchor_timestamp_ms(step: int, anchor_interval: int) -> int:
    _require_nonnegative_int("step", step)
    _require_positive_int("anchor_interval", anchor_interval)
    return BASE_TIMESTAMP_MS + (step + 1) * ANCHOR_INTERVAL_MS // anchor_interval


def _soak_pow_verifier(_header: object, target: bytes, _backend: object) -> bool:
    return isinstance(target, bytes) and len(target) == HASH_LEN_BYTES


def _record_fruit(transcript: _Transcript, fruit: Fruit, backend: Backend) -> bytes:
    for tx in fruit.transactions:
        transcript.record("tx", tx)
    transcript.record("fruit-header", fruit.header.serialize())
    transcript.record("fruit-body", fruit.serialize())
    digest = pow_digest_for_header(fruit.header.to_pow_header(()), backend=backend)
    transcript.record("fruit-pow-digest", digest)
    transcript.record("fruit-hash", fruit.block_hash())
    return digest


def _record_anchor(transcript: _Transcript, anchor: Anchor, backend: Backend) -> bytes:
    transcript.record("anchor-header", anchor.header.serialize())
    transcript.record("anchor-body", anchor.serialize())
    digest = pow_digest_for_header(anchor.header.to_pow_header(), backend=backend)
    transcript.record("anchor-pow-digest", digest)
    transcript.record("anchor-hash", anchor.block_hash())
    return digest


def _utxo_outputs_digest(utxos: tuple[UTXO, ...]) -> str:
    hasher = blake3()
    for utxo in utxos:
        raw = utxo.to_bytes()
        hasher.update(len(raw).to_bytes(8, "little"))
        hasher.update(raw)
    return hasher.hexdigest()


def _require_accepted(kind: str, result: NodeResult) -> None:
    if not result.accepted:
        raise RuntimeError(f"{kind} rejected: {result.reason or 'unknown'}")


def _parse_platforms(value: object) -> tuple[PlatformRun, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("platforms must contain at least one platform")
    return tuple(_parse_platform_item(item) for item in value)


def _parse_platform_item(item: object) -> PlatformRun:
    if isinstance(item, str):
        return _parse_platform_spec(item)
    if not isinstance(item, Mapping):
        raise ConfigError("each platform must be a string or object")
    unknown = set(item) - {"backend", "label"}
    if unknown:
        raise ConfigError(f"unknown platform key: {sorted(unknown)[0]}")
    label = item.get("label")
    if not isinstance(label, str):
        raise ConfigError("platform label must be a string")
    backend = _parse_backend(item.get("backend", "auto"))
    return PlatformRun(label=label, backend=backend)


def _result_from_mapping(item: object) -> SoakResult:
    if not isinstance(item, Mapping):
        raise ReportError("each report result must be an object")
    return SoakResult(
        label=_expect_nonempty_str(item, "label"),
        backend=_parse_backend(item.get("backend")),
        resolved_backend=_parse_resolved_backend(item.get("resolved_backend")),
        platform=_expect_nonempty_str(item, "platform"),
        python_version=_expect_nonempty_str(item, "python_version"),
        steps=_expect_nonnegative_int(item, "steps"),
        elapsed_seconds=_nonnegative_float("elapsed_seconds", item.get("elapsed_seconds")),
        final_state_root=_expect_hash_hex(item, "final_state_root"),
        utxo_outputs_digest=_expect_hash_hex(item, "utxo_outputs_digest"),
        consensus_bytes_digest=_expect_hash_hex(item, "consensus_bytes_digest"),
        last_pow_digest=_expect_hash_hex(item, "last_pow_digest"),
        utxo_count=_expect_nonnegative_int(item, "utxo_count"),
        tx_count=_expect_nonnegative_int(item, "tx_count"),
        fruit_count=_expect_nonnegative_int(item, "fruit_count"),
        anchor_count=_expect_nonnegative_int(item, "anchor_count"),
        consensus_object_count=_expect_nonnegative_int(item, "consensus_object_count"),
    )


def _parse_platform_spec(spec: str) -> PlatformRun:
    _require_nonempty_str("platform", spec)
    delimiter = ":" if ":" in spec else "=" if "=" in spec else ""
    if delimiter:
        label, backend_raw = spec.rsplit(delimiter, 1)
        backend = _parse_backend(backend_raw)
    else:
        label = spec
        backend = "auto"
    return PlatformRun(label=label, backend=backend)


def _parse_backend(value: object) -> Backend:
    if not isinstance(value, str) or value == "":
        raise ConfigError("backend must be a non-empty string")
    if value not in VALID_BACKENDS:
        raise ConfigError(f"backend must be one of {sorted(VALID_BACKENDS)}")
    return cast(Backend, value)


def _parse_resolved_backend(value: object) -> Literal["cpu", "cuda", "mps"]:
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        return "cuda"
    if value == "mps":
        return "mps"
    raise ReportError("resolved_backend must be cpu, cuda, or mps")


def _expect_nonempty_str(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or value == "":
        raise ReportError(f"{key} must be a non-empty string")
    return value


def _expect_hash_hex(mapping: Mapping[str, object], key: str) -> str:
    value = _expect_nonempty_str(mapping, key)
    if len(value) != HASH_LEN_BYTES * 2:
        raise ReportError(f"{key} must be 32-byte hex")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ReportError(f"{key} must be hex") from exc
    if decoded.hex() != value:
        raise ReportError(f"{key} must be canonical lowercase hex")
    return value


def _expect_nonnegative_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReportError(f"{key} must be an integer")
    if value < 0:
        raise ReportError(f"{key} must be nonnegative")
    return value


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-") or "platform"


def _optional_positive_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    _require_positive_int(name, value)
    return cast(int, value)


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


def _require_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    if value < 0:
        raise ConfigError(f"{name} must be nonnegative")


def _nonnegative_float(name: str, value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ConfigError(f"{name} must be finite and non-negative")
    return numeric


def _optional_str(name: str, value: object, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string")
    _require_nonempty_str(name, value)
    return value


def _optional_path(name: str, value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"{name} must be a non-empty string")
    return Path(value)


def _require_nonempty_str(name: str, value: object) -> None:
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"{name} must be a non-empty string")


def _require_label(value: str) -> None:
    _require_nonempty_str("platform label", value)
    if value.strip() != value:
        raise ConfigError("platform label must not have leading or trailing whitespace")


if __name__ == "__main__":
    raise SystemExit(main())
