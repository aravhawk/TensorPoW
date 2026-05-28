"""Integration tests for the deterministic soak runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import scripts.soak_test as soak_test
from scripts.soak_test import (
    ConfigError,
    DeterminismMismatchError,
    PlatformRun,
    ReportError,
    SoakConfig,
    build_arg_parser,
    compare_report_files,
    config_from_args,
    load_config,
    load_report_results,
    run_soak,
)


def test_soak_test_mode_compares_two_cpu_labels(tmp_path: Path) -> None:
    config = SoakConfig(
        duration_seconds=0,
        max_steps=2,
        platforms=(
            PlatformRun("macos-arm-cpu", "cpu"),
            PlatformRun("linux-x86-cpu", "cpu"),
        ),
        seed="integration-soak",
        anchor_interval=2,
        data_dir=tmp_path,
    )

    report = run_soak(config)

    assert report.comparison["match"] is True
    assert report.comparison["compared_fields"] == [
        "final_state_root",
        "utxo_outputs_digest",
        "consensus_bytes_digest",
        "utxo_count",
        "tx_count",
        "fruit_count",
        "anchor_count",
    ]
    assert len(report.results) == 2
    first, second = report.results
    assert first.final_state_root == second.final_state_root
    assert first.utxo_outputs_digest == second.utxo_outputs_digest
    assert first.consensus_bytes_digest == second.consensus_bytes_digest
    assert first.steps == second.steps == 2
    assert first.tx_count == second.tx_count == 2
    assert first.fruit_count == second.fruit_count == 2
    assert first.anchor_count == second.anchor_count == 1
    assert len(first.final_state_root) == 64


def test_duration_only_soak_uses_fixed_step_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(soak_test, "DURATION_STEP_CADENCE_SECONDS", 0.01)
    config = SoakConfig(
        duration_seconds=0.025,
        platforms=(PlatformRun("duration-a", "cpu"), PlatformRun("duration-b", "cpu")),
        data_dir=tmp_path,
    )

    report = run_soak(config)

    assert report.comparison["match"] is True
    assert [result.steps for result in report.results] == [3, 3]
    assert [result.tx_count for result in report.results] == [3, 3]
    assert [result.anchor_count for result in report.results] == [1, 1]


def test_soak_timestamps_keep_reward_claims_inside_daa_budget(tmp_path: Path) -> None:
    config = SoakConfig(
        duration_seconds=0,
        max_steps=45,
        platforms=(PlatformRun("reward-budget", "cpu"),),
        data_dir=tmp_path,
        anchor_interval=3,
    )

    report = run_soak(config)

    result = report.results[0]
    assert result.steps == 45
    assert result.anchor_count == 15
    assert result.utxo_count > 0


def test_cli_test_mode_configures_short_bounded_run(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--test-mode",
            "--platform",
            "ci-a:cpu",
            "--platform",
            "ci-b:cpu",
            "--data-dir",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    config = config_from_args(args)

    assert config.duration_seconds == 0
    assert config.max_steps == 3
    assert config.platforms == (PlatformRun("ci-a", "cpu"), PlatformRun("ci-b", "cpu"))
    assert config.report_path == tmp_path / "report.json"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not-json", "valid JSON"),
        ([], "JSON object"),
        ({"unknown": True}, "unknown config key"),
        ({"duration_seconds": -1}, "duration_seconds"),
        ({"duration_seconds": 0}, "duration_seconds must be positive"),
        ({"duration_seconds": 0, "max_steps": 0}, "max_steps"),
        ({"duration_seconds": 0, "max_steps": 1, "platforms": []}, "at least one"),
        (
            {
                "duration_seconds": 0,
                "max_steps": 1,
                "platforms": [
                    {"label": "same", "backend": "cpu"},
                    {"label": "same", "backend": "cpu"},
                ],
            },
            "unique",
        ),
        (
            {
                "duration_seconds": 0,
                "max_steps": 1,
                "platforms": [{"label": "bad", "backend": "gpu"}],
            },
            "backend",
        ),
        (
            {
                "duration_hours": 1,
                "duration_seconds": 1,
                "max_steps": 1,
            },
            "mutually exclusive",
        ),
        (
            {
                "duration_seconds": 0,
                "max_steps": 1,
                "platforms": [{"label": " bad ", "backend": "cpu"}],
            },
            "whitespace",
        ),
    ],
)
def test_malformed_configs_are_rejected(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    path = tmp_path / "soak.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_valid_config_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "soak.json"
    path.write_text(
        json.dumps(
            {
                "anchor_interval": 2,
                "data_dir": str(tmp_path / "data"),
                "duration_seconds": 0,
                "max_steps": 1,
                "platforms": ["local-cpu:cpu"],
                "report_path": str(tmp_path / "report.json"),
                "seed": "config-seed",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.duration_seconds == 0
    assert config.max_steps == 1
    assert config.platforms == (PlatformRun("local-cpu", "cpu"),)
    assert config.seed == "config-seed"
    assert config.anchor_interval == 2
    assert config.data_dir == tmp_path / "data"
    assert config.report_path == tmp_path / "report.json"


def test_saved_soak_reports_compare_required_final_platforms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(soak_test, "DEFAULT_DURATION_SECONDS", 12.0)
    first = run_soak(
        SoakConfig(
            duration_seconds=12.0,
            platforms=(
                PlatformRun("mac-arm-cpu", "cpu"),
                PlatformRun("mac-arm-mps", "cpu"),
            ),
            data_dir=tmp_path / "first",
            anchor_interval=1,
        )
    )
    second = run_soak(
        SoakConfig(
            duration_seconds=12.0,
            platforms=(
                PlatformRun("ubuntu-x86-cpu", "cpu"),
                PlatformRun("ubuntu-x86-cuda", "cpu"),
            ),
            data_dir=tmp_path / "second",
            anchor_interval=1,
        )
    )
    first_payload = first.to_json_obj()
    second_payload = second.to_json_obj()
    _rewrite_report_backend(first_payload, "mac-arm-mps", "mps")
    _rewrite_report_backend(second_payload, "ubuntu-x86-cuda", "cuda")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first_payload), encoding="utf-8")
    second_path.write_text(json.dumps(second_payload), encoding="utf-8")

    comparison = compare_report_files((first_path, second_path))

    assert comparison["result_count"] == 4
    assert comparison["labels"] == [
        "mac-arm-cpu",
        "mac-arm-mps",
        "ubuntu-x86-cpu",
        "ubuntu-x86-cuda",
    ]
    assert comparison["comparison"] == {
        "baseline_label": "mac-arm-cpu",
        "compared_fields": [
            "final_state_root",
            "utxo_outputs_digest",
            "consensus_bytes_digest",
            "utxo_count",
            "tx_count",
            "fruit_count",
            "anchor_count",
        ],
        "label_count": 4,
        "match": True,
    }
    assert load_report_results(first_path)[0].label == "mac-arm-cpu"


def test_saved_soak_report_comparison_rejects_mismatch_and_malformed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(soak_test, "DEFAULT_DURATION_SECONDS", 12.0)
    report = run_soak(
        SoakConfig(
            duration_seconds=12.0,
            platforms=(
                PlatformRun("mac-arm-cpu", "cpu"),
                PlatformRun("mac-arm-mps", "cpu"),
                PlatformRun("ubuntu-x86-cpu", "cpu"),
                PlatformRun("ubuntu-x86-cuda", "cpu"),
            ),
            data_dir=tmp_path / "baseline",
            anchor_interval=1,
        )
    )
    baseline_path = tmp_path / "baseline.json"
    mismatch_path = tmp_path / "mismatch.json"
    malformed_path = tmp_path / "malformed.json"
    payload = report.to_json_obj()
    _rewrite_report_backend(payload, "mac-arm-mps", "mps")
    _rewrite_report_backend(payload, "ubuntu-x86-cuda", "cuda")
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    mismatched = json.loads(json.dumps(payload))
    mismatched["results"][0]["final_state_root"] = "aa" * 32
    mismatch_path.write_text(json.dumps(mismatched), encoding="utf-8")
    malformed = json.loads(json.dumps(payload))
    malformed["comparison"] = {"match": False}
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(DeterminismMismatchError):
        compare_report_files((mismatch_path,))
    with pytest.raises(ReportError, match="matched"):
        load_report_results(malformed_path)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload["results"].pop(), "result labels"),
        (
            lambda payload: payload["config"].__setitem__("max_steps", 1),
            "duration-based",
        ),
        (
            lambda payload: payload["config"].__setitem__("duration_seconds", 11.0),
            "six hours",
        ),
        (
            lambda payload: payload["results"][1].__setitem__("resolved_backend", "cpu"),
            "mac-arm-mps",
        ),
        (
            lambda payload: payload["results"][0].__setitem__("elapsed_seconds", 9.0),
            "elapsed_seconds",
        ),
        (
            lambda payload: payload["results"][0].__setitem__("anchor_count", 0),
            "anchor_count",
        ),
        (
            lambda payload: payload["results"][0].__setitem__(
                "final_state_root",
                ("aa" * 31) + "  ",
            ),
            "canonical",
        ),
    ],
)
def test_final_soak_report_comparison_rejects_incomplete_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutator: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    monkeypatch.setattr(soak_test, "DEFAULT_DURATION_SECONDS", 12.0)
    report = run_soak(
        SoakConfig(
            duration_seconds=12.0,
            platforms=(
                PlatformRun("mac-arm-cpu", "cpu"),
                PlatformRun("mac-arm-mps", "cpu"),
                PlatformRun("ubuntu-x86-cpu", "cpu"),
                PlatformRun("ubuntu-x86-cuda", "cpu"),
            ),
            data_dir=tmp_path,
            anchor_interval=1,
        )
    )
    payload = report.to_json_obj()
    _rewrite_report_backend(payload, "mac-arm-mps", "mps")
    _rewrite_report_backend(payload, "ubuntu-x86-cuda", "cuda")
    mutator(payload)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReportError, match=match):
        compare_report_files((report_path,))


def _rewrite_report_backend(payload: dict[str, object], label: str, backend: str) -> None:
    config = payload["config"]
    assert isinstance(config, dict)
    platforms = config["platforms"]
    assert isinstance(platforms, list)
    for platform in platforms:
        if isinstance(platform, dict) and platform.get("label") == label:
            platform["backend"] = backend
    results = payload["results"]
    assert isinstance(results, list)
    for result in results:
        if isinstance(result, dict) and result.get("label") == label:
            result["backend"] = backend
            result["resolved_backend"] = backend
