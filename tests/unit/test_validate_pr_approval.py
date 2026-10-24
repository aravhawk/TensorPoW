"""Unit tests for the maintainer-approval gate in front of approved PR CI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_pr_approval.py"
HEAD_SHA = "a" * 40
OLD_SHA = "b" * 40
FORK = "contributor/TensorPoW"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_pr_approval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_script()

PERMISSIONS = {"aravhawk": "admin", "contributor": "read", "helper": "write"}


def _review(user: str, state: str, commit: str, submitted_at: str) -> dict[str, Any]:
    return {
        "user": {"login": user},
        "state": state,
        "commit_id": commit,
        "submitted_at": submitted_at,
    }


def _run_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reviews: list[dict[str, Any]],
    artifact: dict[str, Any] | None = None,
    pr_state: str = "open",
) -> dict[str, str]:
    artifact_dir = tmp_path / "artifacts" / "approved-pr-7"
    artifact_dir.mkdir(parents=True)
    payload = {
        "pr_number": 7,
        "reviewer": "aravhawk",
        "state": "approved",
        "head_sha": HEAD_SHA,
        "head_repo": FORK,
        "head_ref": "feature",
    }
    payload.update(artifact or {})
    (artifact_dir / "approval.json").write_text(json.dumps(payload), encoding="utf-8")

    responses: dict[str, Any] = {
        "pulls/7": {
            "state": pr_state,
            "head": {"sha": HEAD_SHA, "repo": {"full_name": FORK}, "ref": "feature"},
        },
        "pulls/7/reviews?per_page=100": reviews,
    }
    for user, permission in PERMISSIONS.items():
        responses[f"collaborators/{user}/permission"] = {"permission": permission}
    monkeypatch.setattr(gate, "api_json", lambda path: responses[path])

    output_file = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("APPROVAL_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    gate.main()

    outputs: dict[str, str] = {}
    for line in output_file.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        outputs[key] = value
    return outputs


def test_maintainer_approval_of_current_fork_head_runs_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        reviews=[_review("aravhawk", "APPROVED", HEAD_SHA, "2026-10-24T10:00:00Z")],
    )

    assert outputs["should_run"] == "true"
    assert outputs["head_sha"] == HEAD_SHA
    assert outputs["head_repo"] == FORK


def test_forged_artifact_without_real_approval_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        reviews=[_review("contributor", "COMMENTED", HEAD_SHA, "2026-10-24T10:00:00Z")],
    )

    assert outputs["should_run"] == "false"


def test_approval_from_user_without_write_access_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        reviews=[_review("contributor", "APPROVED", HEAD_SHA, "2026-10-24T10:00:00Z")],
    )

    assert outputs["should_run"] == "false"


def test_artifact_naming_a_read_only_reviewer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit):
        _run_gate(
            tmp_path,
            monkeypatch,
            artifact={"reviewer": "contributor"},
            reviews=[_review("aravhawk", "APPROVED", HEAD_SHA, "2026-10-24T10:00:00Z")],
        )


def test_new_commits_after_approval_need_a_new_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        artifact={"head_sha": OLD_SHA},
        reviews=[_review("aravhawk", "APPROVED", OLD_SHA, "2026-10-24T10:00:00Z")],
    )

    assert outputs["should_run"] == "false"


def test_approval_of_an_older_commit_does_not_cover_the_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        reviews=[_review("aravhawk", "APPROVED", OLD_SHA, "2026-10-24T10:00:00Z")],
    )

    assert outputs["should_run"] == "false"


def test_changes_requested_after_approval_blocks_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        reviews=[
            _review("aravhawk", "APPROVED", HEAD_SHA, "2026-10-24T10:00:00Z"),
            _review("helper", "CHANGES_REQUESTED", HEAD_SHA, "2026-10-24T11:00:00Z"),
        ],
    )

    assert outputs["should_run"] == "false"


def test_closed_pr_does_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        pr_state="closed",
        reviews=[_review("aravhawk", "APPROVED", HEAD_SHA, "2026-10-24T10:00:00Z")],
    )

    assert outputs["should_run"] == "false"


def test_artifact_head_repo_must_match_the_pr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit):
        _run_gate(
            tmp_path,
            monkeypatch,
            artifact={"head_repo": "attacker/TensorPoW"},
            reviews=[_review("aravhawk", "APPROVED", HEAD_SHA, "2026-10-24T10:00:00Z")],
        )
