"""Unit tests for the trust and maintainer-approval gates in front of PR CI."""

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
BASE = "aravhawk/TensorPoW"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_pr_approval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_script()

PERMISSIONS = {
    "aravhawk": "admin",
    "contributor": "read",
    "helper": "write",
    "dependabot[bot]": "none",
}


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
    pr_author: str = "contributor",
    head_repo: str = FORK,
) -> dict[str, str]:
    artifact_dir = tmp_path / "artifacts" / "approved-pr-7"
    artifact_dir.mkdir(parents=True)
    payload = {
        "pr_number": 7,
        "reviewer": "aravhawk",
        "state": "approved",
        "head_sha": HEAD_SHA,
        "head_repo": head_repo,
        "head_ref": "feature",
    }
    payload.update(artifact or {})
    (artifact_dir / "approval.json").write_text(json.dumps(payload), encoding="utf-8")

    responses: dict[str, Any] = {
        "pulls/7": {
            "state": pr_state,
            "user": {"login": pr_author},
            "head": {"sha": HEAD_SHA, "repo": {"full_name": head_repo}, "ref": "feature"},
        },
        "pulls/7/reviews?per_page=100": reviews,
    }
    _mock_permissions(monkeypatch, responses)
    monkeypatch.setenv("APPROVAL_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    return _main(tmp_path, monkeypatch, [])


def _mock_permissions(monkeypatch: pytest.MonkeyPatch, responses: dict[str, Any]) -> None:
    for user, permission in PERMISSIONS.items():
        responses[f"collaborators/{user}/permission"] = {"permission": permission}
    monkeypatch.setattr(gate, "api_json", lambda path: responses[path])


def _main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]) -> dict[str, str]:
    output_file = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_REPOSITORY", BASE)
    gate.main(args)

    outputs: dict[str, str] = {}
    for line in output_file.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        outputs[key] = value
    return outputs


def _run_trusted_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, author: str, head_repo: str
) -> dict[str, str]:
    _mock_permissions(monkeypatch, {})
    monkeypatch.setenv("PR_NUMBER", "7")
    monkeypatch.setenv("PR_AUTHOR", author)
    monkeypatch.setenv("HEAD_REPO", head_repo)
    return _main(tmp_path, monkeypatch, ["--trusted-pr"])


@pytest.mark.parametrize("author", ["aravhawk", "helper"])
def test_same_repo_pr_by_write_access_author_runs_without_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, author: str
) -> None:
    outputs = _run_trusted_gate(tmp_path, monkeypatch, author=author, head_repo=BASE)

    assert outputs["should_run"] == "true"


@pytest.mark.parametrize("author", ["aravhawk", "contributor"])
def test_fork_pr_is_never_trusted_even_from_a_maintainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, author: str
) -> None:
    head_repo = f"{author}-fork/TensorPoW"
    outputs = _run_trusted_gate(tmp_path, monkeypatch, author=author, head_repo=head_repo)

    assert outputs["should_run"] == "false"


@pytest.mark.parametrize("author", ["contributor", "dependabot[bot]"])
def test_same_repo_pr_by_author_without_write_access_needs_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, author: str
) -> None:
    outputs = _run_trusted_gate(tmp_path, monkeypatch, author=author, head_repo=BASE)

    assert outputs["should_run"] == "false"


def test_unexpected_arguments_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _main(tmp_path, monkeypatch, ["--trusted"])


def test_approval_of_a_trusted_pr_does_not_run_ci_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        pr_author="helper",
        head_repo=BASE,
        reviews=[_review("aravhawk", "APPROVED", HEAD_SHA, "2026-10-24T10:00:00Z")],
    )

    assert outputs["should_run"] == "false"


def test_approval_of_a_same_repo_bot_pr_runs_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = _run_gate(
        tmp_path,
        monkeypatch,
        pr_author="dependabot[bot]",
        head_repo=BASE,
        reviews=[_review("aravhawk", "APPROVED", HEAD_SHA, "2026-10-24T10:00:00Z")],
    )

    assert outputs["should_run"] == "true"
    assert outputs["head_repo"] == BASE


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
