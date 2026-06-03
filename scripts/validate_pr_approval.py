"""Validate that an approved PR review still authorizes CI for the current head SHA."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

AUTHORIZED_PERMISSIONS = {"admin", "maintain", "write"}


@dataclass(frozen=True)
class Approval:
    pr_number: int
    reviewer: str
    state: str
    head_sha: str
    head_repo: str
    head_ref: str


def fail(message: str) -> NoReturn:
    print(f"approval validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def output(name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    print(f"{name}={value}")


def api_json(path: str) -> Any:
    repository = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        fail(f"GitHub API {path!r} returned {exc.code}: {detail}")


def load_approval() -> Approval | None:
    artifact_dir = Path(os.environ.get("APPROVAL_ARTIFACT_DIR", "approval-artifacts"))
    files = sorted(artifact_dir.glob("**/approval.json"))
    if not files:
        output("should_run", "false")
        return None
    if len(files) > 1:
        fail(f"expected one approval artifact, found {len(files)}")

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    return Approval(
        pr_number=int(payload["pr_number"]),
        reviewer=str(payload["reviewer"]),
        state=str(payload["state"]).lower(),
        head_sha=str(payload["head_sha"]),
        head_repo=str(payload["head_repo"]),
        head_ref=str(payload["head_ref"]),
    )


def validate(approval: Approval) -> None:
    if approval.state != "approved":
        output("should_run", "false")
        return

    permission_payload = api_json(f"collaborators/{approval.reviewer}/permission")
    permission = str(permission_payload.get("permission", "")).lower()
    if permission not in AUTHORIZED_PERMISSIONS:
        fail(
            f"{approval.reviewer!r} has permission {permission!r}, "
            f"not one of {sorted(AUTHORIZED_PERMISSIONS)}"
        )

    pr_payload = api_json(f"pulls/{approval.pr_number}")
    if str(pr_payload.get("state")) != "open":
        output("should_run", "false")
        return

    head = pr_payload.get("head", {})
    current_sha = str(head.get("sha", ""))
    current_repo = str(head.get("repo", {}).get("full_name", ""))
    current_ref = str(head.get("ref", ""))
    if current_sha != approval.head_sha:
        output("should_run", "false")
        return
    if current_repo != approval.head_repo:
        fail(
            f"artifact head repo {approval.head_repo!r} "
            f"does not match current PR head repo {current_repo!r}"
        )

    reviews = api_json(f"pulls/{approval.pr_number}/reviews?per_page=100")
    if not isinstance(reviews, list):
        fail("reviews API returned a non-list payload")

    latest_state = ""
    latest_reviewer = ""
    latest_submitted_at = ""
    for review in reviews:
        if str(review.get("commit_id", "")) != current_sha:
            continue
        user = review.get("user") or {}
        reviewer = str(user.get("login", ""))
        if not reviewer:
            continue
        review_permission = api_json(f"collaborators/{reviewer}/permission")
        if str(review_permission.get("permission", "")).lower() not in AUTHORIZED_PERMISSIONS:
            continue
        submitted_at = str(review.get("submitted_at", ""))
        if submitted_at >= latest_submitted_at:
            latest_state = str(review.get("state", "")).lower()
            latest_reviewer = reviewer
            latest_submitted_at = submitted_at

    if latest_state != "approved":
        output("should_run", "false")
        return

    output("should_run", "true")
    output("pr_number", str(approval.pr_number))
    output("head_sha", current_sha)
    output("head_repo", current_repo)
    output("head_ref", current_ref)
    output("approved_by", latest_reviewer)


def main() -> int:
    output("should_run", "false")
    approval = load_approval()
    if approval is None:
        return 0
    validate(approval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
