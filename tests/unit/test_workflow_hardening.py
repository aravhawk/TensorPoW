"""M-10 / L-5 / L-9: CI workflows pin third-party actions and keep tokens minimal."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = sorted((Path(__file__).resolve().parents[2] / ".github" / "workflows").glob("*.yml"))
USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)(.*)$", re.MULTILINE)


def test_every_external_action_is_pinned_to_a_full_commit_sha() -> None:
    assert WORKFLOWS
    for workflow in WORKFLOWS:
        for reference, trailer in USES_RE.findall(workflow.read_text(encoding="utf-8")):
            if reference.startswith("./"):
                continue
            _action, _, ref = reference.partition("@")
            assert re.fullmatch(r"[0-9a-f]{40}", ref), (
                f"{workflow.name}: {reference} not SHA-pinned"
            )
            assert re.search(r"#\s*v\d+\.\d+\.\d+", trailer), f"{workflow.name}: missing version"


def test_workflows_default_to_no_token_permissions() -> None:
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        assert re.search(r"^permissions: \{\}$", text, re.MULTILINE), workflow.name
