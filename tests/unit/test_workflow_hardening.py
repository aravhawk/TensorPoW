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


def test_daytona_key_is_step_scoped_and_runner_deps_are_hash_pinned() -> None:
    """L-5: the Daytona key never sits in job env next to unpinned installs."""

    requirements = WORKFLOWS[0].parents[1] / "requirements" / "daytona-runner.txt"
    pinned = requirements.read_text(encoding="utf-8")
    assert "daytona==" in pinned
    assert pinned.count("--hash=sha256:") >= pinned.count("==")
    for workflow in WORKFLOWS:
        lines = workflow.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "pip install" in line and "daytona" in line:
                assert "--require-hashes" in line, workflow.name
            if re.match(r"^\s+DAYTONA_API_KEY: \$\{\{ secrets\.DAYTONA_API_KEY \}\}$", line):
                indent = len(line) - len(line.lstrip())
                previous = lines[index - 1].strip()
                assert indent == 10 or previous == "secrets:", (workflow.name, index + 1)


def test_sandbox_bootstrap_verifies_pinned_uv_and_installs_from_lockfile() -> None:
    import scripts.run_daytona_cuda_ci as runner

    command = runner.sandbox_install_command()
    assert not re.search(r"\|\s*(?:ba)?sh\b", command)
    assert "install.sh" not in command
    for digest in runner.UV_RELEASE_SHA256.values():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert digest in command
    assert command.index("sha256sum -c") < command.index("tar -xzf")
    assert f"/download/{runner.UV_VERSION}/" in command
    assert "sync --locked" in command
