"""Run TensorPoW CUDA determinism checks inside a Daytona GPU sandbox."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from dataclasses import dataclass
from typing import Protocol, cast

from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig

DEFAULT_REPOSITORY = "aravhawk/TensorPoW"
DEFAULT_SNAPSHOT = "tensorpow-rtx-pro-6000"
DEFAULT_TARGET = "us-east-1"
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    output: str


class ExecuteResponse(Protocol):
    exit_code: int
    result: object


class SandboxProcess(Protocol):
    def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecuteResponse: ...


class Sandbox(Protocol):
    id: str

    @property
    def process(self) -> SandboxProcess: ...

    def delete(self, timeout: float | None = 60) -> None: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=os.environ.get(
            "TENSORPOW_CI_REPOSITORY",
            os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY),
        ),
        help="GitHub repository to clone, in owner/name form.",
    )
    parser.add_argument(
        "--sha",
        default=os.environ.get("TENSORPOW_CI_SHA", os.environ.get("GITHUB_SHA", "")),
        help="Exact 40-character commit SHA to test.",
    )
    parser.add_argument(
        "--snapshot",
        default=os.environ.get("DAYTONA_GPU_SNAPSHOT", DEFAULT_SNAPSHOT),
        help="Daytona GPU snapshot name or ID.",
    )
    parser.add_argument(
        "--target",
        default=os.environ.get("DAYTONA_TARGET", DEFAULT_TARGET),
        help="Daytona target/region.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("DAYTONA_CUDA_COMMAND_TIMEOUT", "1800")),
        help="Per-command timeout in seconds.",
    )
    return parser.parse_args()


def validate_inputs(repository: str, sha: str, snapshot: str) -> None:
    if not REPOSITORY_RE.fullmatch(repository):
        raise ValueError(f"invalid repository: {repository!r}")
    if not SHA_RE.fullmatch(sha):
        raise ValueError("sha must be a 40-character hex commit")
    if not snapshot.strip():
        raise ValueError("DAYTONA_GPU_SNAPSHOT must not be empty")
    if not os.environ.get("DAYTONA_API_KEY"):
        raise ValueError("DAYTONA_API_KEY is required")


def run(sandbox: Sandbox, command: str, *, cwd: str | None = None, timeout: int) -> CommandResult:
    print(f"\n$ {command}", flush=True)
    response = sandbox.process.exec(command, cwd=cwd, timeout=timeout)
    output = str(response.result)
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    result = CommandResult(exit_code=int(response.exit_code), output=output)
    if result.exit_code != 0:
        raise RuntimeError(f"command failed with exit code {result.exit_code}: {command}")
    return result


def shell_join(*parts: str) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def assert_rtx_pro_6000(sandbox: Sandbox, *, timeout: int) -> None:
    run(sandbox, "nvidia-smi", timeout=timeout)
    result = run(
        sandbox,
        "python - <<'PY'\n"
        "import torch\n"
        "assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'\n"
        "print(torch.cuda.get_device_name(0))\n"
        "PY",
        cwd="/workspace/tensorpow",
        timeout=timeout,
    )
    device_name = result.output.strip().splitlines()[-1].lower()
    missing = [token for token in ("rtx", "pro", "6000") if token not in device_name]
    if missing:
        raise RuntimeError(
            "Daytona GPU device is not RTX PRO 6000: "
            f"{device_name!r} is missing {', '.join(missing)}"
        )


def main() -> int:
    args = parse_args()
    repository = str(args.repository)
    sha = str(args.sha)
    snapshot = str(args.snapshot)
    target = str(args.target)
    timeout = int(args.timeout)

    validate_inputs(repository, sha, snapshot)
    config = DaytonaConfig(target=target)
    daytona = Daytona(config)
    sandbox: Sandbox | None = None

    try:
        sandbox = cast(
            Sandbox,
            daytona.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=snapshot,
                    language="python",
                    ephemeral=True,
                    auto_stop_interval=15,
                    labels={
                        "repo": repository.replace("/", "-"),
                        "commit": sha,
                        "purpose": "tensorpow-cuda-ci",
                    },
                ),
                timeout=180,
            ),
        )
        print(
            f"Created Daytona sandbox {sandbox.id} from snapshot {snapshot!r} in target {target!r}"
        )

        clone_url = f"https://github.com/{repository}.git"
        run(sandbox, "mkdir -p /workspace", timeout=timeout)
        run(
            sandbox,
            shell_join(
                "git",
                "clone",
                "--no-tags",
                "--filter=blob:none",
                clone_url,
                "/workspace/tensorpow",
            ),
            timeout=timeout,
        )
        run(
            sandbox,
            shell_join("git", "fetch", "--no-tags", "--depth", "1", "origin", sha),
            cwd="/workspace/tensorpow",
            timeout=timeout,
        )
        run(
            sandbox,
            shell_join("git", "checkout", "--detach", sha),
            cwd="/workspace/tensorpow",
            timeout=timeout,
        )
        run(
            sandbox,
            "if command -v apt-get >/dev/null 2>&1; then "
            'if [ "$(id -u)" -eq 0 ]; then apt-get update && apt-get install -y libgmp-dev; '
            "else sudo apt-get update && sudo apt-get install -y libgmp-dev; fi; fi",
            timeout=timeout,
        )
        run(
            sandbox,
            'python -m pip install -e ".[dev]"',
            cwd="/workspace/tensorpow",
            timeout=timeout,
        )
        assert_rtx_pro_6000(sandbox, timeout=timeout)
        run(
            sandbox,
            "TENSORPOW_DETERMINISM_BACKEND=cuda pytest -m determinism tests/determinism",
            cwd="/workspace/tensorpow",
            timeout=timeout,
        )
        return 0
    finally:
        if sandbox is not None:
            print(f"Deleting Daytona sandbox {sandbox.id}")
            sandbox.delete(timeout=120)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Daytona CUDA CI failed: {exc}", file=sys.stderr)
        raise
