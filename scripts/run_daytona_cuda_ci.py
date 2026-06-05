"""Run TensorPoW CUDA determinism checks inside a Daytona GPU sandbox."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from dataclasses import dataclass
from typing import Protocol, cast

from daytona import (
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    Daytona,
    DaytonaConfig,
    Resources,
)

DEFAULT_REPOSITORY = "aravhawk/TensorPoW"
DEFAULT_SNAPSHOT = "daytona-gpu"
DEFAULT_SNAPSHOT_IMAGE = "python:3.12"
DEFAULT_EXPECTED_DEVICE_TOKENS = "h100"
SANDBOX_REPO_DIR = "/tmp/tensorpow"
SANDBOX_PYTHON = f"{SANDBOX_REPO_DIR}/.venv/bin/python"
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


class SnapshotService(Protocol):
    def get(self, name: str) -> object: ...

    def create(
        self,
        params: CreateSnapshotParams,
        *,
        on_logs: object | None = None,
        timeout: float | None = 0,
    ) -> object: ...


class DaytonaClient(Protocol):
    snapshot: SnapshotService

    def create(
        self,
        params: CreateSandboxFromSnapshotParams | None = None,
        *,
        timeout: float = 60,
        on_snapshot_create_logs: object | None = None,
    ) -> object: ...


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
        default=os.environ.get("DAYTONA_TARGET"),
        help="Optional Daytona target/region. Defaults to the organization default.",
    )
    parser.add_argument(
        "--snapshot-image",
        default=os.environ.get("DAYTONA_GPU_SNAPSHOT_IMAGE", DEFAULT_SNAPSHOT_IMAGE),
        help="Base image to use if the Daytona GPU snapshot must be created.",
    )
    parser.add_argument(
        "--expected-device-tokens",
        default=os.environ.get(
            "DAYTONA_GPU_EXPECTED_DEVICE_TOKENS",
            DEFAULT_EXPECTED_DEVICE_TOKENS,
        ),
        help="Comma- or space-separated tokens that must appear in the CUDA device name.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("DAYTONA_CUDA_COMMAND_TIMEOUT", "1800")),
        help="Per-command timeout in seconds.",
    )
    return parser.parse_args()


def validate_inputs(
    repository: str,
    sha: str,
    snapshot: str,
    snapshot_image: str,
    expected_device_tokens: tuple[str, ...],
) -> None:
    if not REPOSITORY_RE.fullmatch(repository):
        raise ValueError(f"invalid repository: {repository!r}")
    if not SHA_RE.fullmatch(sha):
        raise ValueError("sha must be a 40-character hex commit")
    if not snapshot.strip():
        raise ValueError("DAYTONA_GPU_SNAPSHOT must not be empty")
    if not snapshot_image.strip():
        raise ValueError("DAYTONA_GPU_SNAPSHOT_IMAGE must not be empty")
    if not expected_device_tokens:
        raise ValueError("DAYTONA_GPU_EXPECTED_DEVICE_TOKENS must not be empty")
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


def parse_expected_device_tokens(value: str) -> tuple[str, ...]:
    tokens = tuple(token for token in re.split(r"[\s,]+", value.lower().strip()) if token)
    return tokens


def assert_expected_gpu(
    sandbox: Sandbox,
    *,
    expected_device_tokens: tuple[str, ...],
    timeout: int,
) -> None:
    run(sandbox, "nvidia-smi", timeout=timeout)
    result = run(
        sandbox,
        f"{shlex.quote(SANDBOX_PYTHON)} - <<'PY'\n"
        "import torch\n"
        "assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'\n"
        "print(torch.cuda.get_device_name(0))\n"
        "PY",
        cwd=SANDBOX_REPO_DIR,
        timeout=timeout,
    )
    device_name = result.output.strip().splitlines()[-1].lower()
    missing = [token for token in expected_device_tokens if token not in device_name]
    if missing:
        raise RuntimeError(
            "Daytona GPU device does not match expected GPU: "
            f"{device_name!r} is missing {', '.join(missing)}"
        )


def ensure_gpu_snapshot(daytona: DaytonaClient, *, name: str, image: str) -> None:
    try:
        daytona.snapshot.get(name)
    except Exception as exc:
        if "not found" not in str(exc).lower():
            raise
        if name == DEFAULT_SNAPSHOT:
            raise RuntimeError(
                f"Daytona GPU snapshot {name!r} was not found. "
                "Daytona docs expect this prebuilt GPU snapshot in us-east-1."
            ) from exc
        print(f"Daytona GPU snapshot {name!r} was not found; creating it from {image!r}.")
        daytona.snapshot.create(
            CreateSnapshotParams(
                name=name,
                image=image,
                resources=Resources(gpu=1),
            ),
            timeout=0,
        )


def main() -> int:
    args = parse_args()
    repository = str(args.repository)
    sha = str(args.sha)
    snapshot = str(args.snapshot)
    snapshot_image = str(args.snapshot_image)
    expected_device_tokens = parse_expected_device_tokens(str(args.expected_device_tokens))
    target = str(args.target) if args.target else None
    timeout = int(args.timeout)

    validate_inputs(repository, sha, snapshot, snapshot_image, expected_device_tokens)
    config = DaytonaConfig(target=target)
    daytona = cast(DaytonaClient, Daytona(config))
    sandbox: Sandbox | None = None

    try:
        ensure_gpu_snapshot(daytona, name=snapshot, image=snapshot_image)
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
        target_label = target or "organization default"
        print(
            f"Created Daytona sandbox {sandbox.id} "
            f"from snapshot {snapshot!r} in target {target_label!r}"
        )

        clone_url = f"https://github.com/{repository}.git"
        run(
            sandbox,
            shell_join(
                "git",
                "clone",
                "--no-tags",
                "--filter=blob:none",
                clone_url,
                SANDBOX_REPO_DIR,
            ),
            timeout=timeout,
        )
        run(
            sandbox,
            shell_join("git", "fetch", "--no-tags", "--depth", "1", "origin", sha),
            cwd=SANDBOX_REPO_DIR,
            timeout=timeout,
        )
        run(
            sandbox,
            shell_join("git", "checkout", "--detach", sha),
            cwd=SANDBOX_REPO_DIR,
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
            "if ! command -v uv >/dev/null 2>&1; then "
            "curl -LsSf https://astral.sh/uv/install.sh | sh; "
            "fi; "
            'UV_BIN="$(command -v uv || printf %s "$HOME/.local/bin/uv")"; '
            '"$UV_BIN" venv --python 3.12 .venv; '
            f'"$UV_BIN" pip install --python {shlex.quote(SANDBOX_PYTHON)} -e ".[dev]"',
            cwd=SANDBOX_REPO_DIR,
            timeout=timeout,
        )
        assert_expected_gpu(sandbox, expected_device_tokens=expected_device_tokens, timeout=timeout)
        run(
            sandbox,
            f"TENSORPOW_DETERMINISM_BACKEND=cuda {shlex.quote(SANDBOX_PYTHON)} "
            "-m pytest -m determinism tests/determinism",
            cwd=SANDBOX_REPO_DIR,
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
