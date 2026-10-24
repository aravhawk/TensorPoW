"""Run TensorPoW CUDA determinism checks inside a Daytona GPU sandbox."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from types import FrameType
from typing import Protocol, cast

from daytona import (
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    Daytona,
    DaytonaBadRequestError,
    DaytonaConfig,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    ListSandboxesQuery,
    Resources,
    SandboxState,
)

DEFAULT_REPOSITORY = "aravhawk/TensorPoW"
DEFAULT_SNAPSHOT = "daytona-gpu"
DEFAULT_SNAPSHOT_IMAGE = "python:3.12"
DEFAULT_EXPECTED_DEVICE_TOKENS = "h100"
DEFAULT_GPU_CREATE_ATTEMPTS = 30
DEFAULT_GPU_CREATE_WAIT_SECONDS = 60
DEFAULT_GPU_QUEUE_WAIT_MINUTES = 30
SANDBOX_START_TIMEOUT_SECONDS = 300
SANDBOX_REPO_DIR = "/tmp/tensorpow"
SANDBOX_PYTHON = f"{SANDBOX_REPO_DIR}/.venv/bin/python"
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
# "gpu" as its own word, so the "daytona-gpu" snapshot name alone doesn't count.
GPU_WORD_RE = re.compile(r"(?<![\w-])gpus?\b|graphics processing unit")
CI_RUN_LABEL = "ci-run"


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
    state: object

    @property
    def process(self) -> SandboxProcess: ...

    def delete(self, timeout: float | None = 60, wait: bool = False) -> None: ...


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

    def get(self, sandbox_id_or_name: str, request_timeout: float | None = None) -> object: ...

    def list(
        self, query: ListSandboxesQuery | None = None, request_timeout: float | None = None
    ) -> Iterable[object]: ...

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
        help=(
            "Optional Daytona target/region. Leave unset for shared GPU sandboxes, "
            "which Daytona schedules in its global 'earth' region."
        ),
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
    parser.add_argument(
        "--gpu-create-attempts",
        type=int,
        default=int(
            os.environ.get(
                "DAYTONA_GPU_CREATE_ATTEMPTS",
                str(DEFAULT_GPU_CREATE_ATTEMPTS),
            )
        ),
        help="Sandbox creation attempts when Daytona reports GPU quota exhaustion.",
    )
    parser.add_argument(
        "--gpu-create-wait-seconds",
        type=int,
        default=int(
            os.environ.get(
                "DAYTONA_GPU_CREATE_WAIT_SECONDS",
                str(DEFAULT_GPU_CREATE_WAIT_SECONDS),
            )
        ),
        help="Seconds to wait between GPU quota retry attempts.",
    )
    queue_timeout_env = os.environ.get("DAYTONA_GPU_QUEUE_TIMEOUT_MINUTES")
    parser.add_argument(
        "--gpu-queue-timeout-minutes",
        type=int,
        default=int(queue_timeout_env) if queue_timeout_env else None,
        help=(
            "Minutes Daytona may queue the sandbox for GPU capacity before cancelling it. "
            "Requires Daytona's queue timeout feature for the organization."
        ),
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Only delete sandboxes left behind by this GitHub Actions run, then exit.",
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


def validate_retry_settings(
    attempts: int, wait_seconds: int, queue_timeout_minutes: int | None
) -> None:
    if attempts < 1:
        raise ValueError("DAYTONA_GPU_CREATE_ATTEMPTS must be at least 1")
    if wait_seconds < 1:
        raise ValueError("DAYTONA_GPU_CREATE_WAIT_SECONDS must be at least 1")
    if queue_timeout_minutes is not None and queue_timeout_minutes < 1:
        raise ValueError("DAYTONA_GPU_QUEUE_TIMEOUT_MINUTES must be at least 1")


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


def is_gpu_quota_error(exc: BaseException) -> bool:
    if isinstance(exc, DaytonaRateLimitError):
        return True
    # Daytona reports quota and resource-limit rejections as HTTP 400.
    if not isinstance(exc, DaytonaBadRequestError):
        return False
    message = str(exc).lower()
    has_gpu_context = GPU_WORD_RE.search(message) is not None
    has_limit_context = (
        "quota" in message
        or "rate limit" in message
        or "limit exceeded" in message
        or "maximum allowed" in message
        or "too many" in message
        or "capacity" in message
        or "insufficient" in message
        or "not enough" in message
    )
    return has_gpu_context and has_limit_context


def sandbox_name(sha: str) -> str:
    return f"tensorpow-cuda-{sha[:12].lower()}-{uuid.uuid4().hex[:12]}"


def ci_run_label() -> str | None:
    """Identify the current GitHub Actions run attempt, if running in Actions."""

    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        return None
    return f"{run_id}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"


def delete_orphaned_sandbox(daytona: DaytonaClient, name: str) -> None:
    """Delete a sandbox that Daytona accepted before a create call failed."""

    try:
        sandbox = cast(Sandbox, daytona.get(name, request_timeout=10))
    except DaytonaNotFoundError:
        return
    print(f"Deleting Daytona sandbox {sandbox.id} left by a failed create", flush=True)
    # Don't wait for 'destroyed': a cancelled job only has seconds to clean up.
    sandbox.delete(timeout=60)


def create_sandbox_with_quota_wait(
    daytona: DaytonaClient,
    params: CreateSandboxFromSnapshotParams,
    *,
    attempts: int,
    wait_seconds: int,
) -> Sandbox:
    # On-demand GPU sandboxes queue server-side until capacity frees up, so the
    # client timeout must outlast the queue wait plus sandbox startup.
    queue_wait_minutes = params.queue_timeout or DEFAULT_GPU_QUEUE_WAIT_MINUTES
    create_timeout = queue_wait_minutes * 60 + SANDBOX_START_TIMEOUT_SECONDS
    for attempt in range(1, attempts + 1):
        # A fresh name per attempt, so a retry never collides with a sandbox that a
        # failed attempt left behind while it is still being deleted.
        attempt_params = (
            params.model_copy(update={"name": f"{params.name}-{attempt}"})
            if params.name
            else params
        )
        try:
            return cast(Sandbox, daytona.create(attempt_params, timeout=create_timeout))
        except BaseException as exc:
            # A create can time out, fail, or be cancelled after Daytona has accepted
            # (and queued) the sandbox, leaving no handle to delete it; find it by name.
            if attempt_params.name:
                try:
                    delete_orphaned_sandbox(daytona, attempt_params.name)
                except Exception as cleanup_exc:
                    print(
                        f"Failed to clean up Daytona sandbox {attempt_params.name!r}: "
                        f"{cleanup_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            if not is_gpu_quota_error(exc):
                raise
            if attempt == attempts:
                raise
            print(
                "Daytona GPU quota is currently unavailable; "
                f"waiting {wait_seconds}s before retry {attempt + 1}/{attempts}.",
                flush=True,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("unreachable GPU sandbox retry state")


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
    except DaytonaNotFoundError as exc:
        if name == DEFAULT_SNAPSHOT:
            raise RuntimeError(
                f"Daytona GPU snapshot {name!r} was not found. "
                "Daytona provides this prebuilt GPU snapshot in its shared 'earth' region."
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


def cleanup_run_sandboxes(daytona: DaytonaClient, run_label: str) -> int:
    """Delete every live sandbox labelled with this CI run; return how many."""

    deleted = 0
    query = ListSandboxesQuery(labels={CI_RUN_LABEL: run_label})
    for item in daytona.list(query, request_timeout=30):
        sandbox = cast(Sandbox, item)
        if sandbox.state in (SandboxState.DESTROYED, SandboxState.DESTROYING):
            continue
        print(f"Deleting leftover Daytona sandbox {sandbox.id}", flush=True)
        sandbox.delete(timeout=60)
        deleted += 1
    return deleted


def main() -> int:
    args = parse_args()
    if args.cleanup:
        if not os.environ.get("DAYTONA_API_KEY"):
            raise ValueError("DAYTONA_API_KEY is required")
        run_label = ci_run_label()
        if run_label is None:
            raise ValueError("--cleanup only works inside GitHub Actions (GITHUB_RUN_ID unset)")
        target = str(args.target) if args.target else None
        client = cast(DaytonaClient, Daytona(DaytonaConfig(target=target)))
        deleted = cleanup_run_sandboxes(client, run_label)
        print(f"Deleted {deleted} leftover Daytona sandbox(es) for run {run_label}.")
        return 0

    repository = str(args.repository)
    sha = str(args.sha)
    snapshot = str(args.snapshot)
    snapshot_image = str(args.snapshot_image)
    expected_device_tokens = parse_expected_device_tokens(str(args.expected_device_tokens))
    target = str(args.target) if args.target else None
    timeout = int(args.timeout)
    gpu_create_attempts = int(args.gpu_create_attempts)
    gpu_create_wait_seconds = int(args.gpu_create_wait_seconds)
    gpu_queue_timeout_minutes = (
        int(args.gpu_queue_timeout_minutes) if args.gpu_queue_timeout_minutes is not None else None
    )

    validate_inputs(repository, sha, snapshot, snapshot_image, expected_device_tokens)
    validate_retry_settings(gpu_create_attempts, gpu_create_wait_seconds, gpu_queue_timeout_minutes)
    config = DaytonaConfig(target=target)
    daytona = cast(DaytonaClient, Daytona(config))
    sandbox: Sandbox | None = None

    try:
        ensure_gpu_snapshot(daytona, name=snapshot, image=snapshot_image)
        sandbox = create_sandbox_with_quota_wait(
            daytona,
            CreateSandboxFromSnapshotParams(
                name=sandbox_name(sha),
                snapshot=snapshot,
                language="python",
                ephemeral=True,
                auto_stop_interval=15,
                queue_timeout=gpu_queue_timeout_minutes,
                labels={
                    "repo": repository.replace("/", "-"),
                    "commit": sha,
                    "purpose": "tensorpow-cuda-ci",
                    **({CI_RUN_LABEL: run_label} if (run_label := ci_run_label()) else {}),
                },
            ),
            attempts=gpu_create_attempts,
            wait_seconds=gpu_create_wait_seconds,
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
            # Don't let a slow or failed teardown replace the test result; in Actions
            # the cleanup step retries deletion of anything left behind.
            try:
                sandbox.delete(timeout=120, wait=True)
            except Exception as exc:
                print(f"Failed to delete Daytona sandbox {sandbox.id}: {exc}", file=sys.stderr)


def _interrupt_on_sigterm(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    # GitHub Actions cancels with SIGINT, then SIGTERM; turn both into
    # KeyboardInterrupt so the sandbox cleanup paths run.
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Daytona CUDA CI failed: {exc}", file=sys.stderr)
        raise
