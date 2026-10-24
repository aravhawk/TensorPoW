"""Unit tests for the Daytona CUDA CI launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from daytona import (
    CreateSandboxFromSnapshotParams,
    DaytonaBadRequestError,
    DaytonaForbiddenError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
    SandboxState,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_daytona_cuda_ci.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_daytona_cuda_ci", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ci = _load_script()


class FakeSnapshots:
    def __init__(self, get_error: Exception | None = None) -> None:
        self.get_error = get_error
        self.created: list[object] = []

    def get(self, name: str) -> object:
        if self.get_error is not None:
            raise self.get_error
        return object()

    def create(self, params: object, **_: object) -> object:
        self.created.append(params)
        return object()


class FakeSandbox:
    def __init__(self, sandbox_id: str, state: SandboxState = SandboxState.STARTED) -> None:
        self.id = sandbox_id
        self.state = state
        self.deleted = False

    def delete(self, timeout: float | None = 60, wait: bool = False) -> None:
        self.deleted = True


class FakeDaytona:
    def __init__(
        self,
        create_errors: list[BaseException] | None = None,
        orphans: dict[str, FakeSandbox] | None = None,
    ) -> None:
        self.snapshot = FakeSnapshots()
        self.create_errors = list(create_errors or [])
        self.create_timeouts: list[float] = []
        self.created_names: list[str | None] = []
        self.orphans = orphans or {}
        self.listed: list[FakeSandbox] = []
        self.list_queries: list[object] = []

    def create(
        self, params: CreateSandboxFromSnapshotParams, *, timeout: float = 60, **_: object
    ) -> object:
        self.create_timeouts.append(timeout)
        self.created_names.append(params.name)
        if self.create_errors:
            raise self.create_errors.pop(0)
        return object()

    def get(self, sandbox_id_or_name: str, request_timeout: float | None = None) -> FakeSandbox:
        if sandbox_id_or_name not in self.orphans:
            raise DaytonaNotFoundError("Sandbox not found", status_code=404)
        return self.orphans[sandbox_id_or_name]

    def list(self, query: object = None, request_timeout: float | None = None) -> list[FakeSandbox]:
        self.list_queries.append(query)
        return self.listed


def test_default_gpu_snapshot_must_exist() -> None:
    daytona = FakeDaytona()
    daytona.snapshot = FakeSnapshots(DaytonaNotFoundError("Snapshot not found", status_code=404))

    with pytest.raises(RuntimeError, match="shared 'earth' region"):
        ci.ensure_gpu_snapshot(daytona, name=ci.DEFAULT_SNAPSHOT, image="python:3.12")
    assert daytona.snapshot.created == []


def test_missing_custom_gpu_snapshot_is_created() -> None:
    daytona = FakeDaytona()
    daytona.snapshot = FakeSnapshots(DaytonaNotFoundError("Snapshot not found", status_code=404))

    ci.ensure_gpu_snapshot(daytona, name="tensorpow-gpu", image="python:3.12")

    assert len(daytona.snapshot.created) == 1


def test_snapshot_lookup_errors_other_than_not_found_propagate() -> None:
    daytona = FakeDaytona()
    daytona.snapshot = FakeSnapshots(DaytonaBadRequestError("bad request", status_code=400))

    with pytest.raises(DaytonaBadRequestError):
        ci.ensure_gpu_snapshot(daytona, name="tensorpow-gpu", image="python:3.12")


def test_create_timeout_outlasts_gpu_queue_timeout() -> None:
    daytona = FakeDaytona()
    params = CreateSandboxFromSnapshotParams(snapshot="daytona-gpu", queue_timeout=10)

    ci.create_sandbox_with_quota_wait(daytona, params, attempts=1, wait_seconds=1)

    assert daytona.create_timeouts == [10 * 60 + ci.SANDBOX_START_TIMEOUT_SECONDS]


def test_create_timeout_covers_gpu_queue_without_server_queue_timeout() -> None:
    daytona = FakeDaytona()
    params = CreateSandboxFromSnapshotParams(snapshot="daytona-gpu")

    ci.create_sandbox_with_quota_wait(daytona, params, attempts=1, wait_seconds=1)

    expected = ci.DEFAULT_GPU_QUEUE_WAIT_MINUTES * 60 + ci.SANDBOX_START_TIMEOUT_SECONDS
    assert daytona.create_timeouts == [expected]


def test_gpu_quota_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    daytona = FakeDaytona([DaytonaBadRequestError("GPU quota exceeded", status_code=400)])
    params = CreateSandboxFromSnapshotParams(snapshot="daytona-gpu", queue_timeout=1)

    ci.create_sandbox_with_quota_wait(daytona, params, attempts=2, wait_seconds=1)

    assert len(daytona.create_timeouts) == 2


def test_region_errors_are_not_retried() -> None:
    error = DaytonaBadRequestError(
        "Snapshot daytona-gpu is not available in region us-east-1", status_code=400
    )
    daytona = FakeDaytona([error])
    params = CreateSandboxFromSnapshotParams(snapshot="daytona-gpu", queue_timeout=1)

    with pytest.raises(DaytonaBadRequestError, match="us-east-1"):
        ci.create_sandbox_with_quota_wait(daytona, params, attempts=3, wait_seconds=1)
    assert len(daytona.create_timeouts) == 1


def test_non_daytona_errors_are_not_treated_as_gpu_quota() -> None:
    assert not ci.is_gpu_quota_error(RuntimeError("GPU quota exceeded"))


def test_quota_classification_needs_a_limit_error_about_gpus() -> None:
    assert ci.is_gpu_quota_error(DaytonaRateLimitError("Too many requests", status_code=429))
    assert ci.is_gpu_quota_error(
        DaytonaBadRequestError("Insufficient GPU capacity for H100", status_code=400)
    )
    # Mentioning the "daytona-gpu" snapshot is not GPU context on its own.
    assert not ci.is_gpu_quota_error(
        DaytonaBadRequestError("Snapshot daytona-gpu: not enough disk space", status_code=400)
    )
    assert not ci.is_gpu_quota_error(
        DaytonaForbiddenError("Insufficient permissions for GPU sandboxes", status_code=403)
    )


def test_retries_use_a_fresh_sandbox_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    daytona = FakeDaytona([DaytonaBadRequestError("GPU quota exceeded", status_code=400)])
    params = CreateSandboxFromSnapshotParams(name="tensorpow-cuda-run", snapshot="daytona-gpu")

    ci.create_sandbox_with_quota_wait(daytona, params, attempts=2, wait_seconds=1)

    assert daytona.created_names == ["tensorpow-cuda-run-1", "tensorpow-cuda-run-2"]


def test_failed_create_deletes_sandbox_daytona_already_accepted() -> None:
    orphan = FakeSandbox("sandbox-1")
    daytona = FakeDaytona(
        [DaytonaTimeoutError("Sandbox did not start in time")],
        orphans={"tensorpow-cuda-run-1": orphan},
    )
    params = CreateSandboxFromSnapshotParams(name="tensorpow-cuda-run", snapshot="daytona-gpu")

    with pytest.raises(DaytonaTimeoutError):
        ci.create_sandbox_with_quota_wait(daytona, params, attempts=3, wait_seconds=1)
    assert orphan.deleted
    assert len(daytona.create_timeouts) == 1


def test_cancelled_create_deletes_sandbox_daytona_already_accepted() -> None:
    orphan = FakeSandbox("sandbox-1")
    daytona = FakeDaytona([KeyboardInterrupt()], orphans={"tensorpow-cuda-run-1": orphan})
    params = CreateSandboxFromSnapshotParams(name="tensorpow-cuda-run", snapshot="daytona-gpu")

    with pytest.raises(KeyboardInterrupt):
        ci.create_sandbox_with_quota_wait(daytona, params, attempts=3, wait_seconds=1)
    assert orphan.deleted


def test_sandbox_names_are_unique_per_run() -> None:
    sha = "A" * 40

    first, second = ci.sandbox_name(sha), ci.sandbox_name(sha)

    assert first != second
    assert first.startswith("tensorpow-cuda-aaaaaaaaaaaa-")


def test_zero_queue_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="DAYTONA_GPU_QUEUE_TIMEOUT_MINUTES"):
        ci.validate_retry_settings(1, 1, 0)


def test_cleanup_deletes_only_live_sandboxes_for_this_run() -> None:
    live = FakeSandbox("live")
    gone = FakeSandbox("gone", SandboxState.DESTROYED)
    daytona = FakeDaytona()
    daytona.listed = [live, gone]

    assert ci.cleanup_run_sandboxes(daytona, "123-1") == 1

    assert live.deleted and not gone.deleted
    query = daytona.list_queries[0]
    assert query.labels == {ci.CI_RUN_LABEL: "123-1"}


def test_ci_run_label_comes_from_github_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    assert ci.ci_run_label() is None

    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    assert ci.ci_run_label() == "123-2"
