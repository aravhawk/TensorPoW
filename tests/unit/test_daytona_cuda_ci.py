"""Unit tests for the Daytona CUDA CI launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from daytona import CreateSandboxFromSnapshotParams, DaytonaBadRequestError, DaytonaNotFoundError

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


class FakeDaytona:
    def __init__(self, create_errors: list[Exception] | None = None) -> None:
        self.snapshot = FakeSnapshots()
        self.create_errors = list(create_errors or [])
        self.create_timeouts: list[float] = []

    def create(self, params: object = None, *, timeout: float = 60, **_: object) -> object:
        self.create_timeouts.append(timeout)
        if self.create_errors:
            raise self.create_errors.pop(0)
        return object()


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
