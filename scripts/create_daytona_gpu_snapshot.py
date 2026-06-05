"""Create or verify TensorPoW's Daytona GPU snapshot from a committed template."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from daytona import CreateSnapshotParams, Daytona, DaytonaConfig, Resources

DEFAULT_TEMPLATE = "daytona/snapshots/tensorpow-h100.json"


@dataclass(frozen=True)
class SnapshotSpec:
    name: str
    target: str | None
    image: str
    resources: Resources


class Snapshot(Protocol):
    name: str
    state: str
    image_name: str


class SnapshotService(Protocol):
    def get(self, name: str) -> Snapshot: ...

    def create(
        self,
        params: CreateSnapshotParams,
        *,
        on_logs: object | None = None,
        timeout: float | None = 0,
    ) -> Snapshot: ...

    def activate(self, snapshot: Snapshot) -> Snapshot: ...


class DaytonaClient(Protocol):
    snapshot: SnapshotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        default=os.environ.get("DAYTONA_GPU_SNAPSHOT_TEMPLATE", DEFAULT_TEMPLATE),
        help="Path to the committed Daytona snapshot template.",
    )
    parser.add_argument(
        "--target",
        default=os.environ.get("DAYTONA_TARGET"),
        help="Daytona target/region override. Defaults to the template target.",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Only verify that the snapshot exists; do not create it.",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate the snapshot if Daytona reports it as inactive.",
    )
    return parser.parse_args()


def load_spec(path: Path, target_override: str | None) -> SnapshotSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    resources_payload = payload.get("resources", {})
    if not isinstance(resources_payload, dict):
        raise ValueError("resources must be an object")

    name = require_string(payload, "name")
    target = target_override or optional_string(payload, "target")
    image = require_string(payload, "image")
    gpu = resources_payload.get("gpu")
    if gpu != 1:
        raise ValueError("resources.gpu must be 1 for the TensorPoW CUDA CI snapshot")

    return SnapshotSpec(
        name=name,
        target=target,
        image=image,
        resources=Resources(
            cpu=optional_int(resources_payload, "cpu"),
            memory=optional_int(resources_payload, "memory"),
            disk=optional_int(resources_payload, "disk"),
            gpu=1,
        ),
    )


def require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be null or a non-empty string")
    if value in {"auto", "default"}:
        return None
    return value


def optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"resources.{key} must be a positive integer")
    return value


def ensure_snapshot(daytona: DaytonaClient, spec: SnapshotSpec, *, create: bool) -> Snapshot:
    try:
        snapshot = daytona.snapshot.get(spec.name)
        print(
            f"Found Daytona snapshot {snapshot.name!r}: "
            f"state={snapshot.state!r}, image={snapshot.image_name!r}"
        )
        return snapshot
    except Exception as exc:
        if "not found" not in str(exc).lower():
            raise
        if not create:
            raise RuntimeError(f"Daytona snapshot {spec.name!r} does not exist") from exc

    print(f"Creating Daytona GPU snapshot {spec.name!r} from image {spec.image!r}.")
    return daytona.snapshot.create(
        CreateSnapshotParams(
            name=spec.name,
            image=spec.image,
            resources=spec.resources,
        ),
        timeout=0,
    )


def main() -> int:
    args = parse_args()
    if not os.environ.get("DAYTONA_API_KEY"):
        raise ValueError("DAYTONA_API_KEY is required")

    spec = load_spec(Path(args.template), str(args.target) if args.target else None)
    daytona = cast(DaytonaClient, Daytona(DaytonaConfig(target=spec.target)))
    snapshot = ensure_snapshot(daytona, spec, create=not bool(args.no_create))

    if bool(args.activate) and snapshot.state.lower() == "inactive":
        snapshot = daytona.snapshot.activate(snapshot)
        print(f"Activated Daytona snapshot {snapshot.name!r}; state={snapshot.state!r}")

    target_label = spec.target or "organization default"
    print(f"Daytona GPU snapshot {snapshot.name!r} is configured for target {target_label!r}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Daytona GPU snapshot setup failed: {exc}", file=sys.stderr)
        raise
