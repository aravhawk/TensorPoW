"""Command-line node lifecycle controls for TensorPoW."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Final, cast

DEFAULT_NODE_DIR: Final[Path] = Path.home() / ".tensorpow" / "node"
CONFIG_FILENAME: Final[str] = "tensorpow.toml"
PID_FILENAME: Final[str] = "tensorpow.pid"
PID_LOCK_FILENAME: Final[str] = f"{PID_FILENAME}.lock"
STATUS_FILENAME: Final[str] = "status.json"
PEERS_FILENAME: Final[str] = "peers.json"
LOG_FILENAME: Final[str] = "node.log"

DEFAULT_NETWORK: Final[str] = "local"
DEFAULT_LISTEN: Final[str] = "/ip4/127.0.0.1/tcp/0"
START_TIMEOUT_SECONDS: Final[float] = 3.0
STOP_TIMEOUT_SECONDS: Final[float] = 5.0

type _Command = Callable[[argparse.Namespace], int]


class NodeCliError(ValueError):
    """Raised when node CLI state or arguments are invalid."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``tensorpow-node`` CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = cast(_Command, args.handler)
    try:
        return handler(args)
    except (OSError, TypeError, ValueError, NodeCliError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensorpow-node")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    _add_data_dir(init)
    init.add_argument("--network", default=DEFAULT_NETWORK)
    init.add_argument("--listen", default=DEFAULT_LISTEN)
    init.add_argument("--overwrite", action="store_true")
    init.set_defaults(handler=_cmd_init)

    start = subparsers.add_parser("start")
    _add_data_dir(start)
    start.set_defaults(handler=_cmd_start)

    stop = subparsers.add_parser("stop")
    _add_data_dir(stop)
    stop.set_defaults(handler=_cmd_stop)

    status = subparsers.add_parser("status")
    _add_data_dir(status)
    status.set_defaults(handler=_cmd_status)

    peers = subparsers.add_parser("peers")
    _add_data_dir(peers)
    peers.set_defaults(handler=_cmd_peers)

    serve = subparsers.add_parser("_serve", help=argparse.SUPPRESS)
    _add_data_dir(serve)
    serve.set_defaults(handler=_cmd_serve)
    return parser


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_NODE_DIR)


def _cmd_init(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    config_path = _config_path(data_dir)
    if config_path.exists() and not args.overwrite:
        raise FileExistsError("node config already exists")
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_config(data_dir, network=args.network, listen=args.listen)
    _write_peers(data_dir, ())
    _write_status(
        data_dir,
        {
            "listen": args.listen,
            "network": args.network,
            "pid": None,
            "running": False,
            "started_at_ms": None,
        },
    )
    _print_json({"data_dir": str(data_dir), "initialized": True, "network": args.network})
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    config = _load_config(data_dir)
    existing_pid = _running_pid_or_clear_stale(data_dir)
    if existing_pid is not None:
        _print_json({"data_dir": str(data_dir), "pid": existing_pid, "running": True})
        return 0

    log_path = data_dir / LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorpow.cli.node",
                "_serve",
                "--data-dir",
                str(data_dir),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            running_pid = _running_pid_or_clear_stale(data_dir)
            if running_pid is not None:
                _print_json(
                    {
                        "data_dir": str(data_dir),
                        "listen": config.listen,
                        "network": config.network,
                        "pid": running_pid,
                        "running": True,
                    }
                )
                return 0
            raise NodeCliError("node process exited during startup")
        pid = _read_pid(data_dir)
        if pid == process.pid and _pid_is_running(pid):
            _print_json(
                {
                    "data_dir": str(data_dir),
                    "listen": config.listen,
                    "network": config.network,
                    "pid": pid,
                    "running": True,
                }
            )
            return 0
        if pid is not None and pid != process.pid and _pid_is_running(pid):
            process.terminate()
            with suppress(NodeCliError):
                _wait_until_stopped(process.pid)
            _print_json(
                {
                    "data_dir": str(data_dir),
                    "listen": config.listen,
                    "network": config.network,
                    "pid": pid,
                    "running": True,
                }
            )
            return 0
        time.sleep(0.05)
    process.terminate()
    raise NodeCliError("node process did not report ready")


def _cmd_stop(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    _load_config(data_dir)
    pid = _read_pid(data_dir)
    was_running = pid is not None and _pid_is_running(pid)
    if was_running and pid is not None:
        os.kill(pid, signal.SIGTERM)
        _wait_until_stopped(pid)
    if pid is not None:
        _remove_pid_if_matches(data_dir, pid)
    else:
        _remove_pid(data_dir)
    _write_stopped_status(data_dir)
    _print_json({"data_dir": str(data_dir), "running": False, "was_running": was_running})
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    status = _status_payload(data_dir)
    _print_json(status)
    return 0


def _cmd_peers(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    _load_config(data_dir)
    peers = _read_peers(data_dir)
    _print_json({"data_dir": str(data_dir), "peer_count": len(peers), "peers": list(peers)})
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    config = _load_config(data_dir)
    pid = os.getpid()

    stop_event = Event()
    pid_acquired = False

    def _handle_signal(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _write_pid(data_dir, pid)
    pid_acquired = True
    _write_status(
        data_dir,
        {
            "listen": config.listen,
            "network": config.network,
            "pid": pid,
            "running": True,
            "started_at_ms": int(time.time() * 1000),
        },
    )
    _read_peers(data_dir)
    try:
        while not stop_event.wait(0.2):
            pass
    finally:
        if pid_acquired:
            _remove_pid_if_matches(data_dir, pid)
            _write_stopped_status(data_dir)
    return 0


@dataclass(frozen=True, slots=True)
class _NodeConfig:
    network: str
    listen: str


def _write_config(data_dir: Path, *, network: str, listen: str) -> None:
    _require_nonempty_str("network", network)
    _require_nonempty_str("listen", listen)
    text = "\n".join(
        (
            f"network = {json.dumps(network)}",
            f"listen = {json.dumps(listen)}",
            "",
        )
    )
    _config_path(data_dir).write_text(text, encoding="utf-8")


def _load_config(data_dir: Path) -> _NodeConfig:
    config_path = _config_path(data_dir)
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NodeCliError("node is not initialized") from exc
    except tomllib.TOMLDecodeError as exc:
        raise NodeCliError(f"node config is malformed: {exc}") from exc
    network = raw.get("network")
    listen = raw.get("listen")
    if not isinstance(network, str) or not network:
        raise NodeCliError("node config network must be a nonempty string")
    if not isinstance(listen, str) or not listen:
        raise NodeCliError("node config listen must be a nonempty string")
    return _NodeConfig(network=network, listen=listen)


def _status_payload(data_dir: Path) -> dict[str, object]:
    config = _load_config(data_dir)
    pid = _running_pid_or_clear_stale(data_dir)
    running = pid is not None
    if not running:
        _write_stopped_status(data_dir)
    status = _read_status(data_dir)
    return {
        "data_dir": str(data_dir),
        "listen": config.listen,
        "network": config.network,
        "peer_count": len(_read_peers(data_dir)),
        "pid": pid if running else None,
        "running": running,
        "started_at_ms": status.get("started_at_ms") if running else None,
    }


def _write_stopped_status(data_dir: Path) -> None:
    config = _load_config(data_dir)
    _write_status(
        data_dir,
        {
            "listen": config.listen,
            "network": config.network,
            "pid": None,
            "running": False,
            "started_at_ms": None,
        },
    )


def _read_status(data_dir: Path) -> dict[str, object]:
    path = data_dir / STATUS_FILENAME
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NodeCliError(f"node status is malformed: {exc}") from exc
    if not isinstance(raw, dict):
        raise NodeCliError("node status must be an object")
    return cast(dict[str, object], raw)


def _write_status(data_dir: Path, status: dict[str, object]) -> None:
    (data_dir / STATUS_FILENAME).write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_peers(data_dir: Path) -> tuple[str, ...]:
    path = data_dir / PEERS_FILENAME
    if not path.exists():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NodeCliError(f"peer file is malformed: {exc}") from exc
    if not isinstance(raw, list) or not all(isinstance(peer, str) for peer in raw):
        raise NodeCliError("peer file must contain a list of strings")
    return tuple(cast(list[str], raw))


def _write_peers(data_dir: Path, peers: tuple[str, ...]) -> None:
    (data_dir / PEERS_FILENAME).write_text(
        json.dumps(list(peers), indent=2) + "\n",
        encoding="utf-8",
    )


def _config_path(data_dir: Path) -> Path:
    return data_dir / CONFIG_FILENAME


def _pid_path(data_dir: Path) -> Path:
    return data_dir / PID_FILENAME


def _pid_lock_path(data_dir: Path) -> Path:
    return data_dir / PID_LOCK_FILENAME


@contextmanager
def _pid_file_lock(data_dir: Path) -> Iterator[None]:
    data_dir.mkdir(parents=True, exist_ok=True)
    with _pid_lock_path(data_dir).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _running_pid_or_clear_stale(data_dir: Path) -> int | None:
    with _pid_file_lock(data_dir):
        pid = _read_pid(data_dir)
        if pid is None:
            return None
        if _pid_is_running(pid):
            return pid
        _remove_pid(data_dir)
        return None


def _read_pid(data_dir: Path) -> int | None:
    path = _pid_path(data_dir)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise NodeCliError("node pid file is malformed") from exc
    if pid <= 0:
        raise NodeCliError("node pid must be positive")
    return pid


def _write_pid(data_dir: Path, pid: int) -> None:
    if pid <= 0:
        raise NodeCliError("node pid must be positive")
    with _pid_file_lock(data_dir):
        existing_pid = _read_pid(data_dir)
        if existing_pid is not None:
            if _pid_is_running(existing_pid):
                raise NodeCliError("node is already running")
            _remove_pid(data_dir)
        path = _pid_path(data_dir)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, f"{pid}\n".encode("ascii"))
        except OSError:
            with suppress(FileNotFoundError):
                path.unlink()
            raise
        finally:
            os.close(fd)


def _remove_pid(data_dir: Path) -> None:
    with suppress(FileNotFoundError):
        _pid_path(data_dir).unlink()


def _remove_pid_if_matches(data_dir: Path, pid: int) -> None:
    with _pid_file_lock(data_dir):
        current_pid = _read_pid(data_dir)
        if current_pid == pid:
            _remove_pid(data_dir)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until_stopped(pid: int) -> None:
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _reap_child(pid) or not _pid_is_running(pid):
            return
        time.sleep(0.05)
    raise NodeCliError("node process did not stop")


def _reap_child(pid: int) -> bool:
    try:
        waited_pid, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    return waited_pid == pid


def _require_nonempty_str(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str")
    if not value:
        raise NodeCliError(f"{name} must be nonempty")


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
