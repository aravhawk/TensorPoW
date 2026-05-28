#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
TensorPoW Mac mini handoff runner

Usage:
  scripts/mac_mini_handoff.sh quick --cuda-host USER@HOST [options]
  scripts/mac_mini_handoff.sh launch-soak --cuda-host USER@HOST [options]
  scripts/mac_mini_handoff.sh check-soak --cuda-host USER@HOST [options]
  scripts/mac_mini_handoff.sh cancel-soak --cuda-host USER@HOST [options]

Options:
  --cuda-host HOST        SSH target for the CUDA PC, e.g. aravhawk@192.168.1.50
  --remote-dir DIR        Remote TensorPoW checkout directory (default: ~/tensorpow)
  --python PATH           Local conda Python (default: /opt/anaconda3/bin/python)
  --remote-python PATH    Remote conda Python (default: /home/aravhawk/anaconda3/bin/python)
  --run-id ID             Artifact id (default: UTC timestamp)
  --skip-install          Do not run pip install -e ".[dev]" locally/remotely
  --skip-uv-lock          Skip uv lock --check when uv is not installed
  -h, --help              Show this help

Commands:
  quick
    Run Mac local gates, sync to CUDA PC, run Ubuntu/CUDA gates, run bounded
    CPU/MPS and CPU/CUDA node-backed smokes, and compare consensus fields.

  launch-soak
    Sync to CUDA PC and launch the final six-hour Mac CPU/MPS + Ubuntu
    CPU/CUDA soak. This is detached and must be checked later.

  check-soak
    Inspect the detached six-hour soak, copy the remote report if complete,
    and run the strict final report comparator.

  cancel-soak
    Stop any detached six-hour soak and archive partial artifacts on both hosts.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '\n==> %s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

ensure_no_whitespace() {
  local label="$1"
  local value="$2"
  if [[ "$value" =~ [[:space:]] ]]; then
    fail "$label must not contain whitespace: $value"
  fi
}

ssh_run() {
  local script="$1"
  ssh "$CUDA_HOST" "bash -lc $(printf '%q' "$script")"
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --cuda-host)
        [[ $# -ge 2 && "${2:-}" != --* ]] || fail "--cuda-host requires a value"
        CUDA_HOST="${2:-}"
        shift 2
        ;;
      --remote-dir)
        [[ $# -ge 2 && "${2:-}" != --* ]] || fail "--remote-dir requires a value"
        REMOTE_DIR="${2:-}"
        shift 2
        ;;
      --python)
        [[ $# -ge 2 && "${2:-}" != --* ]] || fail "--python requires a value"
        PYTHON="${2:-}"
        shift 2
        ;;
      --remote-python)
        [[ $# -ge 2 && "${2:-}" != --* ]] || fail "--remote-python requires a value"
        REMOTE_PYTHON="${2:-}"
        shift 2
        ;;
      --run-id)
        [[ $# -ge 2 && "${2:-}" != --* ]] || fail "--run-id requires a value"
        RUN_ID="${2:-}"
        shift 2
        ;;
      --skip-install)
        INSTALL=0
        shift
        ;;
      --skip-uv-lock)
        RUN_UV_LOCK=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        fail "unknown argument: $1"
        ;;
    esac
  done
}

require_cuda_args() {
  [[ -n "$CUDA_HOST" ]] || fail "--cuda-host is required for $COMMAND"
  ensure_no_whitespace "--cuda-host" "$CUDA_HOST"
  ensure_no_whitespace "--remote-dir" "$REMOTE_DIR"
  ensure_no_whitespace "--remote-python" "$REMOTE_PYTHON"
}

local_install() {
  if [[ "$INSTALL" -eq 1 ]]; then
    log "Installing local editable package in the active conda environment"
    run_cmd "$PYTHON" -m pip install -e ".[dev]"
  fi
}

local_gates() {
  log "Running Mac local gates"
  local_install
  run_cmd env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
    "$PYTHON" -m pytest -q -p no:cacheprovider
  run_cmd "$PYTHON" -m ruff check .
  run_cmd "$PYTHON" -m ruff format --check .
  run_cmd "$PYTHON" -m mypy src scripts
  if [[ "$RUN_UV_LOCK" -eq 1 ]]; then
    require_command uv
    run_cmd uv lock --check
  fi
  run_cmd env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. TENSORPOW_DETERMINISM_BACKEND=cpu \
    "$PYTHON" -m pytest -q -p no:cacheprovider tests/determinism
  run_cmd env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. TENSORPOW_DETERMINISM_BACKEND=mps \
    "$PYTHON" -m pytest -q -p no:cacheprovider tests/determinism
}

local_bounded_smoke() {
  log "Running Mac CPU/MPS bounded node-backed smoke"
  mkdir -p "final_check/handoff-$RUN_ID"
  rm -rf "final_check/handoff-$RUN_ID/mac-smoke-data"
  run_cmd env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
    "$PYTHON" scripts/soak_test.py \
    --duration-seconds 0 \
    --max-steps 45 \
    --platform mac-arm-cpu:cpu \
    --platform mac-arm-mps:mps \
    --data-dir "final_check/handoff-$RUN_ID/mac-smoke-data" \
    --report "final_check/handoff-$RUN_ID/mac-smoke.json" \
    --json
}

sync_to_cuda() {
  require_command rsync
  require_cuda_args
  log "Syncing repo to CUDA PC: $CUDA_HOST:$REMOTE_DIR"
  ssh_run "mkdir -p $REMOTE_DIR"
  run_cmd rsync -az --delete \
    --exclude ".git/" \
    --exclude ".mypy_cache/" \
    --exclude ".pytest_cache/" \
    --exclude ".ruff_cache/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "*.egg-info/" \
    --exclude "build/" \
    --exclude "dist/" \
    --exclude "data/" \
    --exclude "final_check/" \
    --exclude "tensorpow-data/" \
    ./ "$CUDA_HOST:$REMOTE_DIR/"
}

remote_gates() {
  require_cuda_args
  log "Running Ubuntu/CUDA gates"
  ssh_run "set -euo pipefail
cd $REMOTE_DIR
if [[ $INSTALL -eq 1 ]]; then
  $REMOTE_PYTHON -m pip install -e '.[dev]'
fi
$REMOTE_PYTHON - <<'PY'
import tensorpow
print(tensorpow.__file__)
assert '/tensorpow/src/tensorpow/__init__.py' in tensorpow.__file__
PY
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. $REMOTE_PYTHON -m pytest -q -p no:cacheprovider
$REMOTE_PYTHON -m ruff check .
$REMOTE_PYTHON -m ruff format --check .
$REMOTE_PYTHON -m mypy src scripts
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. TENSORPOW_DETERMINISM_BACKEND=cuda \
  $REMOTE_PYTHON -m pytest -q -p no:cacheprovider tests/determinism
"
}

remote_bounded_smoke() {
  require_cuda_args
  log "Running Ubuntu CPU/CUDA bounded node-backed smoke"
  ssh_run "set -euo pipefail
cd $REMOTE_DIR
mkdir -p final_check/handoff-$RUN_ID
rm -rf final_check/handoff-$RUN_ID/ubuntu-smoke-data
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  $REMOTE_PYTHON scripts/soak_test.py \
  --duration-seconds 0 \
  --max-steps 45 \
  --platform ubuntu-x86-cpu:cpu \
  --platform ubuntu-x86-cuda:cuda \
  --data-dir final_check/handoff-$RUN_ID/ubuntu-smoke-data \
  --report final_check/handoff-$RUN_ID/ubuntu-smoke.json \
  --json
"
  mkdir -p "final_check/handoff-$RUN_ID"
  run_cmd rsync -az \
    "$CUDA_HOST:$REMOTE_DIR/final_check/handoff-$RUN_ID/ubuntu-smoke.json" \
    "final_check/handoff-$RUN_ID/ubuntu-smoke.json"
}

compare_bounded_smokes() {
  log "Comparing bounded Mac and Ubuntu smoke reports"
  run_cmd "$PYTHON" - "$RUN_ID" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_id = sys.argv[1]
base = Path("final_check") / f"handoff-{run_id}"
paths = (base / "mac-smoke.json", base / "ubuntu-smoke.json")
expected_backends = {
    "mac-arm-cpu": "cpu",
    "mac-arm-mps": "mps",
    "ubuntu-x86-cpu": "cpu",
    "ubuntu-x86-cuda": "cuda",
}
fields = (
    "final_state_root",
    "utxo_outputs_digest",
    "consensus_bytes_digest",
    "utxo_count",
    "tx_count",
    "fruit_count",
    "anchor_count",
)
results: list[dict[str, object]] = []
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["comparison"]["match"] is not True:
        raise SystemExit(f"{path} did not contain a matching intra-host comparison")
    results.extend(payload["results"])

labels = {str(item["label"]) for item in results}
if labels != set(expected_backends):
    raise SystemExit(f"unexpected labels: {sorted(labels)}")

baseline = results[0]
mismatches: list[dict[str, object]] = []
for item in results:
    label = str(item["label"])
    expected_backend = expected_backends[label]
    if item["resolved_backend"] != expected_backend:
        raise SystemExit(f"{label} resolved_backend={item['resolved_backend']!r}")
    if item["steps"] != 45:
        raise SystemExit(f"{label} steps={item['steps']!r}, expected 45")
    for field in fields:
        if item[field] != baseline[field]:
            mismatches.append(
                {
                    "baseline": baseline["label"],
                    "baseline_value": baseline[field],
                    "field": field,
                    "label": label,
                    "value": item[field],
                }
            )
if mismatches:
    raise SystemExit(json.dumps(mismatches, sort_keys=True))

print(
    json.dumps(
        {
            "labels": sorted(labels),
            "match": True,
            "compared_fields": fields,
            "final_state_root": baseline["final_state_root"],
            "utxo_outputs_digest": baseline["utxo_outputs_digest"],
            "consensus_bytes_digest": baseline["consensus_bytes_digest"],
            "utxo_count": baseline["utxo_count"],
            "tx_count": baseline["tx_count"],
            "fruit_count": baseline["fruit_count"],
            "anchor_count": baseline["anchor_count"],
        },
        sort_keys=True,
    )
)
PY
}

quick() {
  require_cuda_args
  local_gates
  local_bounded_smoke
  sync_to_cuda
  remote_gates
  remote_bounded_smoke
  compare_bounded_smokes
  log "Quick Mac mini + CUDA handoff verification passed"
}

launch_soak() {
  require_command tmux
  require_cuda_args
  sync_to_cuda
  log "Launching final six-hour soak on Mac mini and CUDA PC"
  if tmux has-session -t tensorpow-soak-mac 2>/dev/null; then
    fail "tmux session tensorpow-soak-mac already exists"
  fi
  if ps -eo command | awk '/[p]ython scripts\/soak_test.py --duration-hours 6/ {found=1} END {exit found ? 0 : 1}'; then
    fail "local six-hour soak worker already exists"
  fi
  ssh_run "if ps -eo args | awk '/[p]ython scripts\\/soak_test.py --duration-hours 6/ {found=1} END {exit found ? 0 : 1}'; then exit 42; fi" \
    || fail "remote six-hour soak worker already exists"

  rm -f final_check/soak-6h-mac.pid final_check/soak-6h-mac.log \
    final_check/soak-6h-mac.exit final_check/soak-6h-mac.json
  rm -rf final_check/soak-6h-mac-data
  tmux new-session -d -s tensorpow-soak-mac \
    "cd '$REPO_DIR'; echo \\\$\\\$ > final_check/soak-6h-mac.pid; \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src '$PYTHON' scripts/soak_test.py \
--duration-hours 6 --platform mac-arm-cpu:cpu --platform mac-arm-mps:mps \
--data-dir final_check/soak-6h-mac-data --report final_check/soak-6h-mac.json \
--json > final_check/soak-6h-mac.log 2>&1; echo \\\$? > final_check/soak-6h-mac.exit"

  ssh_run "set -euo pipefail
cd $REMOTE_DIR
rm -f final_check/soak-6h.pid final_check/soak-6h.log final_check/soak-6h.exit final_check/soak-6h.json
rm -rf final_check/soak-6h-data
nohup bash -lc 'echo \$\$ > final_check/soak-6h.pid; PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 $REMOTE_PYTHON scripts/soak_test.py --duration-hours 6 --platform ubuntu-x86-cpu:cpu --platform ubuntu-x86-cuda:cuda --data-dir final_check/soak-6h-data --report final_check/soak-6h.json --json > final_check/soak-6h.log 2>&1; echo \$? > final_check/soak-6h.exit' >/dev/null 2>&1 &
"
  log "Six-hour soak launched. Later run: scripts/mac_mini_handoff.sh check-soak --cuda-host $CUDA_HOST"
}

check_soak() {
  require_cuda_args
  log "Checking six-hour soak reports"
  local local_exit=""
  local remote_exit=""
  [[ -f final_check/soak-6h-mac.exit ]] && local_exit="$(cat final_check/soak-6h-mac.exit)"
  remote_exit="$(ssh "$CUDA_HOST" "cd $REMOTE_DIR && cat final_check/soak-6h.exit 2>/dev/null || true")"

  if [[ -z "$local_exit" || -z "$remote_exit" ]]; then
    printf 'local status:\n'
    ps -eo pid,etime,time,%cpu,%mem,command | awk '/[p]ython scripts\/soak_test.py --duration-hours 6|[t]ensorpow-soak-mac/ {print}' || true
    ls -l final_check/soak-6h-mac.exit final_check/soak-6h-mac.json 2>&1 || true
    printf 'remote status:\n'
    ssh "$CUDA_HOST" "cd $REMOTE_DIR && ps -eo pid,etime,time,%cpu,%mem,args | awk '/[p]ython scripts\\/soak_test.py --duration-hours 6/ {print}'; ls -l final_check/soak-6h.exit final_check/soak-6h.json 2>&1 || true"
    fail "six-hour soak is not complete on both hosts yet"
  fi
  [[ "$local_exit" == "0" ]] || fail "local six-hour soak exit code: $local_exit"
  [[ "$remote_exit" == "0" ]] || fail "remote six-hour soak exit code: $remote_exit"

  run_cmd rsync -az "$CUDA_HOST:$REMOTE_DIR/final_check/soak-6h.json" \
    final_check/soak-6h-ubuntu.json
  run_cmd env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" scripts/soak_test.py \
    --compare-report final_check/soak-6h-mac.json \
    --compare-report final_check/soak-6h-ubuntu.json \
    --json
}

cancel_soak() {
  require_cuda_args
  local cancel_id="cancelled-soak-$(date -u +%Y%m%dT%H%M%SZ)"
  log "Cancelling six-hour soak and archiving artifacts as $cancel_id"
  mkdir -p "final_check/$cancel_id"
  ps -eo pid,ppid,etime,time,%cpu,%mem,command \
    | awk '/[p]ython scripts\/soak_test.py --duration-hours 6|[t]ensorpow-soak-mac|[s]oak-6h-mac/ {print}' \
    > "final_check/$cancel_id/ps-before-stop.txt"
  local worker_pids
  worker_pids="$(ps -eo pid,command | awk '/[p]ython scripts\/soak_test.py --duration-hours 6/ {print $1}')"
  if [[ -n "$worker_pids" ]]; then
    kill $worker_pids 2>/dev/null || true
  fi
  if tmux has-session -t tensorpow-soak-mac 2>/dev/null; then
    tmux kill-session -t tensorpow-soak-mac
  fi
  for item in final_check/soak-6h-mac.pid final_check/soak-6h-mac.log \
    final_check/soak-6h-mac.exit final_check/soak-6h-mac.json final_check/soak-6h-mac-data; do
    [[ -e "$item" ]] && mv "$item" "final_check/$cancel_id/"
  done

  ssh_run "set -euo pipefail
cd $REMOTE_DIR
archive=final_check/$cancel_id
mkdir -p \"\$archive\"
ps -eo pid,ppid,etime,time,%cpu,%mem,args | awk '/[p]ython scripts\\/soak_test.py --duration-hours 6|[s]oak-6h/ {print}' > \"\$archive/ps-before-stop.txt\"
pids=\$(ps -eo pid,args | awk '/[p]ython scripts\\/soak_test.py --duration-hours 6|[b]ash -lc echo \\\$\\\$ > final_check\\/soak-6h.pid/ {print \$1}')
if [[ -n \"\$pids\" ]]; then kill \$pids 2>/dev/null || true; fi
for item in final_check/soak-6h.pid final_check/soak-6h.log final_check/soak-6h.exit final_check/soak-6h.json final_check/soak-6h-data; do
  [[ -e \"\$item\" ]] && mv \"\$item\" \"\$archive/\"
done
"
}

COMMAND="${1:-}"
[[ -n "$COMMAND" ]] || { usage; exit 2; }
shift || true

REPO_DIR="$(repo_root)"
cd "$REPO_DIR"

CUDA_HOST=""
REMOTE_DIR="${TENSORPOW_REMOTE_DIR:-~/tensorpow}"
PYTHON="${TENSORPOW_PYTHON:-/opt/anaconda3/bin/python}"
REMOTE_PYTHON="${TENSORPOW_REMOTE_PYTHON:-/home/aravhawk/anaconda3/bin/python}"
RUN_ID="${TENSORPOW_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
INSTALL=1
RUN_UV_LOCK=1

case "$COMMAND" in
  quick|launch-soak|check-soak|cancel-soak)
    parse_common_args "$@"
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    fail "unknown command: $COMMAND"
    ;;
esac

ensure_no_whitespace "--python" "$PYTHON"
[[ -x "$PYTHON" ]] || fail "local Python is not executable: $PYTHON"
require_command ssh

case "$COMMAND" in
  quick)
    quick
    ;;
  launch-soak)
    launch_soak
    ;;
  check-soak)
    check_soak
    ;;
  cancel-soak)
    cancel_soak
    ;;
esac
