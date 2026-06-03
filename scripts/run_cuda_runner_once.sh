#!/usr/bin/env bash
set -euo pipefail

# Usage: gh auth login, then run this script on the CUDA box when the GPU is idle.
# Optional overrides: REPO, RUNNER_ROOT, RUNNER_LABELS, MAX_GPU_UTIL, MAX_GPU_MEM_MB.
REPO="${REPO:-aravhawk/TensorPoW}"
RUNNER_ROOT="${RUNNER_ROOT:-$HOME/actions-runners/tensorpow-cuda}"
RUNNER_LABELS="${RUNNER_LABELS:-cuda,tensorpow-cuda}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-15}"
MAX_GPU_MEM_MB="${MAX_GPU_MEM_MB:-1024}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need gh
need curl
need tar
need python3
need nvidia-smi

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
  echo "Do not run the GitHub Actions runner as root." >&2
  exit 1
fi

while IFS=, read -r util mem; do
  util="${util//[[:space:]]/}"
  mem="${mem//[[:space:]]/}"
  if [ "$util" -gt "$MAX_GPU_UTIL" ] || [ "$mem" -gt "$MAX_GPU_MEM_MB" ]; then
    echo "GPU is busy: utilization=${util}%, memory=${mem}MiB." >&2
    echo "Thresholds: MAX_GPU_UTIL=${MAX_GPU_UTIL}%, MAX_GPU_MEM_MB=${MAX_GPU_MEM_MB}MiB." >&2
    exit 1
  fi
done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)

mkdir -p "$RUNNER_ROOT"
cd "$RUNNER_ROOT"

if [ ! -x ./config.sh ] || [ ! -x ./run.sh ]; then
  echo "Downloading latest GitHub Actions runner for Linux x64..."
  asset_url="$(
    python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("https://api.github.com/repos/actions/runner/releases/latest") as response:
    release = json.load(response)

for asset in release["assets"]:
    name = asset["name"]
    if name.startswith("actions-runner-linux-x64-") and name.endswith(".tar.gz"):
        print(asset["browser_download_url"])
        break
else:
    raise SystemExit("No linux-x64 runner asset found")
PY
  )"
  curl -fsSL "$asset_url" -o actions-runner-linux-x64.tar.gz
  tar xzf actions-runner-linux-x64.tar.gz
  rm -f actions-runner-linux-x64.tar.gz
fi

cleanup() {
  if [ -x ./config.sh ] && [ -f .runner ]; then
    remove_token="$(gh api -X POST "repos/$REPO/actions/runners/remove-token" --jq .token 2>/dev/null || true)"
    if [ -n "${remove_token:-}" ]; then
      ./config.sh remove --token "$remove_token" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

registration_token="$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)"

./config.sh \
  --unattended \
  --ephemeral \
  --url "https://github.com/$REPO" \
  --token "$registration_token" \
  --name "tensorpow-cuda-$(hostname)-$$" \
  --labels "$RUNNER_LABELS" \
  --replace

echo "Runner registered for $REPO with labels: self-hosted, linux, x64, $RUNNER_LABELS"
echo "Waiting for one CUDA job. Press Ctrl-C to stop before a job is assigned."
./run.sh
