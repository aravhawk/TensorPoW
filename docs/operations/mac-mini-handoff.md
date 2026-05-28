# Mac Mini Handoff

This runbook is for the travel/off-LAN workflow: copy TensorPoW to the Mac mini
at home, let the Mac mini reach the CUDA PC on the LAN, and run the proof gates
from there.

## Copy From This Mac

From this Mac, build a clean transfer bundle:

```bash
mkdir -p dist
tar \
  --exclude='.git' \
  --exclude='final_check' \
  --exclude='data' \
  --exclude='tensorpow-data' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  --exclude='*.egg-info' \
  --exclude='dist' \
  -czf dist/tensorpow-handoff.tar.gz .
```

Copy the bundle to the Mac mini:

```bash
scp dist/tensorpow-handoff.tar.gz MAC_MINI_USER@MAC_MINI_HOST:~/tensorpow-handoff.tar.gz
```

On the Mac mini:

```bash
rm -rf ~/TensorPoW
mkdir -p ~/TensorPoW
tar -xzf ~/tensorpow-handoff.tar.gz -C ~/TensorPoW
cd ~/TensorPoW
```

## Quick Mac Mini + CUDA Verification

Use a conda Python. The default local path is `/opt/anaconda3/bin/python`; the
default CUDA-PC path is `/home/aravhawk/anaconda3/bin/python`.

```bash
/opt/anaconda3/bin/python -m pip install -e ".[dev]"
bash scripts/mac_mini_handoff.sh quick \
  --cuda-host aravhawk@CUDA_PC_LAN_IP \
  --remote-dir ~/tensorpow \
  --remote-python /home/aravhawk/anaconda3/bin/python
```

The `quick` command runs:

- Mac full pytest, ruff, format check, mypy, lock check, CPU determinism, and MPS
  determinism.
- `rsync` to the CUDA PC, excluding caches and local `final_check/` artifacts.
- CUDA-PC full pytest, ruff, format check, mypy, and CUDA determinism.
- Bounded 45-step node-backed smokes on Mac CPU/MPS and Ubuntu CPU/CUDA.
- A consensus-field comparison across all four smoke labels.

## Final Six-Hour Cross-Platform Soak

After `quick` passes, start the detached final soak:

```bash
bash scripts/mac_mini_handoff.sh launch-soak \
  --cuda-host aravhawk@CUDA_PC_LAN_IP \
  --remote-dir ~/tensorpow \
  --remote-python /home/aravhawk/anaconda3/bin/python
```

Later, check and compare reports:

```bash
bash scripts/mac_mini_handoff.sh check-soak \
  --cuda-host aravhawk@CUDA_PC_LAN_IP \
  --remote-dir ~/tensorpow \
  --remote-python /home/aravhawk/anaconda3/bin/python
```

`check-soak` only succeeds when both six-hour reports are complete, exit with
code `0`, and pass `scripts/soak_test.py --compare-report` with exact required
CPU/MPS/CPU/CUDA labels and matching consensus fields.

To cancel a detached soak:

```bash
bash scripts/mac_mini_handoff.sh cancel-soak \
  --cuda-host aravhawk@CUDA_PC_LAN_IP \
  --remote-dir ~/tensorpow \
  --remote-python /home/aravhawk/anaconda3/bin/python
```
