#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

echo "=============================="
echo "  MIRA arm — SO-101 container"
echo "=============================="

# Renderer auto-selection lives in the scripts, not here:
#   scripts/sim.py::_pick_gl  -> explicit MUJOCO_GL, else egl if NVIDIA GPU, else osmesa
#   scripts/view.py           -> glfw window unless MUJOCO_GL is set explicitly
# We only report the GPU state for diagnostics; exporting MUJOCO_GL here would
# force view.py off its glfw window.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[gpu] none -> CPU mode (sim.py: osmesa, view.py: glfw)"
fi

# Optional virtual display so the interactive viewer can run on headless hosts.
if [ "${XVFB_RUN:-0}" = "1" ]; then
    rm -f /tmp/.X99-lock
    Xvfb :99 -screen 0 1920x1080x24 >/tmp/xvfb.log 2>&1 &
    export DISPLAY=:99
fi

# Keep files created through the bind mount owned by the host user.
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    chown -R "${HOST_UID}:${HOST_GID}" /workspace 2>/dev/null || true
fi

# Smoke-check the stack before handing over to the shell.
python - <<'PY'
import mujoco, numpy

print(f"[mujoco] version {mujoco.__version__}")
print(f"[numpy]  version {numpy.__version__}")
PY

exec "$@"