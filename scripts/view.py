#!/usr/bin/env python3
"""MIRA (SO-101 arm) interactive windowed viewer (requires X11 on the host)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Windowed viewer needs a GL window: use GLFW unless MUJOCO_GL is set.
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "").strip() or "glfw"
import mujoco.viewer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "so101"


def _targets(t: float, ctrl_range: np.ndarray) -> np.ndarray:
    lo, hi = ctrl_range[:, 0], ctrl_range[:, 1]
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    freqs = np.array([1.2, 0.8, 1.5, 1.0, 0.6, 0.3])
    phases = np.array([0.0, np.pi / 3, np.pi / 2, np.pi, 2.0, 0.0])
    out = mid + 0.6 * half * np.sin(freqs * t + phases)
    return np.clip(out, lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-time", type=float, default=None,
                    help="Auto-close after N s (default: run until window closed)")
    ap.add_argument("--manual", action="store_true",
                    help="Do not drive joints; drag them in the left Slider panel")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
    data = mujoco.MjData(model)
    ctrl_range = model.actuator_ctrlrange.copy()

    print(f"[view] Opening SO-101 viewer on DISPLAY={os.environ.get('DISPLAY', '?')}")
    if args.manual:
        print("[view] Manual mode: drag each actuator in the left 'Slider' panel")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        while viewer.is_running():
            if not args.manual:
                data.ctrl[:] = _targets(t, ctrl_range)
            mujoco.mj_step(model, data)
            viewer.sync()
            t += model.opt.timestep
            if args.max_time is not None and t >= args.max_time:
                break
    print("[view] Done. Window closed.")
    os._exit(0)


if __name__ == "__main__":
    main()