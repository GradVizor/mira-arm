#!/usr/bin/env python3
"""MIRA (SO-ARM100/SO-101) MuJoCo simulation with headless rendering.

Loads scene.xml (which includes so101_new_calib.xml + floor + lights),
drives all 6 actuators with smooth sinusoidal trajectories, renders frames,
and saves a GIF demo.

Must set MUJOCO_GL=osmesa BEFORE importing mujoco (CPU-only host, no GPU).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Pick a rendering backend before any MuJoCo import. An explicit MUJOCO_GL
# env var wins; otherwise auto-detect: EGL (hardware) when an NVIDIA GPU is
# visible inside the container, else osmesa (software) on CPU-only hosts.
def _pick_gl(default: str) -> str:
    explicit = os.environ.get("MUJOCO_GL", "").strip()
    if explicit:
        return explicit
    import shutil
    return "egl" if shutil.which("nvidia-smi") else default


os.environ["MUJOCO_GL"] = _pick_gl("osmesa")

import mujoco
import numpy as np

# ---------- constants ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "so101"
SCENE_XML = MODEL_DIR / "scene.xml"
RENDER_W, RENDER_H = 640, 480
FPS_RENDER = 30  # target render frame rate


def _sinusoidal_targets(
    t: float,
    ctrl_range: np.ndarray,
    freqs: list[float],
    phases: list[float],
    amp_frac: float = 0.6,
) -> np.ndarray:
    """Return smooth sinusoidal target for each actuator at time *t*.

    Each joint: mid + amp_frac * half_range * sin(freq * t + phase).
    Gripper (last joint) gets a slower cycle.
    """
    lo, hi = ctrl_range[:, 0], ctrl_range[:, 1]
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    targets = np.empty(len(freqs), dtype=np.float64)
    for i in range(len(freqs)):
        targets[i] = mid[i] + amp_frac * half[i] * np.sin(freqs[i] * t + phases[i])
    return np.clip(targets, lo, hi)


def _setup_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """Create a free camera that frames the ~0.35 m tall arm at the origin."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Look at the middle of the arm (approx z=0.18 m above base).
    cam.lookat[:] = [0.0, 0.0, 0.18]
    # Position camera: slightly above, pulled back, azimuth ~135°.
    cam.distance = 0.55
    cam.azimuth = 135.0
    cam.elevation = 25.0
    return cam


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out) if args.out else REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir  # PNGs go directly in out_dir

    print(f"[sim] Loading scene from {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    total_steps = int(args.duration / dt)
    render_interval = max(1, int(1.0 / (FPS_RENDER * dt)))

    print(f"[sim] Model: nu={model.nu} nq={model.nq} nv={model.nv} dt={dt} "
          f"duration={args.duration}s total_steps={total_steps} "
          f"render_interval={render_interval}")

    # Actuator names & ctrl ranges
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(model.nu)]
    ctrl_range = model.actuator_ctrlrange.copy()

    # Per-joint frequency (rad/s) and phase — distinct per joint.
    # Gripper (index 5) uses a slower frequency for open/close cycle.
    freqs = [1.2, 0.8, 1.5, 1.0, 0.6, 0.3]
    phases = [0.0, np.pi / 3, np.pi / 2, np.pi, 2.0, 0.0]

    # Gripper site id for end-effector tracking
    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                         "gripperframe")
    if gripper_site_id < 0:
        print("[sim] WARNING: gripperframe site not found, reach tracking disabled")
        gripper_site_id = -1

    # Renderer + camera
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    cam = _setup_camera(model)

    # ---- simulation loop ----
    frame_idx = 0
    reach_positions: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    milestone_frames: dict[str, np.ndarray] = {}
    t0 = time.time()

    for step in range(total_steps):
        t = step * dt

        # Set control targets
        targets = _sinusoidal_targets(t, ctrl_range, freqs, phases)
        data.ctrl[:model.nu] = targets

        # Step physics
        mujoco.mj_step(model, data)

        # Collect gripper reach data every 50 steps
        if gripper_site_id >= 0 and step % 50 == 0:
            mujoco.mj_forward(model, data)
            reach_positions.append(data.site_xpos[gripper_site_id].copy())

        # Render at target FPS intervals
        if step % render_interval == 0:
            renderer.update_scene(data, cam)
            rgb = renderer.render()
            frames.append(rgb.copy())

            # Milestone frames
            t_frac = step / total_steps
            if frame_idx == 0:
                milestone_frames["pose_start"] = rgb.copy()
            if 0.45 <= t_frac <= 0.55 and "pose_mid" not in milestone_frames:
                milestone_frames["pose_mid"] = rgb.copy()
            if step >= total_steps - render_interval:
                milestone_frames["pose_end"] = rgb.copy()

            frame_idx += 1

    sim_time = time.time() - t0
    speed = total_steps / sim_time

    # ---- save frames as PNGs ----
    try:
        from PIL import Image
    except ImportError:
        print("[sim] ERROR: Pillow not available, cannot save PNGs")
        sys.exit(1)

    print(f"[sim] Saving {len(frames)} rendered frames ...")
    for i, rgb in enumerate(frames):
        img = Image.fromarray(rgb)
        img.save(str(frames_dir / f"frame_{i:04d}.png"))

    # Save milestone PNGs
    for name, rgb in milestone_frames.items():
        img = Image.fromarray(rgb)
        img.save(str(frames_dir / f"{name}.png"))
        print(f"[sim] Saved milestone: {frames_dir / f'{name}.png'}")

    # ---- build GIF ----
    if not args.no_gif and len(frames) > 1:
        gif_path = frames_dir / "output_demo.gif"
        print(f"[sim] Building GIF: {gif_path}")
        pil_frames = [Image.fromarray(rgb) for rgb in frames]
        duration_ms = int(1000 / FPS_RENDER)
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"[sim] GIF saved: {gif_path} ({len(frames)} frames)")

    # ---- diagnostics ----
    reach_arr = np.array(reach_positions) if reach_positions else np.zeros((0, 3))
    if reach_arr.size:
        reach_min = reach_arr.min(axis=0)
        reach_max = reach_arr.max(axis=0)
    else:
        reach_min = reach_max = np.zeros(3)

    # Final joint angles
    final_qpos = data.qpos.copy()
    final_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.nq)]

    # Verify a rendered frame is not blank (guard: 0 rendered frames -> IndexError)
    if frames:
        check_img = Image.fromarray(frames[len(frames) // 2])
        mean_px = np.mean(np.array(check_img))
    else:
        mean_px = 0.0

    print("\n" + "=" * 60)
    print("SUCCESS — SO-ARM100 simulation complete")
    print("=" * 60)
    print(f"  Output dir    : {out_dir}")
    print(f"  Frames (PNG)  : {len(frames)} files in {frames_dir}")
    print(f"  GIF           : {out_dir / 'output_demo.gif' if not args.no_gif else 'skipped'}")
    print(f"  Milestones    : {list(milestone_frames.keys())}")
    print(f"  Frame check   : mid-frame mean pixel = {mean_px:.1f} "
          f"({'non-blank' if 10 < mean_px < 245 else 'WARNING: possibly blank'})")
    print(f"  SPEED         : {total_steps} steps in {sim_time:.2f}s -> {speed:.0f} Hz")
    print(f"  Model stats   : nu={model.nu} nq={model.nq} dt={dt}")
    print(f"  Actuators     : {act_names}")
    print(f"  ctrl ranges   :")
    for i, name in enumerate(act_names):
        print(f"    {name}: [{ctrl_range[i, 0]:.5f}, {ctrl_range[i, 1]:.5f}]")
    print(f"  Final joints  :")
    for i, name in enumerate(final_names):
        print(f"    {name}: {final_qpos[i]:.5f} rad")
    print(f"  Gripper reach : min={reach_min} max={reach_max}")
    print(f"  Sinusoidal freqs (rad/s): {freqs}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SO-ARM100 MuJoCo simulation with headless rendering")
    parser.add_argument("--duration", type=float, default=6.0,
                        help="Simulation duration in seconds (default: 6)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: <repo_root>/outputs)")
    parser.add_argument("--no-gif", action="store_true",
                        help="Skip GIF generation")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()