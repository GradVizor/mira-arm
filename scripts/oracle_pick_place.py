#!/usr/bin/env python3
"""Scripted oracle for the SO-101 pick-and-place task.

Grasps the marshmallow, lifts it, transports it over the coffee cup, releases
it above the rim, and lets it drop into the cup.  Success must come from
``env.step()`` -- the script never writes ``qpos`` or the success flag.

This is the hand-tuned IK solution, **not** the trained SAC policy.  Use
``--viewer`` to watch it drive the arm in a live MuJoCo window.

Run::

    python3 scripts/oracle_pick_place.py
    python3 scripts/oracle_pick_place.py --seed 0 --verbose
    python3 scripts/oracle_pick_place.py --viewer            # live window
    python3 scripts/oracle_pick_place.py --gif               # save an animated GIF
    python3 scripts/oracle_pick_place.py --out /tmp/oracle_summary.json

Exit code is 0 only when the environment reports ``is_success=True`` at least
once, otherwise 1.  ``--viewer`` requires a display; without it no DISPLAY is
needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mira.envs.so101_pick_place_env import SO101PickPlaceEnv  # noqa: E402
from mira.kinematics import DOWN, PAD_MIDPOINT_Z, Kinematics  # noqa: E402

# The task describes a ~250/150/100/150/250/150/100/150 plan (1300 steps), but
# the environment truncates at max_episode_steps=700.  These values are the
# retuned, budget-fitting ramps actually used (see the final report).
PHASE_STEPS = {
    "approach": 120,
    "descend": 70,
    "close": 60,
    "lift": 70,
    "transport": 140,
    "lower": 80,
    "release": 70,
    "settle": 60,
}

GRASP_HEIGHT = PAD_MIDPOINT_Z  # pad midpoint lands at the box centre
PRE_HEIGHT = 0.09  # pre-grasp, 9 cm above the grasp pose
LIFT_HEIGHT = 0.075  # lift, 7.5 cm above the grasp pose
CUP_HIGH_Z = 0.11  # transport waypoint; 0.13 is unreachable with world-down IK
CUP_LOW_Z = 0.095  # release height; cup rim is z ~= 0.078
IK_POS_TOL = 0.01  # metres; refuse to execute a worse solution

GIF_PATH = REPO_ROOT / "outputs" / "oracle_pick_place.gif"
GIF_TOP_PATH = REPO_ROOT / "outputs" / "oracle_pick_place_top.gif"
GIF_FPS = 30
GIF_SAMPLE_EVERY = 3
GIF_W, GIF_H = 480, 360


def _joint_mid(env: SO101PickPlaceEnv) -> np.ndarray:
    lo = env.model.jnt_range[env.arm_joint_ids, 0]
    hi = env.model.jnt_range[env.arm_joint_ids, 1]
    mid = 0.5 * (lo + hi)
    mid[5] = env.open_q
    return mid


def _solve_with_offset(
    kin: Kinematics,
    site_offset: np.ndarray,
    desired: np.ndarray,
    q_init: np.ndarray | None,
    iters: int = 10,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Solve site pose so the pad midpoint lands at ``desired``.

    Iteratively corrects ``target = desired - rot @ site_offset`` because the
    pad midpoint offset rotates with the wrist.  Returns the joint solution,
    the position/direction error against the final site target, and the target.
    """
    target = desired.copy()
    q = None if q_init is None else np.asarray(q_init, dtype=np.float64).copy()
    for _ in range(iters):
        q = kin.solve(target, q_init=q)
        rot = kin.data.xmat[kin.gripper_id].reshape(3, 3)
        next_target = desired - rot @ site_offset
        if np.linalg.norm(next_target - target) < 1e-6:
            target = next_target
            break
        target = next_target
    q = kin.solve(target, q_init=q)
    pos_error, dir_error = kin.pose_error(target, q)
    return q, pos_error, dir_error, target


def _extra_guesses(mid: np.ndarray, seeds: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    """Additional IK starts used when the base multi-start is still bad."""
    guesses: list[np.ndarray] = []
    for wrist_roll in (-1.5, 1.5, -2.0, 2.0, 0.0):
        q = mid.copy()
        q[4] = wrist_roll
        guesses.append(q)
    for seed in seeds:
        for wrist_roll in (-1.0, 1.0):
            q = np.asarray(seed, dtype=np.float64).copy()
            q[4] = wrist_roll
            guesses.append(q)
    return guesses


def _multi_start(
    env: SO101PickPlaceEnv,
    kin: Kinematics,
    desired: np.ndarray,
    guesses: list[np.ndarray | None],
    mid: np.ndarray,
    label: str,
) -> tuple[np.ndarray, float, float, float]:
    """Pick the lowest ``pos_error + dir_error`` IK solution.

    If the best position error exceeds ``IK_POS_TOL`` extra starts are tried
    before failing loudly (a bad pose is never executed silently).
    """
    best_q: np.ndarray | None = None
    best_score = np.inf
    best_pos = np.inf
    best_dir = np.inf

    def consider(q_init: np.ndarray | None) -> None:
        nonlocal best_q, best_score, best_pos, best_dir
        q, pos_error, dir_error, _ = _solve_with_offset(
            kin, env.pads.site_offset, desired, q_init
        )
        score = pos_error + dir_error
        if score < best_score:
            best_q = q
            best_score = score
            best_pos = pos_error
            best_dir = dir_error

    for guess in guesses:
        consider(guess)

    if best_pos > IK_POS_TOL:
        print(
            f"[ik] {label}: base best position error {best_pos * 100:.3f} cm "
            f"> {IK_POS_TOL * 100:.1f} cm; trying extra starts"
        )
        seeds = tuple(g for g in guesses if g is not None)
        for guess in _extra_guesses(mid, seeds):
            consider(guess)

    if best_q is None or best_pos > IK_POS_TOL:
        raise RuntimeError(
            f"IK failed for {label}: best position error {best_pos:.4f} m "
            f"(dir {best_dir:.4f}) exceeds {IK_POS_TOL:.4f} m"
        )

    # Pad-midpoint error against the requested box centre (the real metric).
    kin.fk(best_q)
    rot = kin.data.xmat[kin.gripper_id].reshape(3, 3)
    pad_mid = kin.data.site_xpos[kin.site_id] + rot @ env.pads.site_offset
    pad_error = float(np.linalg.norm(pad_mid - desired))
    return best_q, best_pos, best_dir, pad_error


def _to_action(env: SO101PickPlaceEnv, q: np.ndarray, grip: float) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    q[5] = grip
    q = np.clip(q, env.ctrl_low, env.ctrl_high)
    action = 2.0 * (q - env.ctrl_low) / (env.ctrl_high - env.ctrl_low) - 1.0
    # Match SO101PickPlaceEnv.step: the gripper action spans
    # [close_q, open_q], not the full joint range.
    action[5] = (
        2.0 * (q[5] - env.close_q) / (env.open_q - env.close_q) - 1.0
    )
    return action.astype(np.float32)


def _update_stats(stats: dict, info: dict) -> None:
    stats["ever_grasped"] = bool(info["ever_grasped"])
    stats["max_box_z"] = max(stats["max_box_z"], float(info["box_pos"][2]))
    stats["success_seen"] = stats["success_seen"] or bool(info["is_success"])
    stats["box_pos"] = np.asarray(info["box_pos"], dtype=np.float64).copy()
    stats["cup_pos"] = np.asarray(info["cup_pos"], dtype=np.float64).copy()
    stats["box_cup_dist_xy"] = float(info["box_cup_dist_xy"])
    stats["cup_upright"] = bool(info["cup_upright"])
    stats["grasped"] = bool(info["grasped"])


def _setup_camera(lookat: np.ndarray, top_view: bool = False) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = 0.45 if top_view else 0.60
    cam.azimuth = 90.0 if top_view else 135.0
    cam.elevation = 90.0 if top_view else 25.0
    return cam


class _GifRecorder:
    """Mutable frame accumulator for the headless GIF capture."""

    def __init__(
        self, model: mujoco.MjModel, lookat: np.ndarray, top_view: bool = False
    ) -> None:
        self.renderer = mujoco.Renderer(model, height=GIF_H, width=GIF_W)
        self.camera = _setup_camera(lookat, top_view)
        self.frames: list[np.ndarray] = []
        self.step = 0

    def capture(self, data: mujoco.MjData) -> None:
        if self.step % GIF_SAMPLE_EVERY == 0:
            self.renderer.update_scene(data, self.camera)
            self.frames.append(self.renderer.render().copy())
        self.step += 1

    def save(self, path: Path) -> None:
        from PIL import Image

        pil_frames = [Image.fromarray(rgb) for rgb in self.frames]
        path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            str(path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / GIF_FPS),
            loop=0,
        )
        print(f"[gif] saved {path} ({len(pil_frames)} frames @ {GIF_FPS} fps)")


def _ramp(
    env: SO101PickPlaceEnv,
    phase: str,
    q_from: np.ndarray,
    q_to: np.ndarray,
    grip_from: float,
    grip_to: float,
    steps: int,
    stats: dict,
    verbose: bool,
    viewer: mujoco.viewer.Handle | None = None,
    recorder: _GifRecorder | None = None,
) -> dict:
    control_dt = env.model.opt.timestep * env.frame_skip
    phase_steps = 0
    for index in range(steps):
        if stats["done"]:
            break
        t = (index + 1) / steps
        q = q_from + t * (q_to - q_from)
        grip = grip_from + t * (grip_to - grip_from)
        _, _, terminated, truncated, info = env.step(_to_action(env, q, grip))
        if recorder is not None:
            recorder.capture(env.data)
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(control_dt)
        _update_stats(stats, info)
        stats["steps"] += 1
        phase_steps += 1
        if verbose and stats["steps"] % 50 == 0:
            print(
                f"  step={stats['steps']:4d} phase={phase:9s} "
                f"box_z={info['box_pos'][2]:.4f} "
                f"xy={info['box_cup_dist_xy']:.4f} "
                f"grasped={info['grasped']}"
            )
        if terminated or truncated:
            stats["done"] = True
            break

    record = {
        "phase": phase,
        "steps": phase_steps,
        "box_z": float(stats["box_pos"][2]),
        "box_cup_dist_xy": float(stats["box_cup_dist_xy"]),
        "ever_grasped": bool(stats["ever_grasped"]),
        "success_seen": bool(stats["success_seen"]),
    }
    print(
        f"phase={phase:9s} steps={phase_steps:3d} "
        f"box_z={record['box_z']:.4f} "
        f"xy={record['box_cup_dist_xy']:.4f} "
        f"grasped={record['ever_grasped']}"
    )
    return record


def run(args: argparse.Namespace) -> bool:
    env = SO101PickPlaceEnv(fix_cup=True, max_episode_steps=700)
    viewer: mujoco.viewer.Handle | None = None
    try:
        obs, info = env.reset(seed=args.seed)
        del obs
        if args.viewer:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        kin = Kinematics(env.model)
        mid = _joint_mid(env)

        box_pos = np.asarray(info["box_pos"], dtype=np.float64).copy()
        cup_pos = np.asarray(info["cup_pos"], dtype=np.float64).copy()

        recorder: _GifRecorder | None = None
        if args.gif:
            lookat = 0.5 * (box_pos + cup_pos)
            lookat[2] = 0.10
            recorder = _GifRecorder(env.model, lookat, args.top_view)

        # --- Grasp pose: pad midpoint at the box centre + PAD_MIDPOINT_Z. ---
        desired_grasp = box_pos + np.array([0.0, 0.0, GRASP_HEIGHT])
        wrist_minus = mid.copy()
        wrist_minus[4] = -1.0
        wrist_plus = mid.copy()
        wrist_plus[4] = 1.0
        q_grasp, grasp_pos_err, grasp_dir_err, grasp_pad_err = _multi_start(
            env,
            kin,
            desired_grasp,
            [mid, wrist_minus, wrist_plus],
            mid,
            "grasp",
        )

        # --- Pre-grasp 9 cm above the grasp pose (seeded from q_grasp). ---
        desired_pre = desired_grasp + np.array([0.0, 0.0, PRE_HEIGHT])
        q_pre, pre_pos_err, pre_dir_err, pre_pad_err = _multi_start(
            env,
            kin,
            desired_pre,
            [q_grasp, mid, wrist_minus, wrist_plus],
            mid,
            "pre-grasp",
        )

        # --- Lift 7.5 cm above the grasp pose. ---
        desired_lift = desired_grasp + np.array([0.0, 0.0, LIFT_HEIGHT])
        q_lift, lift_pos_err, lift_dir_err, _ = _multi_start(
            env,
            kin,
            desired_lift,
            [q_grasp, mid, wrist_minus, wrist_plus],
            mid,
            "lift",
        )

        # --- Above the cup, then a lower release pose above the rim. ---
        desired_cup_high = np.array(
            [cup_pos[0], cup_pos[1], CUP_HIGH_Z], dtype=np.float64
        )
        q_above_cup, cup_pos_err, cup_dir_err, cup_pad_err = _multi_start(
            env,
            kin,
            desired_cup_high,
            [q_grasp, q_lift, mid, wrist_minus, wrist_plus],
            mid,
            "above-cup",
        )

        desired_cup_low = np.array(
            [cup_pos[0], cup_pos[1], CUP_LOW_Z], dtype=np.float64
        )
        q_lower, low_pos_err, low_dir_err, low_pad_err = _multi_start(
            env,
            kin,
            desired_cup_low,
            [q_above_cup, q_grasp, mid, wrist_minus, wrist_plus],
            mid,
            "cup-release",
        )

        _, grasp_approach = kin.fk(q_grasp)
        grasp_down_dot = float(np.dot(DOWN, grasp_approach))
        print(
            f"IK grasp      : pos={grasp_pos_err * 100:.3f} cm "
            f"dir={grasp_dir_err:.4f} pad={grasp_pad_err * 100:.3f} cm "
            f"DOWN.dot={grasp_down_dot:.4f}"
        )
        print(
            f"IK pre-grasp  : pos={pre_pos_err * 100:.3f} cm "
            f"dir={pre_dir_err:.4f} pad={pre_pad_err * 100:.3f} cm"
        )
        print(
            f"IK lift       : pos={lift_pos_err * 100:.3f} cm "
            f"dir={lift_dir_err:.4f}"
        )
        print(
            f"IK above-cup  : pos={cup_pos_err * 100:.3f} cm "
            f"dir={cup_dir_err:.4f} pad={cup_pad_err * 100:.3f} cm"
        )
        print(
            f"IK cup-release: pos={low_pos_err * 100:.3f} cm "
            f"dir={low_dir_err:.4f} pad={low_pad_err * 100:.3f} cm"
        )
        print(
            f"box reset={np.round(box_pos, 4)} cup={np.round(cup_pos, 4)} "
            f"box->cup xy={np.linalg.norm(box_pos[:2] - cup_pos[:2]):.4f} m"
        )

        stats = {
            "steps": 0,
            "done": False,
            "ever_grasped": False,
            "max_box_z": float("-inf"),
            "success_seen": False,
            "box_pos": box_pos.copy(),
            "cup_pos": cup_pos.copy(),
            "box_cup_dist_xy": float(
                np.linalg.norm(box_pos[:2] - cup_pos[:2])
            ),
            "cup_upright": bool(info["cup_upright"]),
            "grasped": False,
        }

        q_start = env.data.qpos[env.arm_qpos_adr].copy()
        phases: list[dict] = []
        phases.append(
            _ramp(
                env,
                "approach",
                q_start,
                q_pre,
                env.open_q,
                env.open_q,
                PHASE_STEPS["approach"],
                stats,
                args.verbose,
                viewer,
                recorder,
            )
        )
        phases.append(
            _ramp(
                env,
                "descend",
                q_pre,
                q_grasp,
                env.open_q,
                env.open_q,
                PHASE_STEPS["descend"],
                stats,
                args.verbose,
                viewer,
                recorder,
            )
        )
        phases.append(
            _ramp(
                env,
                "close",
                q_grasp,
                q_grasp,
                env.open_q,
                env.close_q,
                PHASE_STEPS["close"],
                stats,
                args.verbose,
                viewer,
                recorder,
            )
        )
        phases.append(
            _ramp(
                env,
                "lift",
                q_grasp,
                q_lift,
                env.close_q,
                env.close_q,
                PHASE_STEPS["lift"],
                stats,
                args.verbose,
                viewer,
                recorder,
            )
        )
        phases.append(
            _ramp(
                env,
                "transport",
                q_lift,
                q_above_cup,
                env.close_q,
                env.close_q,
                PHASE_STEPS["transport"],
                stats,
                args.verbose,
                viewer,
                recorder,
            )
        )
        phases.append(
            _ramp(
                env,
                "lower",
                q_above_cup,
                q_lower,
                env.close_q,
                env.close_q,
                PHASE_STEPS["lower"],
                stats,
                args.verbose,
                viewer,
                recorder,
            )
        )
        phases.append(
            _ramp(
                env,
                "release",
                q_lower,
                q_lower,
                env.close_q,
                env.open_q,
                PHASE_STEPS["release"],
                stats,
                args.verbose,
                viewer,
                recorder,
            )
        )
        phases.append(
            _ramp(
                env,
                "settle",
                q_lower,
                q_lower,
                env.open_q,
                env.open_q,
                PHASE_STEPS["settle"],
                stats,
                args.verbose,
                viewer,
                recorder,
            )
        )

        if recorder is not None:
            recorder.save(GIF_TOP_PATH if args.top_view else GIF_PATH)

        final_box = np.asarray(stats["box_pos"], dtype=np.float64)
        final_cup = np.asarray(stats["cup_pos"], dtype=np.float64)
        final_xy = float(np.linalg.norm(final_box[:2] - final_cup[:2]))
        xy_ok = final_xy < env.success_threshold_xy
        z_ok = 0.01 < final_box[2] < env.success_threshold_z
        passed = bool(stats["success_seen"])

        print()
        print(f"steps_total   = {stats['steps']}")
        print(f"ever_grasped  = {stats['ever_grasped']}")
        print(f"max box z     = {stats['max_box_z']:.4f} m")
        print(f"final box z   = {final_box[2]:.4f} m")
        print(f"final box     = {np.round(final_box, 4)}")
        print(f"final cup     = {np.round(final_cup, 4)}")
        print(f"final box-cup xy = {final_xy:.4f} m")
        print(f"cup upright   = {stats['cup_upright']}")
        print(f"grasped_now   = {stats['grasped']}")
        print(
            f"conditions    : xy<{env.success_threshold_xy}={xy_ok} "
            f"0.01<z<{env.success_threshold_z}={z_ok} "
            f"upright={stats['cup_upright']}"
        )
        print(f"success_seen  = {passed}")
        print("PASS" if passed else "FAIL")

        if args.out is not None:
            summary = {
                "seed": args.seed,
                "phases": phases,
                "steps_total": stats["steps"],
                "ik": {
                    "grasp_pos_error_m": grasp_pos_err,
                    "grasp_dir_error": grasp_dir_err,
                    "pre_grasp_pos_error_m": pre_pos_err,
                    "pre_grasp_dir_error": pre_dir_err,
                    "lift_pos_error_m": lift_pos_err,
                    "lift_dir_error": lift_dir_err,
                    "above_cup_pos_error_m": cup_pos_err,
                    "above_cup_dir_error": cup_dir_err,
                    "cup_release_pos_error_m": low_pos_err,
                    "cup_release_dir_error": low_dir_err,
                },
                "ever_grasped": bool(stats["ever_grasped"]),
                "max_box_z": float(stats["max_box_z"]),
                "final_box_pos": final_box.tolist(),
                "final_cup_pos": final_cup.tolist(),
                "final_box_cup_dist_xy": final_xy,
                "cup_upright": bool(stats["cup_upright"]),
                "success_seen": passed,
                "passed": passed,
            }
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print(f"wrote {out_path}")

        if viewer is not None and viewer.is_running():
            print("Viewer open - close the window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 30.0)

        return passed
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open a live MuJoCo viewer (requires a display; real-time playback)",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help=f"save an animated GIF to {GIF_PATH}",
    )
    parser.add_argument(
        "--top-view",
        action="store_true",
        help=f"with --gif, render from above into {GIF_TOP_PATH}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print a trace line every 50 environment steps",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional JSON path for the summary dict",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(0 if run(parse_args()) else 1)
