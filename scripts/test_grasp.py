#!/usr/bin/env python3
"""Step 0 grasp test: approach, close, and lift the marshmallow.

Run:

    python3 scripts/test_grasp.py
    python3 scripts/test_grasp.py --viewer   # live playback (needs a display)

The stock SO-101 finger meshes cannot grasp the small marshmallow: their tapered
tips contact the box corners and press it into the floor.  This test uses the
flat-pad model from ``mira.pick_place_model`` and reports whether the box rises
with the gripper.

Exit code is 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mira.pick_place_model import (  # noqa: E402
    ARM_JOINT_NAMES,
    DEFAULT_SCENE,
    build_padded_model,
)

DOWN = np.array([0.0, 0.0, -1.0])
BOX_CENTER = np.array([0.16, 0.11, 0.0125])
PAD_MIDPOINT_Z = 0.0015


class Kinematics:
    """Five-joint IK for position + world-down, using the gripper approach axis."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES
            ],
            dtype=int,
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError("Arm joint missing from model")
        self.qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.lo = model.jnt_range[self.joint_ids, 0].copy()
        self.hi = model.jnt_range[self.joint_ids, 1].copy()
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self.gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        site_local = model.site_pos[self.site_id]
        self.approach_local = site_local / np.linalg.norm(site_local)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.xmat[self.gripper_id].reshape(3, 3)
        return pos, rot @ self.approach_local

    def solve(
        self,
        target: np.ndarray,
        q_init: np.ndarray | None = None,
        *,
        iters: int = 1200,
        wdir: float = 0.05,
        damping: float = 3e-3,
    ) -> np.ndarray:
        q = (0.5 * (self.lo + self.hi)) if q_init is None else q_init.copy()
        if q_init is None:
            q[5] = 0.25

        for _ in range(iters):
            pos, approach = self.fk(q)
            error = np.concatenate(
                [target - pos, wdir * (DOWN - approach)]
            )
            if (
                np.linalg.norm(target - pos) < 2e-4
                and np.linalg.norm(wdir * (DOWN - approach)) / wdir < 6e-3
            ):
                break

            jac = np.zeros((6, 5))
            eps = 1e-6
            for joint in range(5):
                qp = q.copy()
                qp[joint] += eps
                pos2, approach2 = self.fk(qp)
                jac[:3, joint] = (pos2 - pos) / eps
                jac[3:, joint] = wdir * ((approach2 - approach) / eps)
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + damping**2 * np.eye(6), error
            )
            q[:5] = np.clip(q[:5] + dq, self.lo[:5], self.hi[:5])

        return q

    def pose_error(self, target: np.ndarray, q: np.ndarray) -> tuple[float, float]:
        pos, approach = self.fk(q)
        return (
            float(np.linalg.norm(target - pos)),
            float(np.linalg.norm(DOWN - approach)),
        )


def _set_box(model: mujoco.MjModel, data: mujoco.MjData, yaw: float) -> None:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "marshmallow_free"
    )
    if joint_id < 0:
        raise RuntimeError("marshmallow_free joint not found")
    adr = model.jnt_qposadr[joint_id]
    data.qpos[adr : adr + 3] = BOX_CENTER
    data.qpos[adr + 3 : adr + 7] = [
        np.cos(0.5 * yaw),
        0.0,
        0.0,
        np.sin(0.5 * yaw),
    ]


def _ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    grip: float,
) -> None:
    ctrl = np.clip(q, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    ctrl[5] = grip
    data.ctrl[:] = ctrl


def _step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: mujoco.viewer.Handle | None = None,
) -> None:
    """Step physics and, when requested, sync the live viewer at 60 Hz."""
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()
        time.sleep(1.0 / 60.0)


def _settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int,
    viewer: mujoco.viewer.Handle | None = None,
) -> None:
    for _ in range(steps):
        _step(model, data, viewer)


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        or f"geom{geom_id}"
    )


def _print_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for index in range(data.ncon):
        contact = data.contact[index]
        name1 = _geom_name(model, contact.geom1)
        name2 = _geom_name(model, contact.geom2)
        if "pad" not in name1 + name2 and "marshmallow" not in name1 + name2:
            continue
        normal = contact.frame[:3]
        print(
            f"  {name1:18s} <-> {name2:18s} "
            f"dist={contact.dist * 1000:+7.2f}mm normal={normal}"
        )


def run(args: argparse.Namespace) -> bool:
    model, pads = build_padded_model(
        args.scene, open_q=args.open_q, force_limit=args.force
    )
    kin = Kinematics(model)
    data = kin.data

    viewer = None
    if args.viewer:
        gl = os.environ.get("MUJOCO_GL")
        if gl in {"egl", "osmesa"}:
            raise SystemExit(
                f"--viewer requires a windowed GL backend, but MUJOCO_GL={gl}. "
                "Unset it or use glfw."
            )
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise SystemExit(
                "--viewer requires an X11 display, but DISPLAY is unset. "
                "Run on a desktop or use Xvfb."
            )
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = BOX_CENTER + np.array([0.0, 0.0, 0.04])
        viewer.cam.distance = 0.35
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        viewer.sync()

    mujoco.mj_resetData(model, data)
    _set_box(model, data, 0.0)

    # Solve the site target so the pad midpoint lands at the box centre + 1.5 mm.
    desired = BOX_CENTER + np.array([0.0, 0.0, PAD_MIDPOINT_Z])
    target = desired.copy()
    q_grasp: np.ndarray | None = None
    for _ in range(10):
        q_grasp = kin.solve(target, q_init=q_grasp)
        rot = kin.data.xmat[kin.gripper_id].reshape(3, 3)
        next_target = desired - rot @ pads.site_offset
        if np.linalg.norm(next_target - target) < 1e-6:
            target = next_target
            break
        target = next_target
    q_grasp = kin.solve(target, q_init=q_grasp)
    rot = kin.data.xmat[kin.gripper_id].reshape(3, 3)
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    pad_mid = kin.data.site_xpos[kin.site_id] + rot @ pads.site_offset
    pad_error = float(np.linalg.norm(pad_mid - desired))

    q_pre = kin.solve(target + np.array([0.0, 0.0, args.pre_height]), q_init=q_grasp)
    q_lift = kin.solve(
        target + np.array([0.0, 0.0, args.lift_height]), q_init=q_grasp
    )
    pre_pos_error, pre_dir_error = kin.pose_error(
        target + np.array([0.0, 0.0, args.pre_height]), q_pre
    )
    lift_pos_error, lift_dir_error = kin.pose_error(
        target + np.array([0.0, 0.0, args.lift_height]), q_lift
    )

    # Reset and place the box with the yaw the arm naturally gives at the grasp.
    mujoco.mj_resetData(model, data)
    _set_box(model, data, yaw)
    data.qvel[:] = 0.0
    data.qpos[kin.qpos_adr] = q_pre
    mujoco.mj_forward(model, data)
    _ctrl(model, data, q_pre, args.open_q)
    _settle(model, data, args.settle_steps, viewer)

    box_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "marshmallow_free"
    )
    box_adr = model.jnt_qposadr[box_joint]
    box_dof = model.jnt_dofadr[box_joint]
    box_start = data.qpos[box_adr : box_adr + 3].copy()

    # Descend in joint space; the pre-grasp and grasp solutions differ mainly in
    # the arm joints, so this stays close to a vertical approach.
    for step in range(args.descent_steps):
        t = (step + 1) / args.descent_steps
        q = q_pre + t * (q_grasp - q_pre)
        _ctrl(model, data, q, args.open_q)
        _step(model, data, viewer)
    _ctrl(model, data, q_grasp, args.open_q)
    _settle(model, data, args.settle_steps, viewer)
    box_grasp = data.qpos[box_adr : box_adr + 3].copy()
    site_grasp = data.site_xpos[kin.site_id].copy()

    # Ramp the gripper closed instead of slamming it; high force ejects the box.
    for step in range(args.close_steps):
        grip = args.open_q + (args.close_q - args.open_q) * (step + 1) / args.close_steps
        _ctrl(model, data, q_grasp, grip)
        _step(model, data, viewer)
    _ctrl(model, data, q_grasp, args.close_q)
    _settle(model, data, args.settle_steps, viewer)
    box_closed = data.qpos[box_adr : box_adr + 3].copy()
    qgrip = float(data.qpos[kin.qpos_adr[5]])

    print(f"box target        : {BOX_CENTER}")
    print(f"site target       : {target}")
    print(f"natural box yaw   : {np.degrees(yaw):7.2f} deg")
    print(f"pad midpoint error: {pad_error * 100:7.3f} cm")
    print(
        f"pre-grasp error   : {pre_pos_error * 100:7.3f} cm pos, "
        f"{pre_dir_error:7.4f} dir"
    )
    print(
        f"grasp error       : {kin.pose_error(target, q_grasp)[0] * 100:7.3f} cm pos"
    )
    print(f"descent xy shift  : {np.linalg.norm(box_grasp[:2] - box_start[:2]) * 100:7.3f} cm")
    print(
        f"close shift       : ({(box_closed - box_grasp)[0] * 100:+.3f}, "
        f"{(box_closed - box_grasp)[1] * 100:+.3f}) cm, qgrip={qgrip:+.3f}"
    )
    print("contacts after close:")
    _print_contacts(model, data)

    # Lift while holding the gripper closed.
    for step in range(args.lift_steps):
        t = (step + 1) / args.lift_steps
        q = q_grasp + t * (q_lift - q_grasp)
        _ctrl(model, data, q, args.close_q)
        _step(model, data, viewer)
    _ctrl(model, data, q_lift, args.close_q)
    _settle(model, data, args.settle_steps, viewer)
    box_lifted = data.qpos[box_adr : box_adr + 3].copy()
    ee_lifted = data.site_xpos[kin.site_id].copy()
    box_vel = np.linalg.norm(data.qvel[box_dof : box_dof + 3])
    ee_box = float(np.linalg.norm(ee_lifted - box_lifted))

    print(
        f"lift error        : {lift_pos_error * 100:7.3f} cm pos, "
        f"{lift_dir_error:7.4f} dir"
    )
    print(
        f"box z             : {box_lifted[2] * 100:7.2f} cm "
        f"(rise {(box_lifted[2] - BOX_CENTER[2]) * 100:+.2f} cm)"
    )
    print(f"box speed         : {box_vel * 100:7.3f} cm/s")
    print(f"EE-box distance   : {ee_box * 100:7.3f} cm")
    print("contacts after lift:")
    _print_contacts(model, data)

    passed = (
        pad_error < 1e-3
        and np.linalg.norm(box_grasp[:2] - box_start[:2]) < 0.01
        and box_lifted[2] - BOX_CENTER[2] >= args.min_lift
        and ee_box <= args.max_ee_box
        and lift_pos_error < 0.005
    )
    print("PASS" if passed else "FAIL")
    if viewer is not None:
        print("Viewer open - close the window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 30.0)
        viewer.close()
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open a live MuJoCo viewer (requires a display; real-time playback)",
    )
    parser.add_argument("--force", type=float, default=0.5, help="gripper N.m limit")
    parser.add_argument("--open-q", type=float, default=0.25)
    parser.add_argument("--close-q", type=float, default=0.10)
    parser.add_argument("--pre-height", type=float, default=0.09)
    parser.add_argument("--lift-height", type=float, default=0.075)
    parser.add_argument("--descent-steps", type=int, default=600)
    parser.add_argument("--close-steps", type=int, default=500)
    parser.add_argument("--lift-steps", type=int, default=600)
    parser.add_argument("--settle-steps", type=int, default=300)
    parser.add_argument("--min-lift", type=float, default=0.03)
    parser.add_argument("--max-ee-box", type=float, default=0.025)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(0 if run(parse_args()) else 1)
