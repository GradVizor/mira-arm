"""Shared SO-101 IK helpers for pick-and-place scripts and environments.

Extracted verbatim from ``scripts/test_grasp.py`` so the grasp test and the
Gymnasium environment solve the same position + world-down problem.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mira.pick_place_model import ARM_JOINT_NAMES

DOWN = np.array([0.0, 0.0, -1.0])
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
