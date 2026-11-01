from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces


class SO101PointToPointEnv(gymnasium.Env):
    """SO-101 Point-to-Point environment (Gymnasium).

    Task: Move the end effector from point_A to point_B (fixed).
    Reward: progress_scale * (prev_dist - dist) + success_bonus if dist < success_threshold.
    Success: distance < success_threshold.
    Observation: [arm_qpos(6), arm_qvel(6), ee_pos(3), target(3)] = 18 dims.
    Action: Box(-1,1,6). Linear rescale to actuator ctrlrange.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 250,
        success_threshold: float = 0.02,
        progress_scale: float = 10.0,
        success_bonus: float = 5.0,
        start_site: str = "point_A",
        goal_site: str = "point_B",
    ) -> None:
        super().__init__()
        if xml_path is None:
            xml_path = Path(__file__).resolve().parents[2] / "models" / "so101" / "scene_with_markers.xml"
        self.xml_path = Path(xml_path)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.success_threshold = float(success_threshold)
        self.progress_scale = float(progress_scale)
        self.success_bonus = float(success_bonus)
        self.start_site_name = start_site
        self.goal_site_name = goal_site

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Arm joint names in order
        arm_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self.arm_joint_ids = []
        for name in arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint not found: {name}")
            self.arm_joint_ids.append(jid)
        self.arm_joint_ids = np.array(self.arm_joint_ids, dtype=int)

        # Addresses
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_qvel_adr = self.model.jnt_dofadr[self.arm_joint_ids]

        # EE site
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        if self.ee_site_id < 0:
            raise RuntimeError("Site 'gripperframe' not found")

        # Start/goal sites
        self.start_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.start_site_name)
        if self.start_site_id < 0:
            raise RuntimeError(f"Site '{self.start_site_name}' not found")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.goal_site_name)
        if self.goal_site_id < 0:
            raise RuntimeError(f"Site '{self.goal_site_name}' not found")

        # Actuator ranges
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # ctrl[:6] is indexed positionally below -> actuator order must match
        act_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        if act_names != arm_joint_names:
            raise RuntimeError(
                f"Actuator order {act_names} != arm joint order {arm_joint_names}"
            )

        # Action space [-1,1] for 6 actuators
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation bounds
        qpos_low = self.model.jnt_range[self.arm_joint_ids, 0].copy()
        qpos_high = self.model.jnt_range[self.arm_joint_ids, 1].copy()
        qvel_low = np.array([-20.0] * 6, dtype=np.float64)
        qvel_high = np.array([20.0] * 6, dtype=np.float64)
        ee_low = np.array([-0.6, -0.6, -0.1], dtype=np.float64)
        ee_high = np.array([0.6, 0.6, 0.6], dtype=np.float64)
        obs_low = np.concatenate([qpos_low, qvel_low, ee_low, ee_low]).astype(np.float32)
        obs_high = np.concatenate([qpos_high, qvel_high, ee_high, ee_high]).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Read start/goal positions
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.start_xyz = self.data.site_xpos[self.start_site_id].copy()
        self.goal_xyz = self.data.site_xpos[self.goal_site_id].copy()

        # Solve IK for start position
        self.start_qpos = self._solve_ik(self.start_xyz, iters=200)
        # Check EE error
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_adr] = self.start_qpos
        self.data.ctrl[:6] = np.clip(self.start_qpos, self.ctrl_low, self.ctrl_high)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        ee_err = float(np.linalg.norm(self.data.site_xpos[self.ee_site_id] - self.start_xyz))
        if ee_err > 1e-3:
            raise RuntimeError(f"IK failed for start: EE error {ee_err} > 1e-3")

        self._elapsed = 0
        self._prev_dist = 0.0

    def _solve_ik(self, target_xyz: np.ndarray, iters: int = 200) -> np.ndarray:
        # Initialize from current qpos
        mujoco.mj_resetData(self.model, self.data)
        # Start from a reasonable initial guess (mid of ranges)
        q0 = (self.model.jnt_range[self.arm_joint_ids, 0] + self.model.jnt_range[self.arm_joint_ids, 1]) * 0.5
        self.data.qpos[self.arm_qpos_adr] = q0
        self.data.ctrl[:6] = np.clip(q0, self.ctrl_low, self.ctrl_high)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        damping = 1e-2
        for _ in range(iters):
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.site_xpos[self.ee_site_id]
            error = target_xyz - ee_pos  # (3,)
            if np.linalg.norm(error) < 1e-6:
                break
            # Jacobian for site, shape (3, nv)
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
            # Restrict to arm DOF columns
            J = jacp[:, self.arm_qvel_adr]  # (3, 6)
            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 error
            JJT = J @ J.T
            A = JJT + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, error)
            # Update qpos
            q = self.data.qpos[self.arm_qpos_adr].copy()
            q = q + dq
            # Clip to joint ranges
            q = np.clip(q, self.model.jnt_range[self.arm_joint_ids, 0], self.model.jnt_range[self.arm_joint_ids, 1])
            self.data.qpos[self.arm_qpos_adr] = q
            self.data.ctrl[:6] = np.clip(q, self.ctrl_low, self.ctrl_high)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos[self.arm_qpos_adr].copy()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Set arm qpos = start_qpos
        self.data.qpos[self.arm_qpos_adr] = self.start_qpos
        self.data.ctrl[:6] = np.clip(self.start_qpos, self.ctrl_low, self.ctrl_high)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._target = self.goal_xyz.copy()
        obs = self._get_obs()
        dist = float(np.linalg.norm(self.data.site_xpos[self.ee_site_id] - self._target))
        self._prev_dist = dist
        self._elapsed = 0
        info = {"distance": dist, "is_success": dist < self.success_threshold, "target": self._target.copy(), "start": self.start_xyz.copy(), "goal": self.goal_xyz.copy()}
        return obs.astype(np.float32), info

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self.arm_qpos_adr].copy()
        qvel = self.data.qvel[self.arm_qvel_adr].copy()
        ee = self.data.site_xpos[self.ee_site_id].copy()
        target = self._target.copy()
        obs = np.concatenate([qpos, qvel, ee, target])
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,):
            action = action.reshape(6)
        action_clipped = np.clip(action, -1.0, 1.0)
        # map [-1,1] to ctrlrange (exact mapping from SO101ReachEnv)
        ctrl = self.ctrl_low + (action_clipped.astype(np.float64) + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)
        self.data.ctrl[:6] = ctrl.astype(np.float64)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        ee_pos = self.data.site_xpos[self.ee_site_id]
        dist = float(np.linalg.norm(ee_pos - self._target))
        reward = self.progress_scale * (self._prev_dist - dist)
        terminated = dist < self.success_threshold
        if terminated:
            reward += self.success_bonus
        self._elapsed += 1
        truncated = self._elapsed >= self.max_episode_steps
        self._prev_dist = dist
        obs = self._get_obs()
        info = {"distance": dist, "is_success": terminated, "target": self._target.copy(), "start": self.start_xyz.copy(), "goal": self.goal_xyz.copy()}
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        return None


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import gymnasium.utils.env_checker as env_checker

    import mira  # noqa: F401

    env_checker.check_env(SO101PointToPointEnv())
    print("check_env OK")

    env = gymnasium.make("MIRA/SO101PointToPoint-v0")
    for ep in range(2):
        obs, info = env.reset()
        print(f"ep{ep} start_dist={info['distance']:.6f}")
        done = False
        steps = 0
        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        print(f"ep{ep} final_dist={info['distance']:.6f} success={info['is_success']}")
    env.close()
