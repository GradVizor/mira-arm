from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces


def lerobot_gripper_to_ctrl(x100: float, ctrl_low: float, ctrl_high: float) -> float:
    """Map LeRobot gripper convention (0 = closed, 100 = open) to hinge ctrl.

    LeRobot treats gripper as linear: 0 = closed, 100 = open. This maps to
    the hinge control value in [ctrl_low, ctrl_high].
    """
    x = float(np.clip(x100, 0.0, 100.0))
    return float(ctrl_low + (x / 100.0) * (ctrl_high - ctrl_low))


class SO101ReachEnv(gymnasium.Env):
    """SO-101 Reach environment (Gymnasium).

    Task: Reach — move the gripper site to a random target.
    Reward: -||site_pos - target||.
    Success: distance < success_threshold.
    Observation: [arm_qpos(6), arm_qvel(6), ee_pos(3), target(3)] = 18 dims.
    Action: Box(-1,1,6). Gripper action in [-1,1] corresponds to LeRobot [0,100]
    (closed..open) via action_gripper = (lerobot/50.0 - 1.0) in effect; the
    mapping uses linear rescale to actuator ctrlrange.
    Control: model.opt.timestep * frame_skip (default 0.002*5=0.01s).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 100,
        success_threshold: float = 0.02,
        target_low: tuple[float, float, float] = (0.05, -0.22, 0.04),
        target_high: tuple[float, float, float] = (0.28, 0.22, 0.30),
    ) -> None:
        super().__init__()
        if xml_path is None:
            xml_path = Path(__file__).resolve().parents[2] / "models" / "so101" / "scene.xml"
        self.xml_path = Path(xml_path)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.success_threshold = float(success_threshold)
        self.target_low = np.array(target_low, dtype=np.float64)
        self.target_high = np.array(target_high, dtype=np.float64)

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

        # Actuator ranges
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # ctrl[:6] is indexed positionally below -> actuator order must match
        # the arm joint order (the scene has extra free-joint prop bodies).
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

        self._elapsed = 0
        self._target = np.zeros(3, dtype=np.float64)

    def _sample_target(self) -> np.ndarray:
        return self.np_random.uniform(self.target_low, self.target_high).astype(np.float64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Home pose - mid of ctrlrange + small noise
        home = (self.ctrl_low + self.ctrl_high) * 0.5
        noise = self.np_random.uniform(-0.05, 0.05, size=6)
        qpos_home = home + noise
        # clip to joint ranges if needed
        qpos_home = np.clip(qpos_home, self.model.jnt_range[self.arm_joint_ids, 0], self.model.jnt_range[self.arm_joint_ids, 1])
        self.data.qpos[self.arm_qpos_adr] = qpos_home
        self.data.ctrl[:6] = np.clip(qpos_home, self.ctrl_low, self.ctrl_high)
        self.data.qvel[:] = 0.0
        self._target = self._sample_target()
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        dist = float(np.linalg.norm(self.data.site_xpos[self.ee_site_id] - self._target))
        info = {"distance": dist, "is_success": dist < self.success_threshold, "target": self._target.copy()}
        self._elapsed = 0
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
        # map [-1,1] to ctrlrange
        ctrl = self.ctrl_low + (action_clipped.astype(np.float64) + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)
        self.data.ctrl[:6] = ctrl.astype(np.float64)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        ee_pos = self.data.site_xpos[self.ee_site_id]
        dist = float(np.linalg.norm(ee_pos - self._target))
        reward = -dist
        terminated = dist < self.success_threshold
        self._elapsed += 1
        truncated = self._elapsed >= self.max_episode_steps
        obs = self._get_obs()
        info = {"distance": dist, "is_success": terminated, "target": self._target.copy()}
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        return None


if __name__ == "__main__":
    import sys

    # Allow both `python3 mira/envs/so101_reach_env.py` and `python3 -m ...`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import gymnasium.utils.env_checker as env_checker

    import mira  # noqa: F401  (registers "MIRA/SO101Reach-v0")

    env_checker.check_env(SO101ReachEnv())
    print("check_env OK")

    env = gymnasium.make("MIRA/SO101Reach-v0")
    returns = []
    successes = 0
    episodes = 0
    # run 10000 steps total
    steps = 0
    max_steps = 10000
    while steps < max_steps:
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            steps += 1
            done = terminated or truncated
        episodes += 1
        returns.append(ep_ret)
        if info.get("is_success", False):
            successes += 1
    mean_ret = float(np.mean(returns)) if returns else 0.0
    success_rate = float(successes / episodes) if episodes else 0.0
    print(f"episodes={episodes} mean_return={mean_ret:.4f} success_rate={success_rate:.4f}")
    env.close()
