from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from mira.kinematics import PAD_MIDPOINT_Z, Kinematics
from mira.pick_place_model import DEFAULT_SCENE, build_padded_model


class SO101PickPlaceEnv(gymnasium.Env):
    """SO-101 Pick-and-Place environment (Gymnasium).

    Task: Grasp the marshmallow (2.5 cm box) and place it into the cup.
    Uses the padded grasp model (flat pads, finger mesh collisions disabled).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 700,
        success_threshold_xy: float = 0.02,
        success_threshold_z: float = 0.06,  # inside cup cavity region
        approach_scale: float = 5.0,
        transport_scale: float = 10.0,
        shaping_gamma: float = 1.0,
        grasp_bonus: float = 2.0,
        success_bonus: float = 10.0,
        lift_bonus: float = 5.0,
        time_penalty: float = 0.0,
        use_pads: bool = True,
        gripper_force: float = 0.5,
        open_q: float = 0.25,
        close_q: float = 0.10,
        randomize_box: bool = False,
        randomize_box_pos: bool = False,
        randomize_cup_pos: bool = False,
        randomize_physics: bool = False,
        fix_cup: bool = False,
        failure_penalty: float = 5.0,
        release_required: bool = False,
        warm_start: bool = False,
        pregrasp_height: float = 0.09,
        closure_scale: float = 4.0,
        closure_gate_radius: float = 0.02,
        lift_scale: float = 20.0,
        box_yaw: float | None = None,
        task_mode: str = "full",
    ) -> None:
        super().__init__()
        if xml_path is None:
            xml_path = DEFAULT_SCENE
        self.xml_path = Path(xml_path)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.success_threshold_xy = float(success_threshold_xy)
        self.success_threshold_z = float(success_threshold_z)
        self.approach_scale = float(approach_scale)
        self.transport_scale = float(transport_scale)
        self.shaping_gamma = float(shaping_gamma)
        self.grasp_bonus = float(grasp_bonus)
        self.success_bonus = float(success_bonus)
        self.lift_bonus = float(lift_bonus)
        self.time_penalty = float(time_penalty)
        self.use_pads = bool(use_pads)
        self.gripper_force = float(gripper_force)
        self.open_q = float(open_q)
        self.close_q = float(close_q)
        self.randomize_box = bool(randomize_box)
        self.randomize_box_pos = bool(randomize_box_pos)
        self.randomize_cup_pos = bool(randomize_cup_pos)
        self.randomize_physics = bool(randomize_physics)
        self.fix_cup = bool(fix_cup)
        self.failure_penalty = float(failure_penalty)
        self.release_required = bool(release_required)
        self.warm_start = bool(warm_start)
        self.pregrasp_height = float(pregrasp_height)
        self.closure_scale = float(closure_scale)
        self.closure_gate_radius = float(closure_gate_radius)
        self.lift_scale = float(lift_scale)
        self.box_yaw_fixed = box_yaw  # if None, use natural yaw ~40.87°
        self.task_mode = str(task_mode)
        if self.task_mode == "lift_only":
            self.transport_scale = 0.0

        # Build model with pads
        if self.use_pads:
            self.model, self.pads = build_padded_model(
                self.xml_path, open_q=self.open_q, force_limit=self.gripper_force
            )
        else:
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self.pads = None
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

        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_qvel_adr = self.model.jnt_dofadr[self.arm_joint_ids]

        # EE site
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        if self.ee_site_id < 0:
            raise RuntimeError("Site 'gripperframe' not found")
        self.gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        if self.gripper_body_id < 0:
            raise RuntimeError("Body 'gripper' not found")

        # Box and cup
        self.box_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "marshmallow_free")
        self.cup_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "coffee_cup_free")
        if self.box_joint_id < 0:
            raise RuntimeError("marshmallow_free joint not found")
        if self.cup_joint_id < 0:
            raise RuntimeError("coffee_cup_free joint not found")
        self.box_qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        self.cup_qpos_adr = self.model.jnt_qposadr[self.cup_joint_id]
        self.cup_dof_adr = self.model.jnt_dofadr[self.cup_joint_id]
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "marshmallow")
        self.cup_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_cup")
        self.box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "marshmallow_geom")
        if min(self.box_body_id, self.cup_body_id, self.box_geom_id) < 0:
            raise RuntimeError("marshmallow/coffee_cup body or marshmallow_geom not found")
        if self.use_pads:
            self.fixture_pad_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixture_pad"
            )
            self.moving_pad_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_pad"
            )
            if min(self.fixture_pad_geom_id, self.moving_pad_geom_id) < 0:
                raise RuntimeError("fixture_pad/moving_pad geom not found")
        else:
            self.fixture_pad_geom_id = -1
            self.moving_pad_geom_id = -1
        self.box_mass_default = float(self.model.body_mass[self.box_body_id])
        self.box_friction_default = self.model.geom_friction[self.box_geom_id].copy()

        # Actuator ranges
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        act_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        if act_names != arm_joint_names:
            raise RuntimeError(f"Actuator order {act_names} != arm joint order {arm_joint_names}")

        # Action space [-1,1] for 6 actuators
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation: arm_qpos(6), arm_qvel(6), ee_pos(3), box_pos(3), cup_pos(3) = 21 dims
        qpos_low = self.model.jnt_range[self.arm_joint_ids, 0].copy()
        qpos_high = self.model.jnt_range[self.arm_joint_ids, 1].copy()
        qvel_low = np.array([-20.0] * 6, dtype=np.float64)
        qvel_high = np.array([20.0] * 6, dtype=np.float64)
        ee_low = np.array([-0.6, -0.6, -0.1], dtype=np.float64)
        ee_high = np.array([0.6, 0.6, 0.6], dtype=np.float64)
        box_low = np.array([-0.05, -0.35, -0.05], dtype=np.float64)
        box_high = np.array([0.45, 0.35, 0.40], dtype=np.float64)
        cup_low = np.array([-0.05, -0.35, -0.05], dtype=np.float64)
        cup_high = np.array([0.45, 0.05, 0.40], dtype=np.float64)
        obs_low = np.concatenate([qpos_low, qvel_low, ee_low, box_low, cup_low]).astype(np.float32)
        obs_high = np.concatenate([qpos_high, qvel_high, ee_high, box_high, cup_high]).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Cup position
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.cup_pos = self._get_cup_pos().copy()
        self.cup_base_pos = self.cup_pos.copy()

        # Natural grasp yaw from test_grasp
        self.natural_box_yaw = 40.87 * np.pi / 180.0
        if self.box_yaw_fixed is not None:
            self.natural_box_yaw = float(self.box_yaw_fixed)

        self._elapsed = 0
        self._prev_phi = 0.0
        self._prev_phi_grasp = 0.0
        self._lifted = False
        self._grasped = False
        self._ever_grasped = False
        self._pregrasp_prev: np.ndarray | None = None
        self._pregrasp_box: np.ndarray | None = None

    def _get_box_pos(self) -> np.ndarray:
        return self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3].copy()

    def _get_cup_pos(self) -> np.ndarray:
        return self.data.qpos[self.cup_qpos_adr : self.cup_qpos_adr + 3].copy()

    def _cup_is_upright(self) -> bool:
        # Body local +z dotted with world +z; > 0.9 means < ~26 degrees tilt.
        return bool(self.data.xmat[self.cup_body_id].reshape(3, 3)[2, 2] > 0.9)

    def _pin_cup(self) -> None:
        if not self.fix_cup:
            return
        self.data.qpos[self.cup_qpos_adr : self.cup_qpos_adr + 3] = self.cup_base_pos
        self.data.qpos[self.cup_qpos_adr + 3 : self.cup_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.cup_dof_adr : self.cup_dof_adr + 6] = 0.0

    def _apply_box_physics(self, rng: np.random.Generator) -> None:
        self.model.body_mass[self.box_body_id] = self.box_mass_default
        self.model.geom_friction[self.box_geom_id] = self.box_friction_default
        if not self.randomize_physics:
            return
        self.model.body_mass[self.box_body_id] *= rng.uniform(0.7, 1.3)
        friction = self.box_friction_default.copy()
        friction[0] *= rng.uniform(0.8, 1.2)
        self.model.geom_friction[self.box_geom_id] = friction
        mujoco.mj_setConst(self.model, self.data)

    def _set_box(self, pos: np.ndarray, yaw: float) -> None:
        self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3] = pos
        self.data.qpos[self.box_qpos_adr + 3 : self.box_qpos_adr + 7] = [
            np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)
        ]

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self.arm_qpos_adr].copy()
        qvel = self.data.qvel[self.arm_qvel_adr].copy()
        ee = self.data.site_xpos[self.ee_site_id].copy()
        box = self._get_box_pos()
        cup = self._get_cup_pos()
        obs = np.concatenate([qpos, qvel, ee, box, cup])
        return obs.astype(np.float32)

    def _pads_contact_flags(self) -> tuple[bool, bool]:
        """Return ``(fixture_touch, moving_touch)`` against the marshmallow."""
        if not self.use_pads:
            return False, False
        fixture = int(self.fixture_pad_geom_id)
        moving = int(self.moving_pad_geom_id)
        box = int(self.box_geom_id)
        fixture_touch = False
        moving_touch = False
        for contact in self.data.contact[: self.data.ncon]:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if box != geom1 and box != geom2:
                continue
            other = geom2 if geom1 == box else geom1
            if other == fixture:
                fixture_touch = True
            elif other == moving:
                moving_touch = True
        return fixture_touch, moving_touch

    def _pads_contact_box(self) -> bool:
        """Return True only when both pads touch the marshmallow geom."""
        fixture_touch, moving_touch = self._pads_contact_flags()
        return fixture_touch and moving_touch

    def _pad_midpoint(self) -> np.ndarray:
        """World position of the midpoint between the two grasp pads."""
        site = self.data.site_xpos[self.ee_site_id]
        if not self.use_pads:
            return site.copy()
        rot = self.data.xmat[self.gripper_body_id].reshape(3, 3)
        return (site + rot @ self.pads.site_offset).copy()

    def _pad_box_distances(self) -> tuple[float, float]:
        """Centre-to-centre distances from each pad to the marshmallow."""
        box = self._get_box_pos()
        if not self.use_pads:
            dist = float(np.linalg.norm(self._pad_midpoint() - box))
            return dist, dist
        fixture = self.data.geom_xpos[self.fixture_pad_geom_id]
        moving = self.data.geom_xpos[self.moving_pad_geom_id]
        return (
            float(np.linalg.norm(fixture - box)),
            float(np.linalg.norm(moving - box)),
        )

    def _gripper_openness(self) -> float:
        """0.0 when fully closed, 1.0 at ``open_q`` (clipped to that span)."""
        span = self.open_q - self.close_q
        if span <= 0.0:
            return 0.0
        q = float(self.data.qpos[self.arm_qpos_adr[5]])
        return float(np.clip((q - self.close_q) / span, 0.0, 1.0))

    def _closure_potential(self) -> float:
        """Pad-gated closure potential.

        Zero while the gripper is open, so approaching with an open gripper is
        never penalised; rises to ``closure_scale * gate`` as the pads close
        around the box.  Used as ``phi' - phi`` so closing pays and opening
        returns the same amount.
        """
        if not self.use_pads or self.closure_gate_radius <= 0.0:
            return 0.0
        d_fixture, d_moving = self._pad_box_distances()
        gate = np.clip(
            (self.closure_gate_radius - max(d_fixture, d_moving))
            / self.closure_gate_radius,
            0.0,
            1.0,
        )
        closure = 1.0 - self._gripper_openness()
        return float(self.closure_scale * closure * gate)


    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map ``[-1, 1]`` actions to actuator targets.

        Arm joints span their full ctrlrange.  The gripper spans only
        ``[close_q, open_q]`` (a full joint-range mapping splays the pads to
        12 cm and makes closing unreachable for a tanh policy).
        """
        a = np.asarray(action, dtype=np.float64)
        ctrl = self.ctrl_low + (a + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)
        ctrl[5] = self.close_q + (a[5] + 1.0) * 0.5 * (self.open_q - self.close_q)
        return ctrl.astype(np.float64)

    def _potential(self, ee_box_dist: float, box_cup_dist_xy: float) -> float:
        """Potential used for reward shaping.

        Transport term is gated on grasp: only applies when the box is being
        held, preventing the policy from pushing the box toward the cup.
        """
        # When grasped, disable approach (box moves with gripper) and
        # only shape transport. This forces the policy to transport.
        if self._grasped:
            approach = 0.0
        else:
            approach = self.approach_scale * ee_box_dist
        transport = self.transport_scale * box_cup_dist_xy if self._grasped else 0.0
        return -(approach + transport)

    def _pregrasp_q(self, box_pos: np.ndarray) -> np.ndarray:
        """Solve a pre-grasp joint pose 9 cm above the marshmallow."""
        if (
            self._pregrasp_prev is not None
            and self._pregrasp_box is not None
            and np.array_equal(self._pregrasp_box, box_pos)
        ):
            return self._pregrasp_prev.copy()
        kin = Kinematics(self.model)
        mid = (
            self.model.jnt_range[self.arm_joint_ids, 0]
            + self.model.jnt_range[self.arm_joint_ids, 1]
        ) * 0.5
        mid[5] = self.open_q

        if self.use_pads:
            desired = box_pos + np.array(
                [0.0, 0.0, PAD_MIDPOINT_Z + self.pregrasp_height]
            )
        else:
            desired = box_pos + np.array([0.0, 0.0, self.pregrasp_height])

        guesses: list[np.ndarray | None] = []
        if self._pregrasp_prev is not None:
            guesses.append(self._pregrasp_prev.copy())
        guesses.append(mid.copy())
        q_wrist_minus = mid.copy()
        q_wrist_minus[4] = -1.0
        guesses.append(q_wrist_minus)
        q_wrist_plus = mid.copy()
        q_wrist_plus[4] = 1.0
        guesses.append(q_wrist_plus)

        best_q: np.ndarray | None = None
        best_score = np.inf
        best_pos_error = np.inf
        for q_init in guesses:
            target = desired.copy()
            q = None if q_init is None else q_init.copy()
            for _ in range(10):
                q = kin.solve(target, q_init=q)
                rot = kin.data.xmat[kin.gripper_id].reshape(3, 3)
                if self.use_pads:
                    next_target = desired - rot @ self.pads.site_offset
                else:
                    next_target = desired.copy()
                if np.linalg.norm(next_target - target) < 1e-6:
                    target = next_target
                    break
                target = next_target
            q = kin.solve(target, q_init=q)
            pos_error, dir_error = kin.pose_error(target, q)
            score = pos_error + dir_error
            if score < best_score:
                best_score = score
                best_q = q
                best_pos_error = pos_error

        if best_q is None or best_pos_error > 0.01:
            print(
                "warning: pre-grasp IK position error "
                f"{best_pos_error:.4f} m; falling back to neutral pose"
            )
            q_best = mid.copy()
        else:
            q_best = best_q.copy()
        q_best[5] = self.open_q
        self._pregrasp_prev = q_best.copy()
        self._pregrasp_box = box_pos.copy()
        return q_best

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self._apply_box_physics(rng)

        cup_pos = self.cup_base_pos.copy()
        # Curriculum: move cup closer to box for easier transport learning
        cup_pos[0] = (cup_pos[0] + 0.16) / 2.0  # halfway between original and box x
        cup_pos[1] = (cup_pos[1] + 0.11) / 2.0  # halfway between original and box y
        if self.randomize_cup_pos:
            cup_pos[0] += rng.uniform(-0.02, 0.02)
            cup_pos[1] += rng.uniform(-0.02, 0.02)
        self.data.qpos[self.cup_qpos_adr : self.cup_qpos_adr + 3] = cup_pos
        self.data.qpos[self.cup_qpos_adr + 3 : self.cup_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self._pin_cup()

        box_pos = np.array([0.16, 0.11, 0.0125])
        if self.randomize_box_pos:
            box_pos[0] += rng.uniform(-0.03, 0.03)
            box_pos[1] += rng.uniform(-0.03, 0.03)
        box_yaw = self.natural_box_yaw
        if self.randomize_box:
            box_yaw = self.natural_box_yaw + rng.uniform(-0.1, 0.1)  # ~±6 degrees around natural
        self._set_box(box_pos, box_yaw)
        self._box_init_pos = box_pos.copy()

        # Start from a neutral pose, or a pre-grasp pose above the box.
        if self.warm_start:
            q0 = self._pregrasp_q(box_pos)
        else:
            q0 = (
                self.model.jnt_range[self.arm_joint_ids, 0]
                + self.model.jnt_range[self.arm_joint_ids, 1]
            ) * 0.5
            q0[5] = self.open_q
        self.data.qpos[self.arm_qpos_adr] = q0
        self.data.ctrl[:6] = np.clip(q0, self.ctrl_low, self.ctrl_high)

        mujoco.mj_forward(self.model, self.data)
        self.cup_pos = self._get_cup_pos().copy()

        self._target = box_pos.copy()
        obs = self._get_obs()
        ee_pos = self.data.site_xpos[self.ee_site_id]
        box_pos_cur = self._get_box_pos()
        ee_box_dist = float(np.linalg.norm(ee_pos - box_pos_cur))
        pad_box_dist = float(np.linalg.norm(self._pad_midpoint() - box_pos_cur))
        box_cup_dist = float(np.linalg.norm(box_pos_cur - self.cup_pos))
        box_cup_dist_xy = float(
            np.linalg.norm(box_pos_cur[:2] - self.cup_pos[:2])
        )
        cup_upright = self._cup_is_upright()
        self._prev_phi = self._potential(pad_box_dist, box_cup_dist_xy)
        self._prev_phi_grasp = self._closure_potential()
        self._elapsed = 0
        self._lifted = False
        self._grasped = False
        self._ever_grasped = False

        info = {
            "ee_box_dist": ee_box_dist,
            "pad_box_dist": pad_box_dist,
            "box_cup_dist": box_cup_dist,
            "box_pos": box_pos_cur.copy(),
            "cup_pos": self.cup_pos.copy(),
            "cup_upright": cup_upright,
            "is_success": False,
            "is_failure": False,
            "grasped": self._grasped,
            "ever_grasped": self._ever_grasped,
            "phi": self._prev_phi,
            "phi_grasp": self._prev_phi_grasp,
            "gripper_q": float(self.data.qpos[self.arm_qpos_adr[5]]),
            "closed_at_box": False,
            "fixture_touch": False,
            "moving_touch": False,
            "both_touch": False,
            "gripper_action": 0.0,
        }
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,):
            action = action.reshape(6)
        action_clipped = np.clip(action, -1.0, 1.0)
        ctrl = self._action_to_ctrl(action_clipped)
        self.data.ctrl[:6] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._pin_cup()

        ee_pos = self.data.site_xpos[self.ee_site_id]
        box_pos = self._get_box_pos()
        self.cup_pos = self._get_cup_pos().copy()
        cup_upright = self._cup_is_upright()

        ee_box_dist = float(np.linalg.norm(ee_pos - box_pos))
        pad_box_dist = float(np.linalg.norm(self._pad_midpoint() - box_pos))
        box_cup_dist_xy = float(np.linalg.norm(box_pos[:2] - self.cup_pos[:2]))
        box_cup_dist = float(np.linalg.norm(box_pos - self.cup_pos))
        in_workspace = (
            -0.05 <= box_pos[0] <= 0.45
            and -0.35 <= box_pos[1] <= 0.35
            and -0.02 <= box_pos[2] <= 0.40
        )

        phi_new = self._potential(pad_box_dist, box_cup_dist_xy)
        reward = self.shaping_gamma * phi_new - self._prev_phi
        self._prev_phi = phi_new

        phi_grasp_new = self._closure_potential()
        reward += phi_grasp_new - self._prev_phi_grasp
        self._prev_phi_grasp = phi_grasp_new

        was_grasped = self._grasped  # pre-step grasp state

        fixture_touch, moving_touch = self._pads_contact_flags()
        self._grasped = fixture_touch and moving_touch

        # Velocity-based lift reward: pays for upward motion while grasped
        # AND the box is being held (low relative velocity).
        if was_grasped:
            box_dof = self.model.jnt_dofadr[self.box_joint_id]
            box_vel = self.data.qvel[box_dof:box_dof + 3]
            box_speed = float(np.linalg.norm(box_vel))
            box_vel_z = float(self.data.qvel[box_dof + 2])
            if box_vel_z > 0 and box_speed < 1.0:
                reward += self.lift_scale * box_vel_z

        # Velocity-based transport reward: pays for box moving toward cup
        # while grasped. Cannot be exploited by holding (no motion = no reward).
        if was_grasped:
            box_pos = self._get_box_pos()
            cup_pos = self.cup_pos
            box_to_cup = cup_pos[:2] - box_pos[:2]
            dist = float(np.linalg.norm(box_to_cup))
            if dist > 1e-6:
                # box velocity projected onto direction to cup
                box_dof = self.model.jnt_dofadr[self.box_joint_id]
                box_vel_xy = self.data.qvel[box_dof:box_dof + 2]
                direction = box_to_cup / dist
                vel_toward_cup = float(np.dot(box_vel_xy, direction))
                if vel_toward_cup > 0:
                    reward += self.transport_scale * vel_toward_cup


        if self._grasped and not self._ever_grasped:
            reward += self.grasp_bonus
            self._ever_grasped = True

        if box_pos[2] > 0.02 and self._grasped and not self._lifted:
            reward += self.lift_bonus
            self._lifted = True

        if self.task_mode == "lift_only":
            success = (
                in_workspace
                and self._grasped
                and self._lifted
            )
            failure = not in_workspace
        else:
            success = (
                cup_upright
                and box_cup_dist_xy < self.success_threshold_xy
                and 0.025 < box_pos[2] < self.success_threshold_z
            )
            if self.release_required:
                success = success and not self._grasped
            failure = (not in_workspace) or (not cup_upright)
        terminated = success or failure
        if success:
            reward += self.success_bonus
        elif failure:
            reward -= self.failure_penalty
        reward -= self.time_penalty

        self._elapsed += 1
        truncated = self._elapsed >= self.max_episode_steps

        obs = self._get_obs()
        info = {
            "ee_box_dist": ee_box_dist,
            "box_cup_dist_xy": box_cup_dist_xy,
            "box_cup_dist": box_cup_dist,
            "box_pos": box_pos.copy(),
            "cup_pos": self.cup_pos.copy(),
            "cup_upright": cup_upright,
            "in_workspace": in_workspace,
            "is_success": success,
            "is_failure": failure,
            "lifted": self._lifted,
            "grasped": self._grasped,
            "ever_grasped": self._ever_grasped,
            "phi": phi_new,
            "phi_grasp": phi_grasp_new,
            "pad_box_dist": pad_box_dist,
            "gripper_q": float(self.data.qpos[self.arm_qpos_adr[5]]),
            "closed_at_box": bool(
                pad_box_dist < 0.03
                and float(self.data.qpos[self.arm_qpos_adr[5]]) < self.close_q + 0.05
            ),
            "fixture_touch": bool(fixture_touch),
            "moving_touch": bool(moving_touch),
            "both_touch": bool(self._grasped),
            "gripper_action": float(action_clipped[5]),
        }
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        return None


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import gymnasium.utils.env_checker as env_checker
    import mira  # noqa: F401

    stage_configs: dict[int, dict[str, bool]] = {
        1: {
            "fix_cup": True,
            "randomize_box": False,
            "randomize_box_pos": False,
            "randomize_cup_pos": False,
            "randomize_physics": False,
        },
        2: {
            "fix_cup": True,
            "randomize_box": True,
            "randomize_box_pos": True,
            "randomize_cup_pos": False,
            "randomize_physics": False,
        },
        3: {
            "fix_cup": False,
            "randomize_box": True,
            "randomize_box_pos": True,
            "randomize_cup_pos": True,
            "randomize_physics": False,
        },
        4: {
            "fix_cup": False,
            "randomize_box": True,
            "randomize_box_pos": True,
            "randomize_cup_pos": True,
            "randomize_physics": True,
        },
    }

    default_env = SO101PickPlaceEnv()
    env_checker.check_env(default_env)
    print("check_env OK (default)")
    default_env.close()

    warm = SO101PickPlaceEnv(warm_start=True)
    env_checker.check_env(warm)
    print("check_env OK (warm_start=True)")

    for stage, config in stage_configs.items():
        stage_env = SO101PickPlaceEnv(**config)
        env_checker.check_env(stage_env)
        print(f"check_env OK (stage{stage})")
        stage_env.close()

    obs, info = warm.reset(seed=0)
    ee_pos = warm.data.site_xpos[warm.ee_site_id]
    ee_box_dist = float(np.linalg.norm(ee_pos - info["box_pos"]))
    print(f"warm-start EE-box distance at reset: {ee_box_dist:.4f} m")
    if ee_box_dist >= 0.15:
        print("FAIL: warm-start EE is more than 0.15 m from the box")
        warm.close()
        raise SystemExit(1)

    kin = Kinematics(warm.model)
    box_pos = info["box_pos"].copy()
    desired = box_pos + np.array([0.0, 0.0, PAD_MIDPOINT_Z])
    target = desired.copy()
    q_grasp: np.ndarray | None = None
    for _ in range(10):
        q_grasp = kin.solve(target, q_init=q_grasp)
        rot = kin.data.xmat[kin.gripper_id].reshape(3, 3)
        next_target = desired - rot @ warm.pads.site_offset
        if np.linalg.norm(next_target - target) < 1e-6:
            target = next_target
            break
        target = next_target
    q_grasp = kin.solve(target, q_init=q_grasp)
    q_pre = kin.solve(
        target + np.array([0.0, 0.0, 0.09]), q_init=q_grasp
    )
    q_lift = kin.solve(
        target + np.array([0.0, 0.0, 0.075]), q_init=q_grasp
    )

    ctrl_low = warm.ctrl_low
    ctrl_high = warm.ctrl_high

    def to_action(q: np.ndarray, grip: float) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).copy()
        q[5] = grip
        q = np.clip(q, ctrl_low, ctrl_high)
        action = 2.0 * (q - ctrl_low) / (ctrl_high - ctrl_low) - 1.0
        # Match SO101PickPlaceEnv.step: the gripper action spans
        # [close_q, open_q], not the full joint range.
        action[5] = 2.0 * (q[5] - warm.close_q) / (warm.open_q - warm.close_q) - 1.0
        return action.astype(np.float32)

    state = {
        "total": 0.0,
        "shaping": 0.0,
        "grasp": 0.0,
        "prev_phi": float(info["phi"]),
        "prev_ever": bool(info["ever_grasped"]),
        "box_z": float(info["box_pos"][2]),
        "ever": bool(info["ever_grasped"]),
        "done": False,
    }

    def ramp(
        q_from: np.ndarray,
        q_to: np.ndarray,
        grip_from: float,
        grip_to: float,
        steps: int,
    ) -> None:
        if state["done"]:
            return
        for index in range(steps):
            t = (index + 1) / steps
            q = q_from + t * (q_to - q_from)
            grip = grip_from + t * (grip_to - grip_from)
            _, reward, terminated, truncated, step_info = warm.step(
                to_action(q, grip)
            )
            state["total"] += reward
            state["shaping"] += (
                warm.shaping_gamma * step_info["phi"] - state["prev_phi"]
            )
            state["prev_phi"] = step_info["phi"]
            if step_info["ever_grasped"] and not state["prev_ever"]:
                state["grasp"] += warm.grasp_bonus
                state["prev_ever"] = True
            state["box_z"] = float(step_info["box_pos"][2])
            state["ever"] = bool(step_info["ever_grasped"])
            if terminated or truncated:
                state["done"] = True
                break

    q_start = warm.data.qpos[warm.arm_qpos_adr].copy()
    q_pre = np.asarray(q_pre, dtype=np.float64)
    q_grasp = np.asarray(q_grasp, dtype=np.float64)
    q_lift = np.asarray(q_lift, dtype=np.float64)

    ramp(q_start, q_pre, warm.open_q, warm.open_q, 120)
    ramp(q_pre, q_grasp, warm.open_q, warm.open_q, 120)
    ramp(q_grasp, q_grasp, warm.open_q, warm.close_q, 100)
    ramp(q_grasp, q_lift, warm.close_q, warm.close_q, 120)

    rise = state["box_z"] - 0.0125
    other_reward = state["total"] - state["shaping"] - state["grasp"]
    print(f"ever_grasped={state['ever']}")
    print(f"final box z={state['box_z']:.4f} m (rise {rise * 100:+.2f} cm)")
    print(f"total accumulated reward={state['total']:.3f}")
    print(
        "reward breakdown: "
        f"shaping={state['shaping']:.3f}, "
        f"grasp={state['grasp']:.3f}, "
        f"lift/success/failure/time={other_reward:.3f}"
    )

    passed = state["ever"] and rise > 0.03
    print("PASS" if passed else "FAIL")
    warm.close()
    raise SystemExit(0 if passed else 1)
