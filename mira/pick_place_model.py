"""Build the SO-101 pick-and-place model with flat grasp pads.

The generated calibration XML has tapered finger meshes.  They contact a small
box at its top corners and squeeze it into the floor, so the box slips out when
the arm lifts.  This module leaves the generated XML untouched and augments the
compiled model at runtime:

* original ``gripper`` / ``moving_jaw_so101_v1`` mesh collisions are disabled
  (their visuals remain),
* a 2 mm thick, 12 mm wide flat pad is added to each finger's inner face,
* the gripper actuator force range is capped, because the model's 3.35 N.m max
  ejects a 4.7 g marshmallow.

Pad placement is measured from the actual finger mesh vertices at ``open_q`` so
the helper follows the generated calibration instead of hard-coding finger poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "models" / "so101" / "scene_with_coffee_mug.xml"

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

#: Finger mesh vertices below this gripper-frame z are finger tips, not the palm.
FINGER_Z_MIN = -0.108
FINGER_Z_MAX = -0.0805

PAD_THICKNESS = 0.001  # half-thickness along the opening axis
PAD_WIDTH = 0.006  # half-width perpendicular to the opening axis
PAD_FRICTION = (2.0, 0.1, 0.1)


@dataclass(frozen=True)
class PadGeometry:
    """Pad poses plus the pad-midpoint offset from the gripper site."""

    fixture_pos: np.ndarray
    fixture_size: np.ndarray
    moving_pos: np.ndarray
    moving_size: np.ndarray
    moving_quat: np.ndarray
    site_offset: np.ndarray


def _mesh_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    frame_body_id: int,
) -> np.ndarray:
    """Return finger mesh vertices in ``frame_body_id``'s frame."""
    frame_pos = data.xpos[frame_body_id].copy()
    frame_rot = data.xmat[frame_body_id].reshape(3, 3)
    clouds: list[np.ndarray] = []

    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[geom_id] == 0:
            continue

        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        verts = model.mesh_vert[start : start + count]
        geom_rot = data.geom_xmat[geom_id].reshape(3, 3)
        world = data.geom_xpos[geom_id] + verts @ geom_rot.T
        clouds.append((world - frame_pos) @ frame_rot)

    if not clouds:
        raise RuntimeError(f"No mesh vertices found for body id {body_id}")

    points = np.vstack(clouds)
    return points[(points[:, 2] >= FINGER_Z_MIN) & (points[:, 2] <= FINGER_Z_MAX)]


def _pad_pose(points: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre and size of a pad on one finger's inner face."""
    x = points[:, 0].max() if side < 0 else points[:, 0].min()
    y = 0.5 * (points[:, 1].min() + points[:, 1].max())
    z = 0.5 * (points[:, 2].min() + points[:, 2].max())
    half_z = 0.5 * (points[:, 2].max() - points[:, 2].min()) + 1e-4
    pos = np.array([x, y, z], dtype=np.float64)
    size = np.array([PAD_THICKNESS, PAD_WIDTH, half_z], dtype=np.float64)
    return pos, size


def _add_pad(
    body,
    name: str,
    pos: np.ndarray,
    size: np.ndarray,
    quat: np.ndarray,
) -> None:
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = size.tolist()
    geom.pos = pos.tolist()
    geom.quat = quat.tolist()
    geom.friction = list(PAD_FRICTION)
    geom.contype = 1
    geom.conaffinity = 1
    geom.rgba = [0.9, 0.2, 0.2, 1.0]
    geom.group = 1


def _site_offset(
    model: mujoco.MjModel,
    fixture_pad_id: int,
    moving_pad_id: int,
    gripper_id: int,
    site_id: int,
    open_q: float,
) -> np.ndarray:
    """Pad midpoint minus the site, expressed in the gripper frame."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    data.qpos[model.jnt_qposadr[gripper_joint_id]] = open_q
    mujoco.mj_forward(model, data)

    gripper_pos = data.xpos[gripper_id].copy()
    gripper_rot = data.xmat[gripper_id].reshape(3, 3)
    fixture = gripper_rot.T @ (data.geom_xpos[fixture_pad_id] - gripper_pos)
    moving = gripper_rot.T @ (data.geom_xpos[moving_pad_id] - gripper_pos)
    midpoint = 0.5 * (fixture + moving)
    return midpoint - model.site_pos[site_id]


def build_padded_model(
    scene_path: str | Path = DEFAULT_SCENE,
    *,
    open_q: float = 0.25,
    force_limit: float = 0.5,
) -> tuple[mujoco.MjModel, PadGeometry]:
    """Load ``scene_path``, add grasp pads, and return the compiled model.

    Args:
        scene_path: MJCF scene to load.
        open_q: Gripper joint angle at which the pad poses are measured.
        force_limit: Symmetric gripper actuator force limit in N.m.
    """
    scene_path = Path(scene_path)
    spec = mujoco.MjSpec.from_file(str(scene_path))

    # Compile once to measure the generated finger meshes at the open angle.
    probe_model = spec.compile()
    probe_data = mujoco.MjData(probe_model)
    gripper_joint_id = mujoco.mj_name2id(
        probe_model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
    )
    mujoco.mj_resetData(probe_model, probe_data)
    probe_data.qpos[probe_model.jnt_qposadr[gripper_joint_id]] = open_q
    mujoco.mj_forward(probe_model, probe_data)

    gripper_id = mujoco.mj_name2id(
        probe_model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
    )
    jaw_id = mujoco.mj_name2id(
        probe_model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"
    )
    if gripper_id < 0 or jaw_id < 0:
        raise RuntimeError("Expected 'gripper' and 'moving_jaw_so101_v1' bodies")

    fixture_cloud = _mesh_points(
        probe_model, probe_data, gripper_id, gripper_id
    )
    moving_cloud = _mesh_points(probe_model, probe_data, jaw_id, gripper_id)

    fixture_pos, fixture_size = _pad_pose(fixture_cloud, side=-1)
    moving_pos_g, moving_size = _pad_pose(moving_cloud, side=+1)

    gripper_pos = probe_data.xpos[gripper_id].copy()
    gripper_rot = probe_data.xmat[gripper_id].reshape(3, 3)
    jaw_pos = probe_data.xpos[jaw_id].copy()
    jaw_rot = probe_data.xmat[jaw_id].reshape(3, 3)
    moving_pos = jaw_rot.T @ (
        gripper_pos + gripper_rot @ moving_pos_g - jaw_pos
    )
    moving_rot = jaw_rot.T @ gripper_rot
    moving_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(moving_quat, moving_rot.reshape(-1))

    # The tapered finger meshes must not compete with the flat pads.  Visuals
    # stay; collision is handled by the pads only.
    for body_name in ("gripper", "moving_jaw_so101_v1"):
        body = next(b for b in spec.bodies if b.name == body_name)
        for geom in body.geoms:
            geom.contype = 0
            geom.conaffinity = 0

    fixture_body = next(b for b in spec.bodies if b.name == "gripper")
    moving_body = next(
        b for b in spec.bodies if b.name == "moving_jaw_so101_v1"
    )
    _add_pad(
        fixture_body,
        "fixture_pad",
        fixture_pos,
        fixture_size,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    _add_pad(
        moving_body,
        "moving_pad",
        moving_pos,
        moving_size,
        moving_quat,
    )

    model = spec.compile()

    if force_limit > 0.0:
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        if actuator_id < 0:
            raise RuntimeError("Gripper actuator not found")
        model.actuator_forcerange[actuator_id] = [-force_limit, force_limit]

    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
    )
    final_gripper_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
    )
    fixture_pad_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "fixture_pad"
    )
    moving_pad_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "moving_pad"
    )
    offset = _site_offset(
        model,
        fixture_pad_id,
        moving_pad_id,
        final_gripper_id,
        site_id,
        open_q,
    )

    pads = PadGeometry(
        fixture_pos=fixture_pos,
        fixture_size=fixture_size,
        moving_pos=moving_pos,
        moving_size=moving_size,
        moving_quat=moving_quat,
        site_offset=offset,
    )
    return model, pads


if __name__ == "__main__":
    model, pads = build_padded_model()
    print(f"fixture_pad: pos={pads.fixture_pos} size={pads.fixture_size}")
    print(f"moving_pad:  pos={pads.moving_pos} size={pads.moving_size}")
    print(f"site_offset (gripper frame): {pads.site_offset}")
    print(f"gripper forcerange: {model.actuator_forcerange[-1]}")
