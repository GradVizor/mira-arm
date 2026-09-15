# MIRA — Multimodal Intelligent Robotic Arm

Personal repository for a robotic arm based on the **SO-ARM101 (SO-101)** 6-DOF arm. MIRA is a platform for **multimodal (vision + language + action) robotics research**: simulated MuJoCo models, headless rendering, an interactive viewer, and a clean base for teleop data collection, RL training, and VLA policies.

## Repository layout

| Path | Contents |
|---|---|
| `models/so101/` | MuJoCo (MJCF) model: `scene.xml`, calibration XMLs, `joints_properties.xml`, and `assets/` (13 STL meshes). XML + STL assets must stay together (`meshdir="assets"` is relative). |
| `scripts/` | `sim.py` (headless sinusoidal demo → PNGs + GIF into `outputs/`) and `view.py` (interactive X11 viewer). |
| `outputs/` | Generated render artifacts (gitignored, regenerable). |

## Hardware

- **SO-ARM101 (SO-101)** 6-DOF arm: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_roll`, `wrist_pitch`, `gripper`.
- **STS3215** servos throughout.
- Models generated via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) from an Onshape CAD design; base collision meshes removed (problematic collision behavior during simulation/planning).

## Calibration

`scene.xml` supports two differently calibrated SO-101 files:

- **New calibration (default)** — each joint's virtual zero is the **middle** of its joint range → `so101_new_calib.xml`.
- **Old calibration** — each joint's virtual zero is the configuration where the robot is **fully extended horizontally** → `so101_old_calib.xml`.

Switch by editing the `<include file="..." />` line in `models/so101/scene.xml`.

## Motor parameters

STS3215 motor properties adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini); gains per the RBE501 RL-arm project reference (`kp=998.22`, `kv=2.731`, `forcerange ±2.94`).

## Gripper note

In LeRobot the gripper is a **linear joint**: `0` = fully closed, `100` = fully open. This mapping is **not yet reflected** in the MJCF.

## Quick start

```bash
pip install -r requirements.txt
python scripts/sim.py            # headless demo → renders PNGs + GIF into outputs/
python scripts/view.py           # interactive X11 viewer (auto demo)
python scripts/view.py --manual  # drag sliders in the left panel
```

The X11 viewer needs a display on the host — run `xhost +local:root` first if using Docker.

## Running inside the MCP_Test dev container

The MuJoCo RL/VLA Docker environment (`/home/gradvizor/workspace/openCodeDemo`) provides the Python env, GPU, and MuJoCo. Mount this repo and run:

```bash
docker compose -f /home/gradvizor/workspace/openCodeDemo/docker-compose.yaml run --rm \
  -v /home/gradvizor/workspace/mira-arm:/mira-arm \
  mujoco-rl python /mira-arm/scripts/sim.py

docker compose -f /home/gradvizor/workspace/openCodeDemo/docker-compose.yaml run --rm \
  -v /home/gradvizor/workspace/mira-arm:/mira-arm \
  mujoco-rl python /mira-arm/scripts/view.py
```

## Roadmap

- Teleop data collection
- RL training
- VLA policies (vision + language + action)
- Real-hardware bring-up

## Verification / smoke test

```bash
python -m py_compile scripts/sim.py scripts/view.py
python scripts/sim.py   # must complete with non-blank frames (mid-frame mean pixel check)
```