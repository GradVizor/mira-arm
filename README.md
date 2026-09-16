# MIRA — Multimodal Intelligent Robotic Arm

Personal repository for a robotic arm based on the **SO-ARM101 (SO-101)** 6-DOF arm. MIRA is a platform for **multimodal (vision + language + action) robotics research**: simulated MuJoCo models, headless rendering, an interactive viewer, and a clean base for teleop data collection, RL training, and VLA policies.

## Repository layout

| Path | Contents |
|---|---|
| `models/so101/` | MuJoCo (MJCF) model: `scene.xml`, calibration XMLs, `joints_properties.xml`, and `assets/` (13 STL meshes). XML + STL assets must stay together (`meshdir="assets"` is relative). |
| `scripts/` | `sim.py` (headless sinusoidal demo → PNGs + GIF into `outputs/`) and `view.py` (interactive X11 viewer). |
| `Dockerfile` + `entrypoint.sh` + `.dockerignore` | Container image for the sim + viewer (lean `python:3.12-slim`, EGL/osmesa/GLFW rendering libs). |
| `docker-compose.yaml` / `docker-compose.gpu.yaml` | CPU / NVIDIA-GPU container config; `mira-arm` service (see [Docker](#docker)). |
| `outputs/` | Generated render artifacts (gitignored, regenerable). |

## Hardware

- **SO-ARM101 (SO-101)** 6-DOF arm (MJCF joint names): `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- **STS3215** servos throughout.
- Models generated via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) from an Onshape CAD design; base collision meshes removed (problematic collision behavior during simulation/planning).

## Calibration

`scene.xml` supports two differently calibrated SO-101 files:

- **New calibration (default)** — each joint's virtual zero is the **middle** of its joint range → `so101_new_calib.xml`.
- **Old calibration** — each joint's virtual zero is the configuration where the robot is **fully extended horizontally** → `so101_old_calib.xml`.

Switch by editing the `<include file="..." />` line in `models/so101/scene.xml`.

## Motor parameters

STS3215 position servos, motor properties adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

- **Active model (`so101_new_calib.xml`)** — `kp=998.22`, `kv=2.731` (gains per the [RBE501 RL-arm project](https://github.com/Gregory119/RBE501-RL-arm-project), assuming servo proportional gain 16); joint `damping=0.60`, `frictionloss=0.052`, `armature=0.028`. The `sts3215` class default `forcerange` is ±2.94, but every actuator overrides it to `-3.35 3.35` (also its `ctrlrange`).
- **Old calibration (`so101_old_calib.xml`)** — `kp=17.8`.
- `joints_properties.xml` ships the `kp=17.8` `sts3215` + `backlash` (±0.5°) classes but is **not** `<include>`d anywhere — the calibration XMLs inline their own copies.

## Gripper note

In LeRobot the gripper is a **linear joint**: `0` = fully closed, `100` = fully open. This mapping is **not yet reflected** in the MJCF — it is modeled as a hinge `gripper` joint (range ≈ −10°…100°) on the `moving_jaw_so101_v1` body.

## Quick start

```bash
pip install -r requirements.txt
python scripts/sim.py                        # headless demo → outputs/ (PNGs + GIF)
python scripts/sim.py --duration 10 --no-gif # 10 s sim, skip GIF
python scripts/sim.py --out /tmp/out         # custom output dir
python scripts/view.py                       # interactive X11 viewer (auto demo)
python scripts/view.py --manual              # drag sliders in the left panel
python scripts/view.py --max-time 30         # auto-close after 30 s
```

The X11 viewer needs a display (X11/GLFW).

## Docker

```bash
docker compose up -d                                        # CPU-only host (osmesa software rendering)
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d   # NVIDIA GPU host
docker compose exec mira-arm bash                           # interactive shell in the container
python scripts/sim.py                                       # inside the container → outputs/ on the host
```

- `docker-compose.gpu.yaml` is an override, not a standalone file — it requires the NVIDIA Container Toolkit on the host (`gpus: all`).
- The render backend is chosen by the scripts, not the container: `sim.py` picks EGL when an NVIDIA GPU is visible and osmesa elsewhere; `view.py` opens a GLFW window when `MUJOCO_GL` is unset. Set `MUJOCO_GL` to force a backend.
- `entrypoint.sh` smoke-checks the stack, starts a virtual display when `XVFB_RUN=1`, and keeps bind-mounted files owned by your host user (`HOST_UID`/`HOST_GID`).
- The interactive viewer inside the container needs a host X server: `xhost +local:root`, then `docker compose exec mira-arm python scripts/view.py`.

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