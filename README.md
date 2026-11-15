# MIRA — Multimodal Intelligent Robotic Arm

Personal repository for a robotic arm based on the **SO-ARM101 (SO-101)** 6-DOF arm. MIRA is a platform for **multimodal (vision + language + action) robotics research**: simulated MuJoCo models, headless rendering, an interactive viewer, and a clean base for teleop data collection, RL training, and VLA policies.

## Repository layout

| Path | Contents |
|---|---|
| `models/so101/` | MuJoCo (MJCF) model: `scene.xml`, calibration XMLs, and `assets/` (13 STL meshes). XML + STL assets must stay together (`meshdir="assets"` is relative). |
| `scripts/` | `sim.py` (headless sinusoidal demo), `view.py` (interactive X11 viewer), `watch_gym.py` (MuJoCo viewer driven by the Gym env), `train_pick_place.py` (SAC pick-and-place trainer), `oracle_pick_place.py` (scripted IK oracle), `watch_pick_place.py` (MuJoCo viewer for trained pick-and-place model), `test_grasp.py` (grasp-physics acceptance test). |
| `mira/` | Gymnasium RL package — `SO101ReachEnv` (`MIRA/SO101Reach-v0`), `SO101PointToPointEnv` (`MIRA/SO101PointToPoint-v0`), `SO101PickPlaceEnv` (`MIRA/SO101PickPlace-v0`) + observation/action wrappers. See `docs/GYMNASIUM_ENV.md`. |
| `docs/` | `RL_ROADMAP.md` (zero→VLA learning path), `GYMNASIUM_ENV.md` (Gymnasium interface + flowcharts), `TABULAR_RL.md` (TD-learning study guide, Phase 2 write-up). |
| `test/` | Script-style self-checks: `test_gym.py` (env smoke test). No pytest needed. |
| `Dockerfile` + `entrypoint.sh` + `.dockerignore` | Container image for the sim + viewer (lean `python:3.12-slim`, EGL/osmesa/GLFW rendering libs). |
| `docker-compose.yaml` / `docker-compose.gpu.yaml` | CPU / NVIDIA-GPU container config; `mira-arm` service (see [Docker](#docker)). |
| `outputs/` | Generated render artifacts (gitignored, regenerable). |

## Hardware

- **SO-ARM101 (SO-101)** 6-DOF arm (MJCF joint names): `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- **STS3215** servos throughout.
- Models generated via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) from an Onshape CAD design; base collision meshes removed (problematic collision behavior during simulation/planning).

## Scenes

Two scene files are available in `models/so101/`:
- `scene.xml` — clean arm-only scene (floor, lights, no props). Used by default across scripts and the gym env.
- `scene_with_coffee_mug.xml` — same arm setup plus a coffee mug and a marshmallow cube (dynamic bodies with freejoints) for pick-and-place experiments.

You can select a scene via CLI flags/env vars:
- `scripts/view.py --scene scene_with_coffee_mug.xml`
- `scripts/sim.py --scene scene_with_coffee_mug.xml`
- `MIRA_SCENE=scene_with_coffee_mug.xml python test/test_gym.py` (or pass `xml_path` to `gym.make`)

## Calibration

The scene includes a single calibrated SO-101 file:

- **Calibration (default)** — each joint's virtual zero is the **middle** of its joint range → `so101_calib.xml`.

Switch scenes/calibration by editing the `<include file="..." />` line in the scene XML you want to use.

## Motor parameters

STS3215 position servos, motor properties adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

- **Active model (`so101_calib.xml`)** — `kp=998.22`, `kv=2.731` (gains per the [RBE501 RL-arm project](https://github.com/Gregory119/RBE501-RL-arm-project), assuming servo proportional gain 16); joint `damping=0.60`, `frictionloss=0.052`, `armature=0.028`. The `sts3215` class default `forcerange` is ±2.94, but every actuator overrides it to `-3.35 3.35` (also its `ctrlrange`).

## Gripper note

In LeRobot the gripper is a **linear joint**: `0` = fully closed, `100` = fully open. This mapping is **not yet reflected** in the MJCF — it is modeled as a hinge `gripper` joint (range ≈ −10°…100°) on the `moving_jaw_so101_v1` body.

## Quick start

```bash
pip install -r requirements.txt
python scripts/sim.py                        # headless demo → outputs/ (PNGs + GIF) — clean scene
python scripts/sim.py --scene scene_with_coffee_mug.xml  # demo with coffee mug + marshmallow
python scripts/sim.py --duration 10 --no-gif # 10 s sim, skip GIF
python scripts/sim.py --out /tmp/out         # custom output dir
python scripts/view.py                       # interactive X11 viewer (auto demo) — clean scene
python scripts/view.py --scene scene_with_coffee_mug.xml  # viewer with props
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

## Gymnasium interface (RL)

`mira/` wraps the MuJoCo model in the standard **Gymnasium** API so any RL library can train on the arm without knowing MuJoCo. Gymnasium sits *between* the simulator and a trainer:

```
models/so101/scene.xml  →  MuJoCo (MjModel/MjData)  →  SO101ReachEnv (Gymnasium)  →  wrappers  →  register "MIRA/SO101Reach-v0"  →  SB3 / CleanRL / eval
```

```python
import gymnasium as gym
import mira  # registers "MIRA/SO101Reach-v0"

env = gym.make("MIRA/SO101Reach-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Run the built-in smoke test (env checker + 10k-step random rollout):

```bash
pip install -r requirements.txt
python3 mira/envs/so101_reach_env.py
```

The full contract, the wrappers, and two annotated flowcharts (where Gymnasium fits, and one episode end-to-end) live in [`docs/GYMNASIUM_ENV.md`](docs/GYMNASIUM_ENV.md). RL deps (torch / Stable-Baselines3) are intentionally **not** in the lean sim image — add them via a separate `requirements-rl.txt` when training.

## Pick and place (SO-101)

The `MIRA/SO101PickPlace-v0` env tasks the arm with grasping a marshmallow cube and dropping it into a coffee cup. It uses flat gripper pads (built at runtime via `MjSpec`) with a reduced force limit (0.5 N·m vs the stock 3.35 N·m) to make the 4.7 g marshmallow graspable.

```bash
# Validate grasp physics (exit 0 = PASS)
python3 scripts/test_grasp.py

# Scripted oracle (exit 0 = task solvable)
python3 scripts/oracle_pick_place.py
python3 scripts/oracle_pick_place.py --viewer   # live window

# Train (lift-only bootstrap, then full task)
python3 scripts/train_pick_place.py --task-mode lift_only --stage 1 --total-timesteps 10000 --warm-start --pregrasp-height 0.0
python3 scripts/train_pick_place.py --task-mode full --stage 1 --total-timesteps 10000 --warm-start --pregrasp-height 0.0 --init-model outputs/sac_pick_place/stage1/final_model.zip

# Watch a trained model
python3 scripts/watch_pick_place.py
python3 scripts/watch_pick_place.py --task-mode full
```

Key findings (see `docs/PICK_PLACE_DEBUG_LOG.md`):
- Grasp requires the natural grasp yaw (~40.87°); an axis-aligned box cannot be grasped top-down with the stock finger mesh.
- Closure potential must be gated on pad proximity; entropy collapse is the main failure mode.
- Lift shaping (velocity-based, gated on held contact) is needed to avoid pushing the box.

## Tabular RL (Phase 2 — archived)

Phase 2 is complete. The from-scratch Q-learning + SARSA code that carried it (tabular learner,
CLI, 13 self-checks) has been removed: a Q-table is structurally incapable of representing this
arm (6 joints at 100 bins = 10¹² states ≈ 48 TB), so none of it was reachable from the training
path. Recover it from git history (`git show 8174a96`) if you want the reference implementation.

The write-up is kept — it is the part that transfers:

- **Q-learning (off-policy)** on `CliffWalking-v1` finds the 13-step optimal path (−13).
- **SARSA (on-policy)** takes the safe 17-step route (−17) — it learns the risk it actually faces.
- On slippery `FrozenLake-v1` the ceiling for *any* deterministic policy is **0.733**; tuning
  landed at 0.716–0.730 across seeds.

Derivations, checkpoint answers, and the lab notebook: [`docs/TABULAR_RL.md`](docs/TABULAR_RL.md).

## Roadmap

- Teleop data collection
- RL training
- VLA policies (vision + language + action)
- Real-hardware bring-up

## Verification / smoke test

```bash
python -m py_compile scripts/sim.py scripts/view.py
python scripts/sim.py           # must complete with non-blank frames (mid-frame mean pixel check)
python test/test_gym.py         # Gymnasium env smoke test
```