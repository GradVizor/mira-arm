# MIRA — Multimodal Intelligent Robotic Arm

Personal repository for a robotic arm based on the **SO-ARM101 (SO-101)** 6-DOF arm. MIRA is a platform for **multimodal (vision + language + action) robotics research**: simulated MuJoCo models, headless rendering, an interactive viewer, and a clean base for teleop data collection, RL training, and VLA policies.

## Repository layout

| Path | Contents |
|---|---|
| `models/so101/` | MuJoCo (MJCF) model: `scene.xml`, calibration XMLs, `joints_properties.xml`, and `assets/` (13 STL meshes). XML + STL assets must stay together (`meshdir="assets"` is relative). |
| `scripts/` | `sim.py` (headless sinusoidal demo → PNGs + GIF into `outputs/`), `view.py` (interactive X11 viewer), `watch_gym.py` (MuJoCo viewer driven by the Gym env), `train_reach.py` (PPO stub), `train_tabular.py` (Phase 2 tabular Q-learning / SARSA CLI). |
| `mira/` | Gymnasium RL package — `SO101ReachEnv` (env id `MIRA/SO101Reach-v0`) + optional observation/action wrappers. See `docs/GYMNASIUM_ENV.md`. `mira/rl/` holds the from-scratch tabular TD learners (numpy only). |
| `docs/` | `RL_ROADMAP.md` (zero→VLA learning path), `GYMNASIUM_ENV.md` (Gymnasium interface + flowcharts), `TABULAR_RL.md` (TD-learning study guide). |
| `test/` | Script-style self-checks: `test_gym.py` (env smoke test) and `test_tabular_rl.py` (13 checks for the Phase 2 learners). No pytest needed. |
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

Both scenes support two differently calibrated SO-101 files:

- **New calibration (default)** — each joint's virtual zero is the **middle** of its joint range → `so101_new_calib.xml`.
- **Old calibration** — each joint's virtual zero is the configuration where the robot is **fully extended horizontally** → `so101_old_calib.xml`.

Switch by editing the `<include file="..." />` line in the scene XML you want to use.

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

## Tabular RL (Phase 2)

`mira/rl/tabular.py` is a **from-scratch** Q-learning + SARSA implementation — plain numpy and the
Gymnasium API, no RL library. It exists to make the TD-learning mechanics concrete before a neural
policy ever touches the arm.

```bash
python scripts/train_tabular.py --env FrozenLake-v1 --compare-optimal   # + exact value-iteration reference
python scripts/train_tabular.py --env CliffWalking-v1 --algo sarsa      # on-policy variant
python scripts/train_tabular.py --env FrozenLake-v1 --seeds 0 1 2        # multi-seed
python test/test_tabular_rl.py                                          # 13 self-checks
```

- **Q-learning (off-policy)** on `CliffWalking-v1` finds the 13-step optimal path (−13, 100%).
- **SARSA (on-policy)** takes the safe 17-step route (−17, 100%) — it learns the risk it actually
  faces while exploring.
- On slippery `FrozenLake-v1` the ceiling for *any* deterministic policy is **0.733**; the shipped
  defaults reach 0.716–0.730 across seeds, and `--compare-optimal` proves the ceiling by solving the
  MDP exactly.
- The same code shows why tabular RL cannot scale to the arm: a 6-joint discretisation at 100
  bins/joint is 10¹² states (≈ 48 TB of Q-table). The script prints that arithmetic.

The theory, the derivations, the checkpoint answers, and the lab notebook live in
[`docs/TABULAR_RL.md`](docs/TABULAR_RL.md).

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
python test/test_tabular_rl.py  # Phase 2 self-checks (13 checks)
```