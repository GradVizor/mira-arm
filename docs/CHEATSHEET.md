# MIRA / SO-101 — Train & Watch Cheat Sheet

Every command below is run from the **repository root** (`mira-arm/`).
All scripts derive paths from their own location, so they also work from
elsewhere, but the examples assume the root for brevity.

---

## 0. Setup

```bash
# Base sim + Gymnasium deps (mujoco, numpy, pillow, gymnasium)
pip install -r requirements.txt

# RL deps only (torch, Stable-Baselines3, sb3-contrib, tensorboard)
pip install -r requirements-rl.txt
```

Quick syntax sanity:

```bash
python3 -m py_compile scripts/*.py
bash -n entrypoint.sh
```

---

## 1. View the arm in MuJoCo (no training)

### 1a. Interactive viewer — sinusoidal demo

```bash
python3 scripts/view.py                 # auto-drive, opens a GLFW window
python3 scripts/view.py --manual        # drag joint sliders yourself
python3 scripts/view.py --max-time 10   # auto-close after 10 s
python3 scripts/view.py --scene scene_with_coffee_mug.xml
python3 scripts/view.py --scene scene_with_markers.xml
```

### 1b. Headless render — PNGs + GIF

```bash
python3 scripts/sim.py                              # -> outputs/ (PNGs + output_demo.gif)
python3 scripts/sim.py --duration 6 --out /tmp/out
python3 scripts/sim.py --no-gif                     # frames + pose PNGs only
python3 scripts/sim.py --scene scene_with_coffee_mug.xml
```

Outputs: `outputs/frame_NNNN.png`, `outputs/pose_{start,mid,end}.png`,
`outputs/output_demo.gif`.

### 1c. Watch a Gymnasium env with random actions

```bash
python3 scripts/watch_gym.py                                   # MIRA/SO101Reach-v0, 200 steps
MIRA_SCENE=scene_with_coffee_mug.xml python3 scripts/watch_gym.py
```

`watch_gym.py` always sets `MUJOCO_GL=osmesa` unless already set — it is
**headless-first**; use `scripts/view.py` for a windowed demo.

### 1d. Environment smoke checks

```bash
python3 test/test_gym.py                        # MIRA/SO101Reach-v0 reset + 10 steps
python3 mira/envs/so101_reach_env.py            # env checker + random rollout
python3 -m mira.envs.so101_pick_place_env       # pick-place env checker + rollouts
```

---

## 2. Point-to-point (PPO, marker scene)

Task: move the end effector from fixed point A to fixed point B
(`scene_with_markers.xml`).

### 2a. Train

```bash
python3 scripts/train_point_to_point.py
# defaults: 500k timesteps, 4 envs, 2048 steps/env

# shorter/faster run
python3 scripts/train_point_to_point.py --total-timesteps 100000

# useful knobs
python3 scripts/train_point_to_point.py \
  --total-timesteps 500000 --num-envs 4 --lr 3e-4 --seed 0 \
  --eval-every 20 --eval-episodes 10
```

Saves:

| Artifact | Path |
|---|---|
| Policy weights | `outputs/ppo_point_to_point.pt` |
| Obs-normalizer stats | `outputs/ppo_point_to_point_norm.npz` |

If `outputs/` is not writable it falls back to `/tmp/opencode/`.

### 2b. Watch / evaluate

```bash
python3 scripts/watch_point_to_point.py                      # 5 episodes, mean actions
python3 scripts/watch_point_to_point.py --episodes 10
python3 scripts/watch_point_to_point.py --episodes 0         # until you close the window
python3 scripts/watch_point_to_point.py --stochastic         # sample instead of mean
python3 scripts/watch_point_to_point.py --no-viewer          # headless, print metrics only
python3 scripts/watch_point_to_point.py --model outputs/ppo_point_to_point.pt \
  --norm-stats outputs/ppo_point_to_point_norm.npz
```

---

## 3. Pick-and-place (SAC, marshmallow → cup)

### 3a. Validate the physics first

```bash
# Step-0 grasp test: exit 0 = PASS, 1 = FAIL
python3 scripts/test_grasp.py
python3 scripts/test_grasp.py --viewer                # live playback (needs display)

# Scripted oracle: proves the task is solvable with IK
python3 scripts/oracle_pick_place.py                  # exit 0 = task solvable
python3 scripts/oracle_pick_place.py --viewer         # live window
python3 scripts/oracle_pick_place.py --verbose        # trace every 50 steps
python3 scripts/oracle_pick_place.py --out /tmp/oracle.json
```

### 3b. Train

```bash
# Single stage (default: stage 1, 200k steps)
python3 scripts/train_pick_place.py
python3 scripts/train_pick_place.py --stage 1 --total-timesteps 100000

# Recommended bootstrap: grasp + lift only, warm-started at the grasp pose
python3 scripts/train_pick_place.py \
  --task-mode lift_only --stage 1 --total-timesteps 10000 \
  --warm-start --pregrasp-height 0.0

# Continue into the full pick-and-place task from the lift-only checkpoint
python3 scripts/train_pick_place.py \
  --task-mode full --stage 1 --total-timesteps 10000 \
  --warm-start --pregrasp-height 0.0 \
  --init-model outputs/sac_pick_place/stage1/final_model.zip

# Stage-1 discovery curriculum (0.0 -> 0.03 -> 0.09 m pre-grasp height)
python3 scripts/train_pick_place.py --stage1-curriculum --stage-timesteps 100000

# Full 4-stage curriculum, auto-promoting when success >= 50%
python3 scripts/train_pick_place.py \
  --curriculum --stage-timesteps 100000 --promote-success 0.5

# Other useful flags
--seed 0 --device auto --eval-episodes 10
--learning-rate 3e-4 --batch-size 256 --gamma 0.999
--buffer-size 300000 --learning-starts 5000
--checkpoint-freq 50000 --eval-freq 10000
--time-penalty 0.001 --lift-scale 20 --transport-scale 10
--out outputs/sac_pick_place
```

Curriculum stages (trainer's `STAGE_CONFIGS`):

| Stage | Box pose | Cup pose | Physics |
|---|---|---|---|
| 1 | fixed | fixed | fixed |
| 2 | random yaw + xy | fixed | fixed |
| 3 | random yaw + xy | random xy | fixed |
| 4 | random yaw + xy | random xy | random mass + friction |

Warm start only applies to stage 1; stages 2–4 disable it on purpose
(per-reset IK is slow with a randomized box).

Saves under `outputs/sac_pick_place/` (or `--out`):

```
outputs/sac_pick_place/
├── stage1/            # or stage2/, stage3/, stage4/
│   ├── final_model.zip
│   ├── norm_stats.npz
│   ├── vecnormalize.pkl
│   ├── summary.json
│   └── *.zip          # periodic checkpoints
├── stage1_h0.00/      # --stage1-curriculum sub-runs
├── stage1_h0.03/
├── stage1_h0.09/      # final chained stage-1 output
└── tb/                # TensorBoard logs
```

TensorBoard:

```bash
tensorboard --logdir outputs/sac_pick_place/tb
```

### 3c. Watch the trained model

```bash
python3 scripts/watch_pick_place.py
# default: outputs/sac_pick_place/stage1/final_model.zip, lift_only, 5 episodes

python3 scripts/watch_pick_place.py --task-mode full
python3 scripts/watch_pick_place.py --episodes 10
python3 scripts/watch_pick_place.py --episodes 0        # until window closed
python3 scripts/watch_pick_place.py --stochastic
python3 scripts/watch_pick_place.py --no-viewer         # headless metrics only
python3 scripts/watch_pick_place.py \
  --model outputs/sac_pick_place/stage1/final_model.zip \
  --norm-stats outputs/sac_pick_place/stage1/norm_stats.npz
python3 scripts/watch_pick_place.py --no-warm-start --pregrasp-height 0.09
```

`--model` may be a relative path (resolved from the repo root). The matching
`norm_stats.npz` is auto-loaded from the model's sibling directory.

---

## 4. Docker (headless / GPU hosts)

```bash
# CPU container (drops you into bash)
docker compose up -d
docker compose exec mira-arm bash

# NVIDIA GPU override — must be combined with the base file
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d

# Run a viewer inside the container (host must allow X11)
xhost +local:root
docker compose exec mira-arm python3 scripts/view.py

# Validate config
docker compose config
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml config
```

Headless display via Xvfb:

```bash
XVFB_RUN=1 docker compose up -d
docker compose exec mira-arm python3 scripts/view.py
```

> The container deliberately does **not** export `MUJOCO_GL`; `sim.py`
> auto-selects EGL (NVIDIA) or osmesa, and `view.py` opens a GLFW window.

---

## 5. Rendering backend (host, not Docker)

`MUJOCO_GL` must be set **before** importing MuJoCo (the scripts handle this).

| Situation | Value |
|---|---|
| Windowed viewer | `glfw` (view.py default) |
| Headless CPU | `osmesa` |
| Headless NVIDIA GPU | `egl` |
| macOS | `glfw` (or `cgl`) |

Override explicitly:

```bash
MUJOCO_GL=egl python3 scripts/sim.py
MUJOCO_GL=osmesa python3 scripts/sim.py
```

---

## 6. One-liner: bootstrap → train → watch

```bash
# 1) prove the grasp works
python3 scripts/test_grasp.py

# 2) learn to grasp and lift
python3 scripts/train_pick_place.py \
  --task-mode lift_only --stage 1 --total-timesteps 10000 \
  --warm-start --pregrasp-height 0.0

# 3) continue into the full task
python3 scripts/train_pick_place.py \
  --task-mode full --stage 1 --total-timesteps 10000 \
  --warm-start --pregrasp-height 0.0 \
  --init-model outputs/sac_pick_place/stage1/final_model.zip

# 4) watch it
python3 scripts/watch_pick_place.py --task-mode full --episodes 5
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `PermissionError` writing `outputs/` | Directory may be root-owned (container). `sudo chown -R $USER outputs/` or pass `--out /tmp/mira`. |
| Viewer fails: no display | Use `--no-viewer`, `scripts/sim.py` (headless), or `XVFB_RUN=1` in Docker. |
| `MUJOCO_GL` / EGL errors | Force `MUJOCO_GL=osmesa` (CPU) or `egl` (GPU). Must be set before `import mujoco`. |
| SAC learns nothing / entropy collapse | Ensure `--closure-gate-radius`, `--ent-coef-floor` defaults are intact; start from `lift_only`. |
| Marshmallow slips out | Use `test_grasp.py` (flat pads + 0.5 N·m cap); stock fingers cannot grasp it. |
| Wrong scene | `--scene <file>` for `sim.py` / `view.py`; `MIRA_SCENE=<file>` for `watch_gym.py` / `test/test_gym.py`. |

---

## 8. Environment IDs

| Gymnasium ID | Env class |
|---|---|
| `MIRA/SO101Reach-v0` | `mira/envs/so101_reach_env.py` |
| `MIRA/SO101PointToPoint-v0` | `mira/envs/so101_point_to_point_env.py` |
| `MIRA/SO101PickPlace-v0` | `mira/envs/so101_pick_place_env.py` |

```python
import gymnasium as gym
import mira  # registers the IDs

env = gym.make("MIRA/SO101PickPlace-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```
