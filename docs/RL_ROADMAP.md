# MIRA RL Roadmap — Zero to VLA

**Purpose:** turn MIRA into an RL research platform while learning reinforcement learning deeply, end to end.
**Audience:** strong software engineer, new to RL.
**North star (from README):** teleop data → RL training → VLA policies → real hardware.

> Convention reminders: deps stay unpinned (`>=`); do **not** add torch to the base Docker image — use a separate `requirements-rl.txt` so the lean sim image stays lean (`Dockerfile:4`).

---

## 0. Where the repo stands today

| Have | Missing (you build these) |
|---|---|
| MuJoCo MJCF arm, 6 `position` actuators (`ctrl` = target joint angles) | Gymnasium env wrapper |
| `scene.xml` with `coffee_cup` + `marshmallow` free-joint props | Task definitions + rewards |
| Headless stepping (`sim.py`) + interactive viewer (`view.py`) | Training / eval / teleop scripts |
| `kp=998.22, kv=2.731` position servos (already PD-like) | Obs/action normalization, frame-skip wrapper |
| Calibration XMLs; unused `backlash` class (sim-to-real hook) | RL deps (none yet) |

Key model facts you will rely on:
- Action space = 6-dim continuous target joint angles (radians), clipped to each actuator's `ctrlrange`.
- Joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` (index 5).
- Gripper is a hinge in MJCF but **linear in LeRobot** (`0`=closed, `100`=open) — fix this mapping before training.
- `scene.xml` inherits `angle="radian" meshdir="assets"` from the included calibration XML.

---

## Phase 0 — Foundations (2–4 weeks)

**Learn**
- MDP: state, action, reward, transition, policy π, return, discount γ, value/action-value functions, Bellman equations.
- Probability (expectation, variance), calculus (gradients), linear algebra (enough to follow derivations).
- PyTorch: tensors, autograd, `nn.Module`, optimizers, `.backward()`.

**Do in MIRA**
- Read `scripts/sim.py` line by line. Know `mj_step`, `data.ctrl`, `qpos`, `qvel`, `model.opt.timestep`. This *is* your environment.
- Run `python scripts/view.py --manual` and move each joint — build physical intuition for the 6 DOF.

**Checkpoint** — write the Bellman equation from memory; explain why RL ≠ supervised learning.

**Resources** — Sutton & Barto ch. 1–3 (free PDF); HF Deep RL Course Unit 1; Spinning Up "Key Concepts".

---

## Phase 1 — Turn MIRA into a Gymnasium env (2–3 weeks) ← keystone

**Learn**
- Gymnasium API: `reset()` / `step()`, `observation_space`, `action_space`, wrappers, seeding, `check_env`.
- Control frequency: `env_dt = model.opt.timestep × frame_skip` — get this wrong and the task is unsolvable.
- Reward design; episode termination vs truncation.

**Do in MIRA**
```
mira/__init__.py
mira/envs/so101_reach_env.py   # loads models/so101/scene.xml
mira/wrappers.py               # action scaling [-1,1] → ctrlrange, obs normalization
```
- **Task 1: Reach** — move the gripper site to a random target; `reward = -‖site_pos − target‖`; success inside threshold.
- Normalize actions to [-1, 1]; map gripper to the LeRobot convention now.
- Validate with `gymnasium.utils.env_checker.check_env`; plot random-policy returns.

**Checkpoint** — env-checker passes; random policy runs 10k steps; you can explain why 6-DOF continuous control rules out tabular methods.

---

## Phase 2 — Tabular RL intuition (1 week, don't skip)

**Learn** — Q-learning, SARSA, TD error, ε-greedy, on-policy vs off-policy.

**Do** — implement tabular Q-learning on `FrozenLake` / `CliffWalking` from scratch (~50 lines).

**Checkpoint** — explain TD learning and articulate why it cannot scale to a 6-DOF arm.

**Resources** — Sutton & Barto ch. 6; Denny Britz RL repo.

---

## Phase 3 — Value-based deep RL: DQN (2 weeks)

**Learn** — function approximation, experience replay, target network, overestimation bias, Double/Dueling DQN, n-step returns.

**Do** — DQN on `CartPole`. Then hit the wall on purpose: discretize one MIRA joint and watch it fail to scale.

**Checkpoint** — CartPole solved; you can state why DQN is wrong for your arm.

**Resources** — CleanRL `dqn.py` (read top to bottom); HF course Units 3–4.

---

## Phase 4 — Policy gradients → PPO (3–4 weeks) ← workhorse

**Learn** — policy gradient theorem, REINFORCE, baselines/advantages, GAE, actor–critic, PPO clipped objective, entropy bonus, value loss.

**Do in MIRA**
- REINFORCE on a 1-joint reach → A2C → **PPO on full 6-DOF reach**.
- Read CleanRL's single-file PPO, then train with Stable-Baselines3 PPO on your env.
- Add TensorBoard/W&B logging, eval callback, checkpointing under `scripts/train.py`.

**Checkpoint** — >90% reach success; explain every term in the PPO loss and why the clip exists.

**Resources** — Spinning Up; "The 37 Implementation Details of PPO"; CS285 Lectures 5–9.

---

## Phase 5 — Off-policy continuous control: SAC / TD3 (2–3 weeks)

**Learn** — deterministic policy gradient (DDPG), twin critics + target smoothing (TD3), maximum-entropy RL (SAC), automatic temperature tuning, replay in continuous space.

**Do** — SAC and TD3 on MIRA reach; benchmark sample efficiency vs PPO.

**Checkpoint** — SAC reaches comparable success in fewer env steps; explain the entropy term intuitively.

**Resources** — CleanRL `sac.py` / `td3.py`; CS285 Lecture 6; SB3 docs.

---

## Phase 6 — Scale & engineering hygiene (1–2 weeks)

**Learn** — vectorized envs, parallelism (subprocess vs GPU via MJX/Brax), reproducibility (seeds), proper eval (mean ± std across ≥3 seeds, confidence intervals), hyperparameter sensitivity.

**Do** — benchmark steps/sec single vs vectorized; run multi-seed sweeps; add `scripts/eval.py` and reward-curve plots into `outputs/`.

**Checkpoint** — reproducible multi-seed results + a plot you trust. RL is noisy; this phase makes results real.

---

## Phase 7 — Real tasks, rewards, curricula (2–3 weeks)

**Learn** — reward shaping (dense vs sparse, potential-based), curriculum learning, HER (Hindsight Experience Replay) for sparse goals, goal-conditioned policies, domain randomization.

**Do in MIRA**
- Reach → **Lift** (grasp `coffee_cup`) → **Place** (into target).
- Add HER for sparse pick-and-place.
- Domain randomization: masses, friction, actuator gains, action latency, initial pose.

**Checkpoint** — learned `coffee_cup` pick-and-place robust to randomized dynamics.

**Resources** — CS285 Lectures 8, 11; HER paper; SB3 "RL Tips and Tricks".

---

## Phase 8 — Imitation learning + RL fine-tune (3 weeks)

**Learn** — behavior cloning (BC), DAgger, teleop data, action chunking, diffusion policies, offline RL (CQL/IQL), residual RL.

**Do in MIRA**
- `scripts/teleop.py` records `(obs, action)` via viewer sliders/keyboard → `data/` (already gitignored).
- Train a BC policy on the demos; then fine-tune with RL.
- Align with HuggingFace **LeRobot** dataset format + gripper mapping.

**Checkpoint** — BC reproduces a demo; RL fine-tune measurably improves success rate.

---

## Phase 9 — Model-based RL / MPC (optional; MuJoCo makes it cheap)

**Learn** — Dyna, MPC, iLQR, MPPI/CEM, learned dynamics models, Dreamer-style latent models.

**Do** — MPPI/CEM planner on the true MuJoCo model for reach; compare vs model-free.

**Checkpoint** — planner solves reach; explain model-based vs model-free tradeoffs.

---

## Phase 10 — Vision + language (VLA) — the project's namesake (ongoing)

**Learn** — pixel observations, CNN encoders, vision-based RL, rendering + visual domain randomization, representation learning (VAE/SSL), transformer policies, action chunking, VLA models (RT-1/RT-2, OpenVLA, SmolVLA, π0).

**Do** — add a camera to `scene.xml`, train vision-based PPO/SAC on reach, then condition on pixels **and** a language instruction ("pick the mug").

**Checkpoint** — a policy mapping (pixels + language) → actions performs the task in sim.

---

## Phase 11 — Sim-to-real (when hardware is in reach)

**Learn** — reality gap, system identification, actuator modeling (latency, friction, **backlash**), sim-to-real RL, safety limits.

**Do** — activate the currently-unused `backlash` class, add latency/noise, calibrate against the real SO-101, transfer.

**Checkpoint** — policy works on hardware without retraining from scratch.

---

## Curated resource stack

| Purpose | Resource |
|---|---|
| Theory bible | Sutton & Barto, *Reinforcement Learning: An Introduction* (free PDF) |
| Practical deep RL intro | OpenAI Spinning Up in Deep RL |
| Hands-on, beginner-friendly | HuggingFace Deep RL Course (uses SB3) |
| Graduate depth | Berkeley CS285 (Levine, free YouTube) |
| MDP foundations | David Silver's UCL RL course |
| Reference implementations | CleanRL (single-file) |
| Libraries | Stable-Baselines3 + Gymnasium |
| Simulator | MuJoCo docs, MuJoCo Menagerie |
| Imitation / VLA | HuggingFace LeRobot |
| Task inspiration | Gymnasium-Robotics, Robosuite |

---

## Learning-first advice

- Implement REINFORCE, DQN, and PPO **from scratch once** before leaning on SB3. The bugs teach the theory.
- Keep a lab notebook: hypothesis → change → metric → conclusion. One variable at a time.
- Prefer tasks that train in minutes; scale only when the learning curve looks right.
- Always report ≥3 seeds. A single RL run means nothing.
- When stuck for >2 attempts, write the MDP down explicitly (obs, action, reward, done). Most RL bugs are a mis-specified MDP or wrong control frequency.

---

## Realistic pace

At ~8–10 h/week: **Phases 0–7 ≈ 4–5 months** to a competent "deep RL on a robot" level. Phases 8–11 are ongoing research. Depth beats speed — do not rush Phase 1 or Phase 4.
