# Pick-and-Place Debug Log

Running log of the Stage 1 SAC training failure (0% success) and the fixes.
Started: 2026-11-03.

## 1. Symptom

Command:

```bash
python3 scripts/train_pick_place.py --stage 1 --total-timesteps 500000 \
  --eval-episodes 50 --out outputs/sac_pick_place_long
```

At ~300k steps: `rollout/success_rate = 0` for the entire run, `rollout/ep_rew_mean`
flat at ~1.2–1.8, `rollout/ep_len_mean` pinned at 400.

## 2. Evidence

From `outputs/sac_pick_place_long/tb/stage1/SAC_1`:

| metric | 4k | 84k | 168k | 247k | 326k |
|---|---|---|---|---|---|
| `ep_rew_mean` | 0.78 | 1.82 | 1.68 | 1.27 | 1.24 |
| `success_rate` | 0 | 0 | 0 | 0 | 0 |
| `ep_len_mean` | 400 | 395.6 | 400 | 400 | 391 |
| `critic_loss` | 0.09 | 1.15 | 52.3 | 211.3 | 98.0 |
| `actor_loss` | -35.5 | -77.0 | -360.9 | -470.7 | -233.7 |

Deterministic eval of `outputs/sac_pick_place_long/stage1/sac_stage1_300000_steps.zip`
(8 episodes, saved normalizer):

- success **0/8**
- mean return **2.73**
- EE→box: 0.334 → **0.061 m**
- box→cup xy: 0.233 → **0.233 m** (unchanged)
- box z: **1.24 cm** (unchanged), `lifted=False`
- gripper action oscillates; never held closed.

**Smoking gun:** `2.73 == 10 × (0.334 − 0.061)`. 100% of the return is the
approach-shaping term; the transport term contributes exactly 0. The policy
learned only "hover ~5 cm from the box", then wandered for the remaining 350 steps.

The 100k run (`outputs/sac_pick_place/stage1/summary.json`) already reported
`success_rate: 0.0`, `mean_return: 1.538`, matching a replay of its
`final_model.zip` (EE→box 0.334 → 0.179).

`rollout/success_rate` is not a logging artifact: SB3 2.9 reads
`info["is_success"]` into `ep_success_buffer` at episode end
(`stable_baselines3/common/off_policy_algorithm.py`).

## 3. Root causes

1. **Reward local optimum** (`mira/envs/so101_pick_place_env.py`).
   `10·Δ‖EE−box‖ + 5·Δ‖box−cup‖xy`: the approach term pays in full for reaching
   the box; the transport term only activates after a successful grasp (a
   discontinuous discovery), and there is no reward for closing the gripper.
   **Correction (see §5.6):** the undiscounted `ΔΦ` form was *not* the main
   defect. Switching to `γΦ'−Φ` over a 700-step horizon is actively worse —
   `γ^700≈0.0009` erases the terminal term, while `(γ−1)Φ` becomes a per-step
   living reward that is *larger when farther from the goal*. Keep `ΔΦ`
   (`shaping_gamma=1.0`) and fix the scale/incentives instead.
2. **Discount/horizon mismatch** (`scripts/train_pick_place.py:199`, env
   `max_episode_steps=400`). `γ=0.99` over 400 steps → effective horizon ~100.
   `+10` at step ~300 is discounted to `0.99^300 ≈ 0.049` → ~0.5, vs 2.7 for
   approaching. "Reach the box and stop" is near-optimal under this MDP.
3. **Episode shorter than the demonstrated trajectory.** `scripts/test_grasp.py`
   defaults (600 descend + 500 close + 600 lift single steps = 340 env-steps at
   `frame_skip=5`) just for grasp+lift; a full pick-and-place is ~550–960
   env-steps. `max_episode_steps=400` truncates before the task can finish.
4. **Failure branch unreachable in Stage 1.** `fix_cup=True` pins the cup upright
   and the box stays in the workspace, so `failure` never fires → every episode
   runs to truncation, `ep_len_mean=400`.
5. **Critic divergence** (`critic_loss` 0.09 → 300+). Non-stationary
   `NormalizeObservation` (running stats updated during training), no reward
   normalization, ±3.35 rad action range.

## 4. Fix plan

| # | Change | File |
|---|---|---|
| 1 | Extract position+world-down IK into `mira/kinematics.py` (from `test_grasp.py`) | new |
| 2 | Reward: undiscounted potential shaping (`shaping_gamma=1.0`), `approach_scale=2`, `transport_scale=5`, grasp/lift bonuses, no per-step height bonus | env |
| 3 | Grasp detection (both pads contact marshmallow) + `grasped`/`ever_grasped` info | env |
| 4 | Optional warm start: reset arm above the box | env |
| 5 | Horizon 400 → 700 | env |
| 6 | `gamma` 0.99 → 0.995; reward normalization (`VecNormalize(norm_reward=True)`) | trainer |
| 7 | Periodic eval callback so success is visible mid-run | trainer |
| 8 | Scripted oracle that completes the full pick-and-place and asserts `is_success` | new |

## 5. Implementation log

### 5.1 `mira/kinematics.py` (new, 96 lines)

Extracted the position + world-down IK from `scripts/test_grasp.py` verbatim into
a shared module: `Kinematics` (finite-difference damped-least-squares over the 5
arm joints, `iters=1200`, `wdir=0.05`, `damping=3e-3`), plus `DOWN` and
`PAD_MIDPOINT_Z=0.0015`. `scripts/test_grasp.py` now imports it (−89 lines, net
−54). This is what lets the env and the scripts solve the same pose problem.

### 5.2 `mira/envs/so101_pick_place_env.py` (rewritten, 645 lines)

| Change | Detail |
|---|---|
| New params | `warm_start=False`, `shaping_gamma=0.99`, `approach_scale=2.0` (replaces `progress_scale`), `transport_scale=5.0`, `grasp_bonus=2.0`, `release_required=False`, `time_penalty=0.0` |
| Horizon | `max_episode_steps` 400 → **700** |
| Shaping | `φ = -(approach_scale·‖EE−box‖ + transport_scale·‖box−cup‖xy)`; `reward = shaping_gamma·φ' − φ` (discounted, potential-based) |
| Grasp detection | `_pads_contact_box()`: true only when **both** `fixture_pad` and `moving_pad` contact `marshmallow_geom`; one-shot `+grasp_bonus`, latched in `_ever_grasped` |
| Lift | `+lift_bonus` once at `box_z > 0.03`, then `+0.1·(box_z−0.03)` |
| Warm start | `reset()` solves a pre-grasp pose 9 cm above the box (`_pregrasp_q`) and starts the arm there; multi-start guesses = previous solution, joint mid-range, `wrist_roll=±1.0`; 10 iterations of pad-offset correction |
| Info keys | `grasped`, `ever_grasped`, `phi` added (plus existing `box_pos`, `cup_pos`, `box_cup_dist_xy`, `is_success`, `lifted`) |
| Self-check | `__main__` runs `env_checker` on default/warm-start/stage1–4, then a scripted grasp+lift |

**Deviation:** `_pregrasp_q` caches the solved pose keyed on the exact box
position. Without the cache, Gymnasium's `check_step_determinism` failed because
the cached "previous solution" could select a different near-tied IK local
minimum on two identical seeded resets. The cache still reuses the previous
solution as a multi-start seed for any new box position.

### 5.3 Verified results (independent re-run, not just the agent's report)

```text
$ python3 scripts/test_grasp.py                     # exit 0
pad midpoint error:   0.003 cm
natural box yaw   :   40.87 deg
box z             :    7.67 cm (rise +6.42 cm)
EE-box distance   :    1.243 cm
PASS

$ python3 -m mira.envs.so101_pick_place_env         # exit 0
check_env OK (default / warm_start=True / stage1 / stage2 / stage3 / stage4)
warm-start EE-box distance at reset: 0.0864 m
ever_grasped=True
final box z=0.0838 m (rise +7.13 cm)
total accumulated reward=10.135
reward breakdown: shaping=5.894, grasp=2.000, lift/success/failure/time=2.240
PASS
```

### 5.4 Measured constraints that shaped the next fixes

- **Warm-start reset cost:** fixed box = **0.000 s/reset** (cache hit), randomized
  box position = **3.1 s/reset** (full multi-start IK). → warm start must be
  Stage-1-only; enabling it for stages 2–4 would make resets 4× the cost of an
  episode.
- **Cup geometry:** cavity floor `z≈0.01`, rim top `z≈0.078`, cube half-size
  `0.0125`. The gripper fingers are wider than the cup opening, so the cube
  cannot be lowered into the cup while grasped. The scripted oracle must
  **release with the cube center at ~0.095 m** (fingers clear the rim) and let
  the cube drop; success then requires `xy < 2 cm` and `0.01 < z < 0.06`.

### 5.5 Reward correction: `shaping_gamma` must stay 1.0

The first implementation used `shaping_gamma=0.99`. Measured effect on the
scripted grasp+lift self-check:

| `shaping_gamma` | reported shaping | true `Φ_end − Φ_start` | interpretation |
|---|---|---|---|
| 0.99 | **5.894** | ~0.66 | 5.2 of the "shaping" was a living reward |
| 1.0 | **0.164** | 0.164 | pure potential difference |

Why: with `Φ = -(2·‖EE−box‖ + 5·‖box−cup‖xy) < 0`, the discounted form
`γΦ'−Φ` pays `(γ−1)Φ > 0` per step while standing still, and that payout is
*larger* the farther the arm is from the goal. Over 700 steps `γ^700 ≈ 0.0009`
kills the only behavior-dependent term (`γ^T Φ_T`). The agent is therefore
paid to loiter far away. Reverted the default to `shaping_gamma=1.0`
(undiscounted `Φ'−Φ`): standing still pays 0, progress pays in full.

Also removed the agent-added per-step `+0.1·(box_z−0.03)` height bonus: it paid
more at higher `z`, which directly fights the success condition `z < 0.06`.

Re-verified after the change: `check_env` 6/6 OK, scripted grasp `PASS`
(`shaping=0.164, grasp=2.000, lift=2.000`, total 4.164), oracle `PASS`.

### 5.6 `scripts/train_pick_place.py`

| Change | Detail |
|---|---|
| Discount | `--gamma` default 0.99 → **0.995** (`0.995^300 ≈ 0.22` vs `0.99^300 ≈ 0.049`) |
| Reward norm | `VecNormalize(DummyVecEnv, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=args.gamma)`; obs norm stays in `NormalizeObservation` (no double norm) |
| Periodic eval | `EvalSuccessCallback` every `--eval-freq` (default 10 000): syncs eval obs stats from the training normalizer, calls `evaluate()`, logs `eval/success_rate` + `eval/mean_return`, prints `[eval @ N] ...` |
| Warm start | `--warm-start` (default off), **stage 1 only**; stages ≥ 2 print `warm start disabled ...` because per-reset IK is ~3.1 s with a randomized box |
| Artifacts | `vecnormalize.pkl` saved next to `final_model.zip` / `norm_stats.npz`; `summary.json` gains `gamma`, `warm_start`, `norm_reward` |
| Compat | `NormalizeObservation`, `norm_stats.npz`, checkpoint naming, and `evaluate()` (deterministic, final-step `is_success`, raw return) unchanged |

Verified smoke (independent re-run, 2048 steps):

```text
gamma=0.995 warm_start=True reward_norm=VecNormalize(norm_reward=True, clip_reward=10.0)
[eval @ 1024] success=0.0% mean_return=11.80
[eval @ 2048] success=0.0% mean_return=11.57
Stage 1: success=0.0% mean_return=11.62          # expected at 2k steps
EXIT=0
```

Artifacts: `final_model.zip`, `norm_stats.npz`, `vecnormalize.pkl`,
`sac_stage1_1024_steps.zip`, `sac_stage1_2048_steps.zip`, `summary.json`
(`"gamma": 0.995`, `"warm_start": true`, `"norm_reward": true`).

### 5.7 `scripts/oracle_pick_place.py` (new, 536 lines)

Scripted full pick-and-place: multi-start IK (mid, wrist_roll ±1.0, previous
best, plus extra starts if error > 1 cm), pad-offset iteration, eight joint-space
ramps, success taken only from `env.step()`.

Retuning required by the 700-step budget and reachability:

| Constant | Spec | Used | Reason |
|---|---|---|---|
| phase steps | 250/150/100/150/250/150/100/150 = 1300 | 120/70/60/70/140/80/70/60 = 670 | 1300 > `max_episode_steps=700` |
| `CUP_HIGH_Z` | 0.13 m | **0.11 m** | 0.13 is unreachable with world-down IK (best 3.1 cm error); 0.11 gives 0.428 cm |
| `CUP_LOW_Z` | 0.095 m | 0.095 m | fingers clear the rim (z≈0.078); release and drop |

Verified (independent re-run, 13 s):

```text
IK grasp      : pos=0.000 cm dir=0.0001 pad=0.000 cm DOWN.dot=1.0000
IK pre-grasp  : pos=0.001 cm dir=0.0000 pad=0.001 cm
IK above-cup  : pos=0.428 cm dir=0.0270 pad=0.428 cm
IK cup-release: pos=0.000 cm dir=0.0001 pad=0.000 cm
phase=approach  steps=120 box_z=0.0124 xy=0.2335 grasped=False
phase=descend   steps= 70 box_z=0.0124 xy=0.2340 grasped=False
phase=close     steps= 60 box_z=0.0124 xy=0.2308 grasped=True
phase=lift      steps= 70 box_z=0.0854 xy=0.2307 grasped=True
phase=transport steps=140 box_z=0.1020 xy=0.0008 grasped=True
phase=lower     steps= 80 box_z=0.0843 xy=0.0053 grasped=True
phase=release   steps= 41 box_z=0.0537 xy=0.0015 grasped=True
phase=settle    steps=  0 box_z=0.0537 xy=0.0015 grasped=True
final box = [0.1992 -0.1187 0.0537]  final box-cup xy = 0.0015 m
cup upright = True   grasped_now = False
conditions: xy<0.02=True 0.01<z<0.06=True upright=True
success_seen = True   PASS   EXIT=0
```

`settle` is 0 because the env terminates the moment `is_success` becomes true.

### 5.8 Pending

- 100k-step Stage 1 retrain (`--warm-start`, eval every 10k) to confirm the
  eval success rate actually rises above 0.
- Neutral-reset evaluation of the trained policy (warm-start success is a
  curriculum metric, not the final deployment metric).

## 6. How to validate

```bash
python3 scripts/test_grasp.py                 # Step 0 still PASS
python3 -m mira.envs.so101_pick_place_env     # env_checker on all configs + grasp self-check
python3 scripts/oracle_pick_place.py          # scripted full task, exit 0 = success
python3 scripts/train_pick_place.py --stage 1 --total-timesteps 2048 \
  --eval-episodes 2 --out /tmp/mira_smoke     # short smoke run
```

## 7. Stage 1 retrain #1 (100k) — result and diagnosis

Command (detached so a shell timeout cannot kill it):

```bash
setsid nohup python3 scripts/train_pick_place.py --stage 1 --total-timesteps 100000 \
  --eval-episodes 5 --eval-freq 10000 --warm-start \
  --out outputs/sac_pick_place_fixed < /dev/null > /tmp/mira_train_100k.log 2>&1 &
```

### 7.1 Result: still 0% success

| steps | eval success | eval mean_return |
|---:|---:|---:|
| 10 000 | 0.0% | −0.61 |
| 20 000 | 0.0% | −0.72 |
| 30 000 | 0.0% | −0.13 |
| 40 000 | 0.0% | −0.23 |
| 50 000 | 0.0% | −0.04 |
| 60 000 | 0.0% | −0.23 |
| 70 000 | 0.0% | +0.04 |
| 80 000 | 0.0% | −0.16 |
| 90 000 | 0.0% | −0.10 |
| 100 000 | **0.0%** | **−0.02** |

`time_penalty=0`, so return ≈ 0 means "no net potential progress", not "loitering
penalty". The 2.73 plateau of the original run is gone (the potential is no
longer farmable), but the task is still not learned. Run time 31 min at ~65 fps
(eval included).

### 7.2 Deterministic policy diagnostic (`/tmp/diag_policy.py`)

`final_model.zip` + `norm_stats.npz`, 2 episodes each, print every 100 steps.
`grip_q` is the gripper **joint** position (range `[-0.1745, 1.7453]`;
closed ≈ 0.10, pre-grasp open = 0.25).

**Warm start** (`warm_start=True`):

| t | EE–box | box_z | grip_q | contact |
|---:|---:|---:|---:|---|
| 1 | 0.0857 | 0.0124 | 0.251 | False |
| 100 | 0.1618 | 0.0124 | 0.974 | False |
| 200 | 0.0966 | 0.0124 | 1.693 | False |
| 300 | 0.0946 | 0.0124 | 1.745 | False |
| 700 | 0.0947 | 0.0124 | 1.745 | False |

min EE–box 0.0367 · gripper never < 0.15 · contact never · box never moves.

**Neutral reset** (`warm_start=False`):

| t | EE–box | box_z | grip_q | contact |
|---:|---:|---:|---:|---|
| 1 | 0.3330 | 0.0124 | 0.251 | False |
| 100 | 0.2784 | 0.0124 | 0.914 | False |
| 200 | 0.1695 | 0.0124 | 1.409 | False |
| 300 | 0.1152 | 0.0124 | 1.512 | False |
| 700 | 0.1003 | 0.0124 | 1.720 | False |

### 7.3 Interpretation

1. **The gripper is frozen fully open.** It goes 0.251 → 1.745 (the joint's
   upper limit) and never drops below 0.15. The policy never even attempts a
   grasp, so the +2 contact bonus and the +2 lift bonus are unreachable.
2. **The EE stalls outside the grasp manifold.** 3.7 cm from the box under
   warm start, 10 cm under neutral reset. Both are on the approach-shaping
   plateau. Measured (`/tmp/measure_grasp.py`): at the IK-solved grasp pose
   (box + `PAD_MIDPOINT_Z`) the EE–box distance settles at **1.37 cm**, and a
   *static* close from that pose pushes the cube ~1.3 cm sideways
   (`[0.160,0.110] → [0.170,0.119]`) with no contact — so pad contact needs
   ≈1.4 cm *and* the gradual close-and-follow the oracle uses, not a teleport.
   The policy's 3.7 cm stall is still ~2.3 cm too far.
3. **Exploration collapsed before the grasp was found.** SAC's learned
   `ent_coef` fell 0.55 → 0.0695 → 0.0135 by 21k steps. With a 6-dim
   continuous action and a grasp that pays nothing until *both* pads touch a
   2.5 cm cube, the policy settles into the deterministic "approach and hold
   the gripper open" optimum.
4. Episodes are bit-identical (fixed stage-1 box, deterministic policy), so
   this is a stable local optimum, not seed noise.

The reward is now *correct* (potential difference, no farming — confirmed by
the 5.894 → 0.164 shaping drop) but not *sufficient*: approach shaping cannot
create a closing behavior whose effect is zero until contact.

### 7.4 Next

Awaiting Oracle consultation on the minimal change set (grasp-readiness
potential gated on proximity vs. entropy floor vs. curriculum). Do not launch
another long run until that decision is made.
