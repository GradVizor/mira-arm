# Pick-and-Place Plan — Marshmallow → Coffee Cup

Tracking doc for moving the SO-101 from the point-to-point reach task to
pick-and-place. Check items off as they land.

## Baseline (what already works)

- [x] `SO101PointToPointEnv` — 18-dim obs, 6-dim action, dense progress reward, 2 cm success.
- [x] PPO trainer (`scripts/train_point_to_point.py`) + `NormalizeObservation`.
- [x] Checkpoint `outputs/ppo_point_to_point.pt` + `_norm.npz`.
- [x] Viewer `scripts/watch_point_to_point.py`.
- [x] `scene_with_coffee_mug.xml` has both props as free-joint bodies.

Verified facts (do not re-derive):

| Fact | Value |
|---|---|
| Marshmallow | 2.5 cm box at `(0.16, 0.11, 0.0125)`, density 300, friction 0.9 |
| Cup | at `(0.20, -0.12)`, hollow, inner radius ≈ 3.1 cm, rim ≈ 7.8 cm |
| Gripper convention | joint low (−10°) = **closed**, high (100°) = **open** → action `-1` = close, `+1` = open |
| IK reachability | above marshmallow, grasp height, above cup, inside cup all solve at 0.00 cm position error (`shoulder_lift ≈ −90°`) |
| Scene sizes | `nq=20`, `nv=18`, `nu=6` — index the 6 arm joints **by name**, never `qpos[:6]` |
| `point_B` | `(0.18, -0.12, 0.15)` — 2 cm off the cup axis, 7 cm above the rim (pre-place pose only) |
| Grasp model | `mira/pick_place_model.py`: flat pads added at runtime via `MjSpec`; original finger-mesh collisions disabled; gripper force capped at 0.5 N·m |
| Natural grasp yaw | **40.87°** at site `(0.1477, 0.0993, 0.0074)`; the box yaw must match (5-DOF arm cannot do position + down + yaw) |
| Pad offset | pad midpoint − site = `(0.01673, 0.00006, 0.00529)` in the gripper frame at `open_q=0.25` |
| Step 0 result | `scripts/test_grasp.py` PASS: box rises **6.42 cm**, EE-box **1.24 cm**, 3/3 runs |

## Step 0 — Prove the grasp physics

- [x] Write `scripts/test_grasp.py`: pose above marshmallow → descend → close gripper → wait → lift 10 cm.
- [x] Run it; record whether marshmallow `z` follows the EE.
- [x] **Pass:** marshmallow rises with EE and stays within 1–2 cm of the fingers.
  - Result: **PASS, 3/3 runs**. Box rise **6.42 cm**, EE-box **1.24 cm**, pad-midpoint error **0.003 cm**.
  - Config: `--force 0.5 --open-q 0.25 --close-q 0.10 --pre-height 0.09 --lift-height 0.075`.
  - Stock finger meshes fail: tapered tips contact the box’s top corners, press it into the floor, and the box slips out. The fix is flat pads + disabling the original finger-mesh collisions + capping gripper force.
- [x] If fail, tune in order: finger friction/force → compliant pads (`solref`/`solimp`) → marshmallow density/size → orientation-aware IK.
  - Resolved at the first two levels: flat pads, finger-mesh collision off, force `0.5 N·m`.

> Step 0 passed on 2026-11-02. Proceed to Step 1. The generated calibration XML
> is untouched; the pad patch lives in `mira/pick_place_model.py`.

## Step 1 — New environment `SO101PickPlaceEnv`

- [x] Create `mira/envs/so101_pick_place_env.py` (copy point-to-point, don't mutate it).
- [x] Load the scene through `mira.pick_place_model.build_padded_model()`; do **not** hand-edit the generated calibration XML.
- [x] Register `MIRA/SO101PickPlace-v0` in `mira/__init__.py`.
- [x] Set the box yaw to the arm’s natural grasp yaw (~40.9°) or randomize ±6° around it; a top-down grasp of an axis-aligned box is not reachable with this 5-DOF arm.
- [x] Extend obs: arm `qpos(6)`, `qvel(6)`, `ee(3)` + marshmallow `xyz(3)` + cup `xyz(3)` = **21 dims**.
- [x] Reward staged: approach (`-‖ee − marshmallow‖`) → lift bonus → transport (`-‖marshmallow − cup‖xy`) → success bonus. Potential-based shaping is in place.
- [x] Success on the **object**: marshmallow `xy` within ~2 cm of cup axis, `z` inside cavity, cup upright.
  - Implemented: `xy < 2 cm`, `0.01 < z < 0.06`, and cup local +z dotted with world +z `> 0.9`.
- [x] Termination: success; fail if marshmallow leaves workspace or cup tips; truncate at 300–500 steps.
  - Implemented: workspace bounds `x∈[-0.05,0.45]`, `y∈[-0.35,0.35]`, `z∈[-0.02,0.40]`; cup tip and out-of-workspace terminate with `failure_penalty`; truncate at 400.
- [x] Reset: explicitly set free-joint `qpos` for `marshmallow_free` / `coffee_cup_free` (randomization); `mj_resetData` restores XML defaults.
- [ ] Exclude the gripper DOF from IK; add orientation (approach-direction) constraints.
  - `scripts/test_grasp.py` already uses only the 5 arm joints + world-down; the RL env has no IK.

## Step 2 — Curriculum + algorithm

- [x] Stage 1: fixed pick/place, cup pinned/fixed → learn close → lift → move → drop.
  - `STAGE_CONFIGS[1]`: `fix_cup=True`; marshmallow fixed at the default pose. The optional under-EE warm start is deferred.
- [x] Stage 2: randomize marshmallow yaw + ±2–3 cm xy, cup fixed.
  - `STAGE_CONFIGS[2]`: `randomize_box=True`, `randomize_box_pos=True`, `fix_cup=True`.
- [x] Stage 3: randomize marshmallow and cup positions, cup free.
  - `STAGE_CONFIGS[3]`: `randomize_cup_pos=True`, `fix_cup=False`.
- [x] Stage 4: randomize masses/friction.
  - `STAGE_CONFIGS[4]`: `randomize_physics=True` (box mass ×0.7–1.3, sliding friction ×0.8–1.2).
- [x] Train with **SAC** (SB3 2.9.0 installed) + `NormalizeObservation`; re-save `*_norm.npz` for the new env.
  - `scripts/train_pick_place.py` saves `final_model.zip`, `norm_stats.npz`, checkpoints, and `summary.json` per stage.
- [ ] Optional: add `HerReplayBuffer` (goal-conditioned obs dict) once shaped reward converges.
- [ ] Fallback: scripted IK pick-and-place → validates physics and yields demos for behavior cloning (roadmap Phase 8).

> Step 2 status (2026-11-02): env flags, cup pinning, cup-upright/workspace failure, and the SAC
> curriculum trainer are implemented. `env_checker` passes on all four stage configs; a 256-step
> CPU smoke run completed and saved artifacts under the configured `--out` directory
> (`/tmp/mira_sac_smoke/stage1` for the smoke test). Full >80% training has **not** been run yet —
> that is the next long-running step.

## Step 3 — Evaluate / visualize

- [ ] Create `scripts/watch_pick_place.py` (clone `watch_point_to_point.py`).
- [ ] Draw marshmallow/cup goals; log `grasped`, `marshmallow_z`, distance-to-cup, success rate.
- [ ] Keep `--no-viewer` headless mode for training-loop eval.

## Gotchas

- `scene.xml` is clean — use `scene_with_coffee_mug.xml` for props. (AGENTS.md note about `scene.xml` carrying props is stale.)
- Cup is 16 thin wall boxes: fix/weld it for Stage 1, release later; it can tip on contact.
- **Stock finger meshes cannot grasp the marshmallow.** Their tapered tips contact the box’s top corners and press it into the floor. Use the flat pads in `mira/pick_place_model.py` and keep the original finger-mesh collisions disabled. Do **not** hand-edit `so101_calib.xml`.
- **5-DOF yaw constraint:** position + world-down + yaw is overconstrained. The natural grasp yaw is ~40.87°, so a top-down grasp of an axis-aligned box is not reachable. Align the box yaw to ~40.9° or randomize around it.
- **Gripper force cap:** the model’s 3.35 N·m max ejects the 4.7 g box. The pad model caps it at 0.5 N·m; `test_grasp.py` uses `--force 0.5`.
- `scripts/test_grasp.py` exits 0 on PASS and 1 on FAIL; it does not involve the cup.
- `_solve_ik` in the point-to-point env still includes the gripper DOF (zero Jacobian column) — clean it up if reused.
- State-based obs is sim-only. Real hardware needs vision/markers and a fix for the MJCF-hinge ↔ LeRobot-linear gripper mapping.

## Definition of done

- [x] Grasp test passes reproducibly.
- [x] `MIRA/SO101PickPlace-v0` passes `gymnasium.utils.env_checker`.
- [ ] SAC reaches > 80% success on randomized positions (report seed + step count).
- [ ] `watch_pick_place.py` shows the marshmallow ending inside the cup.
