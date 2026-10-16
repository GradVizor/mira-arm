# models/so101 — MJCF KNOWLEDGE BASE

**Generated:** 2026-10-14
**Commit:** d33014e
**Branch:** main

MJCF model of the SO-101 arm: 4 XMLs + `assets/` (14 STL). `assets/` MUST stay beside the XMLs (`meshdir` / `<include>` are directory-relative). Cross-cutting rules live in `../AGENTS.md` — this file owns model internals only.

## FILE ROLES

| File | Included? | Role |
|------|-----------|------|
| `scene.xml` | entry point | `<include>`s the active calib; adds `visual`/`asset`/floor/props; has **no `<compiler>`** of its own |
| `so101_new_calib.xml` | via `scene.xml` (ACTIVE) | virtual zero = **mid-range** of each joint |
| `so101_old_calib.xml` | not included (inactive) | virtual zero = **fully-extended horizontal** |
| `joints_properties.xml` | **nowhere (ORPHANED + stale)** | edits have NO effect — never rely on it |

## MJCF CHAIN

`scene.xml:2` → `<include file="so101_new_calib.xml"/>`. The included calib carries `<compiler angle="radian" meshdir="assets" autolimits="true"/>`; `scene.xml` inherits it — that is why every `mesh`/`include` path here is relative to this directory.

## DEFAULT CLASSES

- Per-calib root class (`so101_new_calib` / `so101_old_calib`): base `joint damping/frictionloss/armature` + `position kp="50"`, with sub-classes `visual` (mesh geom, `contype=0 conaffinity=0 group=2`) and `collision` (`group=3`). Bodies set `childclass`.
- `sts3215`: used by **all 6 joints + all 6 actuators**. Overrides joint dynamics and position gains (the base `kp="50"` is effectively never used).
- `backlash` (±0.5°): defined in both calibs **and** the orphan but **referenced by no element** — inert.
- Each `<actuator>` overrides `forcerange` to `±3.35` and sets `ctrlrange` per joint — this override, not the class, is the real calibration.

## GAINS

| Source | kp | kv | forcerange |
|--------|----|----|------------|
| new calib `sts3215` class | 998.22 | 2.731 | ±2.94 |
| new calib actuators (override) | — | — | ±3.35 |
| old calib `sts3215` class | 17.8 | — | — |
| `joints_properties.xml` (orphan) | 17.8 | 0.0 | ±3.35 |

`kp=998.22` assumes servo proportional gain 16 (per RBE501-RL-arm-project). New-calib joint dynamics: `damping=0.60 frictionloss=0.052 armature=0.028`.

## CTRLRANGE — the actual calibration diff

| Joint | new calib | old calib |
|-------|-----------|-----------|
| shoulder_pan | ±1.91986 | ±1.91986 |
| shoulder_lift | ±1.74533 | −3.31612 … 0.17453 |
| elbow_flex | −1.69 … 1.69 | −0.17453 … 3.14159 |
| wrist_flex | ±1.65806 | ±1.65806 |
| wrist_roll | −2.74385 … 2.84121 | ±2.79252 |
| gripper | −0.17453 … 1.74533 | −0.17453 … 1.74533 |

`shoulder_lift`/`elbow_flex` shift most (old fully-extended zero is asymmetric). New-calib `elbow_flex` joint carries the comment "5-degree calibration offset applied to joint range" (`so101_new_calib.xml:76`).

Provenance: same Onshape doc/workspace (`7715cc28…/4fd0791b…`), **different elements** — new `e/826c553ede3b…`, old `e/bbd5da878b98…`.

## PROPS

- `coffee_cup` (freejoint, pos `0.20 -0.12`): visual = `coffee_mug.stl` (contype/conaffinity 0, no collision); contact = bottom cylinder + 4 wall boxes tucked ~10 mm inside the mesh silhouette so no contact geom pokes through.
- `marshmallow` (freejoint, pos `0.16 0.11`): 2.5 cm box, graspable.
- Mug mesh + prop materials live in `scene.xml`'s `<asset>`; the mug STL is referenced only here (not by the calib XMLs).

## LOCALLY FORBIDDEN

- Never trust the base `kp="50"` — `sts3215` always wins for the 6 arm joints.
- Never assume the two calibs differ in gains; only `ctrlrange`/joint `range` change.
