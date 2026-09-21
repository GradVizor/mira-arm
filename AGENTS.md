# MIRA — PROJECT KNOWLEDGE BASE

**Generated:** 2026-09-21
**Commit:** d80f0f7
**Branch:** main

MIRA (Multimodal Intelligent Robotic Arm) — personal robotic-arm repo based on the SO-ARM101 (SO-101) 6-DOF arm. MuJoCo models, headless simulation, and an interactive viewer; a platform for multimodal (vision + language + action) robotics research.

## OVERVIEW

Contains a MuJoCo (MJCF) model of the SO-101 arm, two Python scripts (headless sim + interactive viewer), and minimal project scaffolding. No tests, linter, CI, or lockfile.

## STRUCTURE

```
./
├── README.md               # project overview, hardware, calibration, quick start
├── AGENTS.md               # this file
├── .gitignore              # outputs/, caches, model artifacts, agent state
├── requirements.txt        # unpinned deps: mujoco, numpy, pillow
├── Dockerfile              # lean python:3.12-slim image: sim + viewer (EGL/osmesa/GLFW libs)
├── docker-compose.yaml     # CPU compose — mira-arm service, X11 socket mount
├── docker-compose.gpu.yaml # NVIDIA override — gpus: all (needs NVIDIA Container Toolkit)
├── entrypoint.sh           # stack smoke-check, optional Xvfb, HOST_UID/GID chown
├── .dockerignore           # keeps outputs/, .git, caches out of the build context
├── docs/
│   └── RL_ROADMAP.md       # zero→VLA RL learning roadmap (see WHERE TO LOOK)
├── models/
│   └── so101/              # MJCF model — XML + assets/ MUST stay together
│       ├── scene.xml           # entry point; <include>s active calib + scene props (coffee_cup, marshmallow)
│       ├── so101_new_calib.xml # ACTIVE calib (virtual zero = mid-range); kp=998.22, kv=2.731
│       ├── so101_old_calib.xml # inactive calib (virtual zero = fully extended); kp=17.8
│       ├── joints_properties.xml # ORPHANED + stale — included nowhere; edits have no effect
│       └── assets/             # 14 .stl: 13 arm meshes + coffee_mug.stl scene prop (meshdir relative)
└── scripts/
    ├── sim.py              # headless sinusoidal demo → PNGs + GIF into outputs/
    └── view.py             # interactive X11 viewer (glfw; --manual / --max-time)
```

`outputs/` is gitignored and regenerable — never commit it.

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Change the model (joints, actuators, meshes) | `models/so101/*.xml` (keep XMLs + `assets/` together) |
| Switch calibration | `<include file="..." />` line in `models/so101/scene.xml` |
| Add a scene object (prop) | `models/so101/scene.xml` (`<asset>` + free-joint body; never in the calib XMLs) |
| Change simulation behavior / rendering | `scripts/sim.py` |
| Change viewer behavior | `scripts/view.py` |
| Add Python dependencies | `requirements.txt` (keep unpinned `>=`) |
| Change container image / system deps | `Dockerfile` (base is `python:3.12-slim`; apt GL/EGL/osmesa/GLFW libs) |
| Change container entrypoint | `entrypoint.sh` (must NOT export `MUJOCO_GL` — see CONVENTIONS) |
| Change compose / GPU override | `docker-compose.yaml` / `docker-compose.gpu.yaml` (`mira-arm` service) |
| Plan RL work / learn RL | `docs/RL_ROADMAP.md` (phases 0–11) |

## CODE MAP

`scripts/` — 2 standalone scripts, no shared module (small duplication, see NOTES):

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `_pick_gl` | fn | `sim.py:21` | Picks `MUJOCO_GL`: explicit env var → `egl` if `nvidia-smi` present → `osmesa`; **must** run before `import mujoco` |
| `_sinusoidal_targets` | fn | `sim.py:42` | Per-actuator sinusoid: mid + 0.6·half·sin(freq·t + phase), clipped to ctrl range |
| `_setup_camera` | fn | `sim.py:63` | Free camera framing arm (lookat z=0.18, dist 0.55, az 135°) |
| `run` | fn | `sim.py:76` | Sim loop: step physics, render 640×480 @30fps, save frame_NNNN.png + pose_{start,mid,end}.png + output_demo.gif, print diagnostics |
| `main` | fn | `sim.py:231` | CLI: `--duration` (6.0s), `--out`, `--no-gif` |
| `_targets` | fn | `view.py:18` | Same trajectory as `_sinusoidal_targets` (duplicate, not shared) |
| `main` | fn | `view.py:28` | `launch_passive` viewer; auto-drive unless `--manual`; `--max-time` auto-close; ends with `os._exit(0)` |

Key constants: `REPO_ROOT`/`MODEL_DIR`/`SCENE_XML` from `Path(__file__).resolve().parent.parent` in both scripts; `RENDER_W/H=640/480`, `FPS_RENDER=30` in `sim.py`.

`models/so101/` — MJCF chain: `scene.xml` `<include>`s the active calibration XML (currently `so101_new_calib.xml`), which **inlines its own `sts3215` defaults**. `joints_properties.xml` is **orphaned** (included nowhere) *and stale* — its `sts3215` class (`kp=17.8`, `kv=0.0`, `forcerange ±3.35`) matches neither calibration, so edits there have no effect. New calib: `kp=998.22`, `kv=2.731`, class `forcerange ±2.94` overridden per-actuator to `±3.35`; old calib: `kp=17.8`. Both define a `backlash` (±0.5°) class that **no element references**. The real calib difference is the joint `range` / actuator `ctrlrange` values — `shoulder_lift` and `elbow_flex` shift most (the old fully-extended zero is asymmetric); `shoulder_pan`/`wrist_flex`/`gripper` are near-identical. `scene.xml` has no `<compiler>` of its own → it inherits `angle="radian" meshdir="assets"` from the included calib XML. Calibration XMLs are onshape-to-robot generated from **different Onshape element IDs** — never deduplicate.

## COMMANDS

```bash
python scripts/sim.py            # headless demo → outputs/ (PNGs + GIF)
python scripts/sim.py --out /tmp/out --no-gif
python scripts/view.py           # interactive X11 viewer (auto demo)
python scripts/view.py --manual  # drag sliders in the left panel
python -m py_compile scripts/sim.py scripts/view.py   # syntax smoke test

docker compose up -d                                    # CPU container → bash shell
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d   # NVIDIA GPU host
docker compose exec mira-arm bash                       # interactive shell
docker compose exec mira-arm python scripts/view.py     # viewer in container (needs: xhost +local:root)
docker compose config                                   # validate compose files
bash -n entrypoint.sh                                   # entrypoint syntax check
```

## CONVENTIONS

- `sim.py` sets `MUJOCO_GL` **before** `import mujoco` via `_pick_gl` (explicit env var wins; else EGL if `nvidia-smi` present, else osmesa). Preserve that ordering.
- `view.py` defaults to `glfw` (windowed) unless `MUJOCO_GL` is set.
- Renderer selection lives in the scripts, not the container: `sim.py` auto-picks EGL (NVIDIA) vs osmesa, `view.py` uses a glfw window when `MUJOCO_GL` is unset. `entrypoint.sh` must **not** export `MUJOCO_GL`, or it would force `view.py` off its window.
- Compose image/container/service are all named `mira-arm`; `docker-compose.gpu.yaml` is an override, always used with `-f docker-compose.yaml -f docker-compose.gpu.yaml`.
- Compose env defaults: `DISPLAY=:0`, `MUJOCO_GL=` (empty on purpose so scripts self-select), `XVFB_RUN=0`, `HOST_UID/GID=1000`; `ipc: host`, `shm_size: 2gb`, `memlock: -1` are deliberate MuJoCo/rendering settings — not cruft.
- The base image intentionally has **no torch/RL deps**; add them only when the roadmap lands.
- Paths are derived from `Path(__file__).resolve().parent.parent` (repo root) — scripts work from anywhere.
- `requirements.txt` is unpinned (`>=`), matching the parent repo convention.
- `outputs/` is a regenerable artifact directory — delete freely, never commit.

## ANTI-PATTERNS (THIS PROJECT)

- Never move `assets/` away from the XMLs — `meshdir="assets"` and `<include file="...">` are relative to the XML directory.
- Never re-type or reformat the XML/STL files — copy them verbatim.
- Never export `MUJOCO_GL` from `entrypoint.sh` — it forces `view.py` off its GLFW window.
- Never run `docker-compose.gpu.yaml` standalone — it is an override (needs `-f docker-compose.yaml -f ...`).
- Never hand-edit the calibration XMLs as if they were source — they are `onshape-to-robot` output from distinct Onshape element IDs.
- Never expect edits to `joints_properties.xml` to take effect — it is included nowhere.

## VERIFICATION

No lint/typecheck/test targets exist. Smoke test: `python -m py_compile scripts/sim.py scripts/view.py` passes, `bash -n entrypoint.sh` passes, `docker compose config` validates both compose files, and `python scripts/sim.py` completes with non-blank frames (the script's own mid-frame mean-pixel check reports `non-blank`).

## NOTES

- `view.py:_targets` is a hand-duplicated copy of `sim.py:_sinusoidal_targets` + the freqs/phases arrays — change the demo trajectory in **both** files.
- Gripper (actuator index 5) runs a slower sinusoid (freq 0.3) for a visible open/close cycle.
- LeRobot treats the gripper as a linear joint (`0`=closed, `100`=open); this mapping is **not yet reflected** in the MJCF.
- `view.py` ends with `os._exit(0)` — skips interpreter teardown once `launch_passive` closes; don't "fix" it into a normal exit path.
- Calibration XMLs reference the same 13 arm STL meshes in `assets/`; far-field/branch meshes are shared by `<include>` — never deduplicate "identical" XML blocks by hand, they encode different joint zeros.
- `assets/` holds **14** STLs: the 13 arm meshes + `coffee_mug.stl` (scene prop only, not referenced by the calib XMLs).
- `scene.xml` also carries the pick-and-place props (`coffee_cup`, `marshmallow`, both free-joint bodies) — kept there deliberately so recalibration never disturbs them.
- The base body has visual-only geoms (no `collision` class) — base collision meshes were removed on purpose (bad collision behavior in sim/planning).
- `.dockerignore` mirrors `.gitignore` (plus `.git`) so `outputs/` and caches never enter the Docker build context.
- `.omo/` and `.codegraph/` are gitignored agent/editor state — never treat them as sources.
