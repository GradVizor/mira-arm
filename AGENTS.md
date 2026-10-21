# MIRA — PROJECT KNOWLEDGE BASE

**Generated:** 2026-10-14
**Commit:** d33014e
**Branch:** main

MIRA (Multimodal Intelligent Robotic Arm) — personal robotic-arm repo based on the SO-ARM101 (SO-101) 6-DOF arm. MuJoCo models, headless simulation, and an interactive viewer; a platform for multimodal (vision + language + action) robotics research.

## OVERVIEW

Contains a MuJoCo (MJCF) model of the SO-101 arm, Python scripts (headless sim, interactive viewer, RL training), a Gymnasium env package, and a from-scratch tabular RL learner. Script-style self-checks (`test/*.py`), no linter, CI, or lockfile.

## STRUCTURE

```
./
├── README.md               # project overview, hardware, calibration, quick start
├── AGENTS.md               # this file
├── .gitignore              # outputs/, caches, model artifacts, agent state
├── requirements.txt        # unpinned deps: mujoco, numpy, pillow, gymnasium
├── requirements-rl.txt     # RL extras: torch, SB3, sb3_contrib, tensorboard, matplotlib
├── Dockerfile              # lean python:3.12-slim image: sim + viewer (EGL/osmesa/GLFW libs)
├── docker-compose.yaml     # CPU compose — mira-arm service, X11 socket mount
├── docker-compose.gpu.yaml # NVIDIA override — gpus: all (needs NVIDIA Container Toolkit)
├── entrypoint.sh           # stack smoke-check, optional Xvfb, HOST_UID/GID chown
├── .dockerignore           # keeps outputs/, .git, caches out of the build context
├── docs/
│   ├── RL_ROADMAP.md       # zero→VLA RL learning roadmap (see WHERE TO LOOK)
│   ├── GYMNASIUM_ENV.md    # Gymnasium interface contract + flowcharts
│   └── TABULAR_RL.md       # TD-learning study guide (RL_ROADMAP Phase 2)
├── models/
│   └── so101/              # MJCF model — XML + assets/ MUST stay together
│       ├── AGENTS.md           # model internals (classes, gains, ctrlrange, props)
│       ├── scene.xml           # entry point; <include>s active calib + scene props (coffee_cup, marshmallow)
│       ├── scene_with_coffee_mug.xml # prop scene variant (mug + marshmallow)
│       ├── scene_with_markers.xml     # A/B target-marker scene variant
│       ├── so101_new_calib.xml # ACTIVE calib (virtual zero = mid-range)
│       ├── so101_old_calib.xml # inactive calib (virtual zero = fully extended)
│       ├── joints_properties.xml # ORPHANED + stale — included nowhere; edits have no effect
│       └── assets/             # 14 .stl: 13 arm meshes + coffee_mug.stl scene prop (meshdir relative)
├── scripts/
│   ├── sim.py              # headless sinusoidal demo → PNGs + GIF into outputs/
│   ├── view.py             # interactive X11 viewer (glfw; --manual / --max-time)
│   ├── watch_gym.py        # MuJoCo viewer driven by the Gymnasium env
│   ├── train_reach.py      # PPO training stub (SB3)
│   └── train_tabular.py    # Phase 2 CLI: tabular Q-learning / SARSA on toy-text envs
├── test/
│   ├── test_gym.py         # script-style smoke test of MIRA/SO101Reach-v0
│   └── test_tabular_rl.py  # 13 self-checks for the Phase 2 learners
└── mira/                   # Gymnasium env package (RL — RL_ROADMAP Phase 1)
    ├── __init__.py         # registers env id "MIRA/SO101Reach-v0"
    ├── envs/
    │   └── so101_reach_env.py  # SO101ReachEnv: reach task over models/so101/scene.xml
    ├── rl/                 # Tabular RL learners (RL — RL_ROADMAP Phase 2)
    │   ├── __init__.py     # re-exports the public API
    │   └── tabular.py      # from-scratch Q-learning + SARSA (numpy only, no SB3)
    └── wrappers.py         # NormalizeObservation (Welford), ActionScale
```

`outputs/` is gitignored and regenerable — never commit it.

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Change the model (joints, actuators, meshes) | `models/so101/*.xml` (keep XMLs + `assets/` together) |
| Switch calibration | `<include file="..." />` line in `models/so101/scene.xml` |
| Add a scene object (prop) | `models/so101/scene.xml` (`<asset>` + free-joint body; never in the calib XMLs) |
| Model internals: default classes, gains, ctrlrange, props | `models/so101/AGENTS.md` (child knowledge base) |
| Change simulation behavior / rendering | `scripts/sim.py` |
| Change viewer behavior | `scripts/view.py` |
| Add Python dependencies | `requirements.txt` (keep unpinned `>=`) |
| Add RL-only dependencies (torch, SB3) | `requirements-rl.txt` — never in the base image |
| Change container image / system deps | `Dockerfile` (base is `python:3.12-slim`; apt GL/EGL/osmesa/GLFW libs) |
| Change container entrypoint | `entrypoint.sh` (must NOT export `MUJOCO_GL` — see CONVENTIONS) |
| Change compose / GPU override | `docker-compose.yaml` / `docker-compose.gpu.yaml` (`mira-arm` service) |
| Plan RL work / learn RL | `docs/RL_ROADMAP.md` (phases 0–11) |
| RL: Gymnasium reach env / wrappers | `mira/envs/so101_reach_env.py`, `mira/wrappers.py` (env id `MIRA/SO101Reach-v0`) |
| RL: how Gymnasium works + usage guide | `docs/GYMNASIUM_ENV.md` |
| RL: tabular TD learners (Phase 2) | `mira/rl/tabular.py` (Q-learning + SARSA), `scripts/train_tabular.py` (CLI) |
| RL: TD-learning theory + checkpoint answers | `docs/TABULAR_RL.md` |

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

`mira/rl/tabular.py` — Phase 2 learners, pure numpy + the Gymnasium API (no SB3):

| Symbol | Type | Role |
|--------|------|------|
| `TDConfig` | dataclass | `alpha`, `gamma`, `epsilon_start/end/decay`, `algorithm` (`q_learning` \| `sarsa`) |
| `TabularAgent` | class | Q-table `q[s,a]`; `select_action` (ε-greedy), `update` (one TD(0) step), `decay_epsilon` |
| `train_tabular` | fn | Episode loop; seeds `env.reset(seed=seed+episode)` so a run is reproducible from 2 ints |
| `evaluate_greedy` | fn | Greedy-policy rollout → `EvalResult` (mean±std return, success rate, length) |
| `goal_state` / `policy_map` / `render_policy` | fn | Goal index; ASCII grid of the greedy policy over the env's tile map |
| `state_space_size` | fn | `bins ** n_joints` — the number that kills tabular RL on a 6-DOF arm |

`models/so101/` — MJCF chain: `scene.xml` `<include>`s the active calibration XML (currently `so101_new_calib.xml`), which **inlines its own `sts3215` defaults**. `joints_properties.xml` is **orphaned** (included nowhere) *and stale*, so edits there have no effect. `scene.xml` has no `<compiler>` of its own → it inherits `angle="radian" meshdir="assets"` from the included calib XML. Full model internals — default classes, gains, per-joint `ctrlrange`, props, Onshape element IDs — live in **`models/so101/AGENTS.md`**.

## COMMANDS

```bash
python scripts/sim.py            # headless demo → outputs/ (PNGs + GIF)
python scripts/sim.py --out /tmp/out --no-gif
python scripts/view.py           # interactive X11 viewer (auto demo)
python scripts/view.py --manual  # drag sliders in the left panel
python -m py_compile scripts/sim.py scripts/view.py   # syntax smoke test

python scripts/train_tabular.py --env FrozenLake-v1 --compare-optimal  # Phase 2: Q-learning + exact reference
python scripts/train_tabular.py --env CliffWalking-v1 --algo sarsa     # Phase 2: on-policy variant
python test/test_tabular_rl.py                       # Phase 2 self-checks (13 checks)
python test/test_gym.py                              # Gymnasium env smoke test

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
- The base image intentionally has **no torch/SB3 deps** — add them via a separate `requirements-rl.txt` when the roadmap lands. `gymnasium` IS a base dep now (the `mira/` env needs it).
- The Phase 2 tabular learner is **numpy + gymnasium only** — it must stay importable without `requirements-rl.txt` so it runs in the lean image.
- `outputs/` is a regenerable artifact directory — delete freely, never commit. It can be owned by another user (e.g. left by a root-run container) — `train_tabular.py` falls back to a temp dir instead of crashing.
- Paths are derived from `Path(__file__).resolve().parent.parent` (repo root) — scripts work from anywhere.
- `requirements.txt` is unpinned (`>=`), matching the parent repo convention.

## ANTI-PATTERNS (THIS PROJECT)

- Never move `assets/` away from the XMLs — `meshdir="assets"` and `<include file="...">` are relative to the XML directory.
- Never re-type or reformat the XML/STL files — copy them verbatim.
- Never export `MUJOCO_GL` from `entrypoint.sh` — it forces `view.py` off its GLFW window.
- Never run `docker-compose.gpu.yaml` standalone — it is an override (needs `-f docker-compose.yaml -f ...`).
- Never hand-edit the calibration XMLs as if they were source — they are `onshape-to-robot` output from distinct Onshape element IDs.
- Never expect edits to `joints_properties.xml` to take effect — it is included nowhere.

## VERIFICATION

No lint/typecheck/CI targets exist; the self-checks are scripts. Smoke test: `python -m py_compile scripts/sim.py scripts/view.py` passes, `bash -n entrypoint.sh` passes, `docker compose config` validates both compose files, and `python scripts/sim.py` completes with non-blank frames (the script's own mid-frame mean-pixel check reports `non-blank`). RL self-checks: `python test/test_tabular_rl.py` (13 checks, ~3 min — the learning ones train 50k episodes) and `python test/test_gym.py`.

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
- `mira/envs/so101_reach_env.py` reads only the **6 arm joints by name**: `scene.xml` adds two free-joint props, so `nq=20`/`nv=18` — never index `qpos[:6]`. Control dt = `model.opt.timestep × frame_skip` (0.002 × 5 = 0.01 s). Reward = `-‖gripperframe − target‖`; success < 2 cm. Gripper action maps to the LeRobot linear convention (`-1`=closed→`0`, `+1`=open→`100`).
- `FrozenLake-v1` is **slippery by default**: the best any deterministic policy can do is ≈ 0.73 success. With the shipped defaults (α = 0.1, ε floor 0.1) Q-learning lands at 0.716–0.730 across seeds — but a *mis-tuned* α (0.2) drops it to 0.28–0.72. Never report a single seed, and never "fix" a run that reaches 0.73.
- `CliffWalking-v1` puts the start at the **bottom-left** `(3,0)` with the cliff on row 3 (cols 1–10) — the classic Sutton & Barto layout is vertically mirrored. Q-learning's optimum is 13 steps (row 2, hugging the cliff); SARSA settles on 17 (row 0, safe).
