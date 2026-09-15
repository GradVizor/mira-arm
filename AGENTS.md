# AGENTS.md

MIRA (Multimodal Intelligent Robotic Arm) — personal robotic-arm repo based on the SO-ARM101 (SO-101) 6-DOF arm. MuJoCo models, headless simulation, and an interactive viewer; a platform for multimodal (vision + language + action) robotics research.

## OVERVIEW

Contains a MuJoCo (MJCF) model of the SO-101 arm, two Python scripts (headless sim + interactive viewer), and minimal project scaffolding. No tests, linter, CI, or lockfile. The Python env, GPU, and MuJoCo live in the MCP_Test dev container (`/home/gradvizor/workspace/openCodeDemo`) — never give host-Python instructions.

## STRUCTURE

```
./
├── README.md               # project overview, hardware, calibration, quick start
├── AGENTS.md               # this file
├── .gitignore              # outputs/, caches, model artifacts, agent state
├── requirements.txt        # unpinned deps: mujoco, numpy, pillow
├── models/
│   └── so101/              # MJCF model — XML + assets/ MUST stay together
│       ├── scene.xml           # entry point; <include>s the active calibration XML
│       ├── so101_new_calib.xml # default calibration (virtual zero = mid-range)
│       ├── so101_old_calib.xml # old calibration (virtual zero = fully extended)
│       ├── joints_properties.xml
│       └── assets/             # 13 .stl meshes (meshdir="assets" is relative)
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
| Change simulation behavior / rendering | `scripts/sim.py` |
| Change viewer behavior | `scripts/view.py` |
| Add Python dependencies | `requirements.txt` (keep unpinned `>=`) |

## COMMANDS

```bash
python scripts/sim.py            # headless demo → outputs/ (PNGs + GIF)
python scripts/sim.py --out /tmp/out --no-gif
python scripts/view.py           # interactive X11 viewer (auto demo)
python scripts/view.py --manual  # drag sliders in the left panel
python -m py_compile scripts/sim.py scripts/view.py   # syntax smoke test
```

Inside the MCP_Test dev container (mount this repo at `/mira-arm`):

```bash
docker compose -f /home/gradvizor/workspace/openCodeDemo/docker-compose.yaml run --rm \
  -v /home/gradvizor/workspace/mira-arm:/mira-arm \
  mujoco-rl python /mira-arm/scripts/sim.py
```

## CONVENTIONS

- `sim.py` sets `MUJOCO_GL` **before** `import mujoco` via `_pick_gl` (explicit env var wins; else EGL if `nvidia-smi` present, else osmesa). Preserve that ordering.
- `view.py` defaults to `glfw` (windowed) unless `MUJOCO_GL` is set.
- Paths are derived from `Path(__file__).resolve().parent.parent` (repo root) — scripts work from anywhere.
- `requirements.txt` is unpinned (`>=`), matching the parent repo convention.
- `outputs/` is a regenerable artifact directory — delete freely, never commit.

## ANTI-PATTERNS (THIS PROJECT)

- Never instruct `pip install`, python, or venv commands to run on the host — nothing runs there; use the dev container.
- Never move `assets/` away from the XMLs — `meshdir="assets"` and `<include file="...">` are relative to the XML directory.
- Never re-type or reformat the XML/STL files — copy them verbatim.
- Never edit the source repo `/home/gradvizor/workspace/openCodeDemo/experiments/so_arm100/` — this repo is the copy.

## VERIFICATION

No lint/typecheck/test targets exist. Smoke test: `python -m py_compile scripts/sim.py scripts/view.py` passes, and `python scripts/sim.py` completes with non-blank frames (the script's own mid-frame mean-pixel check reports `non-blank`).