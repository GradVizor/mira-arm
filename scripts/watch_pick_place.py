#!/usr/bin/env python3
"""Watch a trained pick-and-place SAC checkpoint drive the arm in MuJoCo.

    python scripts/watch_pick_place.py
    python scripts/watch_pick_place.py --model outputs/sac_pick_place/stage1/final_model.zip
    python scripts/watch_pick_place.py --episodes 5 --stochastic
    python scripts/watch_pick_place.py --no-viewer   # headless metrics only
    python scripts/watch_pick_place.py --task-mode full --warm-start
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mujoco.viewer
import numpy as np

from stable_baselines3 import SAC
from mira.envs.so101_pick_place_env import SO101PickPlaceEnv
from mira.wrappers import NormalizeObservation

CONTROL_DT = 0.01


def load_normalizer(path: Path, normalizer: NormalizeObservation) -> None:
    if not path.exists():
        print(f"WARNING: norm stats not found at {path}; using unnormalized obs")
        return
    data = np.load(path)
    normalizer.mean = data["mean"].copy()
    normalizer.var = data["var"].copy()
    normalizer.count = int(data["count"])
    print(f"loaded norm stats: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="outputs/sac_pick_place/stage1/final_model.zip")
    p.add_argument("--norm-stats", type=str, default=None)
    p.add_argument("--episodes", type=int, default=5, help="0 = run until window closed")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--task-mode", choices=("full", "lift_only"), default="lift_only")
    p.add_argument("--warm-start", action="store_true", default=True)
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    p.add_argument("--pregrasp-height", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=1.0 / CONTROL_DT)
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        raise SystemExit(f"checkpoint not found: {model_path}")

    model = SAC.load(str(model_path), device="cpu")
    print(f"loaded {model_path}")

    env_raw = SO101PickPlaceEnv(
        task_mode=args.task_mode,
        fix_cup=True,
        warm_start=args.warm_start,
        pregrasp_height=args.pregrasp_height,
    )
    env = NormalizeObservation(env_raw, update_running_mean=False)

    norm_path = Path(args.norm_stats) if args.norm_stats else model_path.with_name("norm_stats.npz")
    if not norm_path.is_absolute():
        norm_path = REPO_ROOT / norm_path
    load_normalizer(norm_path, env)

    env_unwrapped = env.unwrapped
    mujoco_model = env_unwrapped.model
    mujoco_data = env_unwrapped.data

    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(mujoco_model, mujoco_data)

    obs, info = env.reset(seed=args.seed)
    episode = 0
    successes = 0
    final_metrics: list[dict] = []
    period = 1.0 / max(args.fps, 1.0)

    try:
        next_tick = time.time()
        while True:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            action = np.asarray(action).reshape(-1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break
                next_tick += period
                time.sleep(max(0.0, next_tick - time.time()))

            if done:
                episode += 1
                successes += int(info.get("is_success", False))
                final_metrics.append(info)
                print(
                    f"episode {episode:>3}  success={bool(info.get('is_success'))}  "
                    f"grasped={info.get('ever_grasped')}  lifted={info.get('lifted')}  "
                    f"max_z={info.get('box_pos', [0,0,0])[2]:.3f}  "
                    f"success_rate={successes / episode:.3f}",
                    flush=True,
                )
                if args.episodes and episode >= args.episodes:
                    break
                obs, info = env.reset(seed=args.seed + episode)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    if final_metrics:
        sr = successes / len(final_metrics)
        print(f"\nFINAL episodes={len(final_metrics)} success_rate={sr:.3f}")
    if viewer is not None:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
