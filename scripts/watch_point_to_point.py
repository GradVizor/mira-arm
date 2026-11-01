#!/usr/bin/env python3
"""Watch a trained point-to-point PPO checkpoint drive the arm.

Loads `outputs/ppo_point_to_point.pt` (state_dict saved by
`scripts/train_point_to_point.py`) and steps the point-to-point env
(SO101PointToPointEnv, fixed point_A -> point_B) in the MuJoCo passive viewer,
drawing the goal (point_B) as a sphere.

    python scripts/watch_point_to_point.py                       # outputs/ppo_point_to_point.pt
    python scripts/watch_point_to_point.py --model ppo_point_to_point.pt
    python scripts/watch_point_to_point.py --episodes 10 --stochastic
    python scripts/watch_point_to_point.py --no-viewer           # headless metrics only

Training normalizes observations and stores running stats beside the checkpoint
(`*_norm.npz`). If that file is missing, the viewer estimates stats from random
rollouts and warns (approximate only).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mujoco.viewer
import numpy as np
import torch

from mira.envs.so101_point_to_point_env import SO101PointToPointEnv

CONTROL_DT = 0.01  # frame_skip 5 * timestep 0.002


def _load_actor_class():
    spec = importlib.util.spec_from_file_location(
        "_train_point_to_point", Path(__file__).with_name("train_point_to_point.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot locate scripts/train_point_to_point.py next to this script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActorCritic


def load_agent(model_path: Path):
    state = torch.load(model_path, map_location="cpu")
    actor_cls = _load_actor_class()
    obs_dim = state["actor.0.weight"].shape[1]
    hidden = state["actor.0.weight"].shape[0]
    act_dim = state["mean.weight"].shape[0]
    agent = actor_cls(obs_dim, act_dim, hidden=hidden)
    agent.load_state_dict(state)
    agent.eval()
    return agent


def estimate_norm_stats(steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env = SO101PointToPointEnv()
    obs, _ = env.reset(seed=seed)
    samples = np.empty((steps, obs.shape[0]), dtype=np.float64)
    for i in range(steps):
        samples[i] = obs
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return samples.mean(axis=0), samples.var(axis=0)


def resolve_norm_stats(model_path: Path, explicit: str | None, steps: int, seed: int):
    if explicit:
        data = np.load(explicit)
        print(f"norm stats: {explicit}")
        return data["mean"], data["var"]

    candidate = model_path.with_name(f"{model_path.stem}_norm.npz")
    if candidate.exists():
        data = np.load(candidate)
        print(f"norm stats: {candidate}")
        return data["mean"], data["var"]

    print(
        f"WARNING: {candidate.name} not found — estimating normalization stats from "
        f"{steps} random steps.\n"
        "         The policy was trained on the running stats of its own visits, so this\n"
        "         view is approximate. Retrain with the current trainer to save exact stats."
    )
    return estimate_norm_stats(steps, seed)


def draw_marker(scene, pos: np.ndarray, rgba, size=0.015) -> None:
    if scene.maxgeom < 1:
        return
    mujoco.mjv_initGeom(
        scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, size, size]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float64),
    )
    scene.ngeom = 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="outputs/ppo_point_to_point.pt")
    p.add_argument("--episodes", type=int, default=5, help="0 = run until you close the window")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    p.add_argument("--norm-stats", type=str, default=None, help="path to *_norm.npz")
    p.add_argument("--estimate-norm-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=1.0 / CONTROL_DT, help="playback speed cap")
    p.add_argument("--no-viewer", action="store_true", help="headless: print metrics only")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        raise SystemExit(f"checkpoint not found: {model_path}")

    agent = load_agent(model_path)
    print(f"loaded {model_path}  (obs={agent.actor[0].in_features}, act={agent.mean.out_features})")

    mean, var = resolve_norm_stats(model_path, args.norm_stats, args.estimate_norm_steps, args.seed)

    env = SO101PointToPointEnv()
    obs, info = env.reset(seed=args.seed)
    goal = info["goal"].copy()
    start = info["start"].copy()

    def policy_actions(obs_raw: np.ndarray) -> np.ndarray:
        x = (obs_raw.astype(np.float64) - mean) / np.sqrt(var + 1e-8)
        x = np.clip(x, -10.0, 10.0)
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            hidden = agent.actor(xt)
            mu = agent.mean(hidden)
            if args.stochastic:
                std = agent.log_std.exp().clamp(1e-3, 1.0)
                action = torch.distributions.Normal(mu, std).sample()
            else:
                action = mu
        return np.clip(action.squeeze(0).numpy(), -1.0, 1.0)

    episode, successes, final_dist = 0, 0, []
    period = 1.0 / max(args.fps, 1.0)

    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)

    try:
        next_tick = time.time()
        while True:
            action = policy_actions(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if viewer is not None:
                # draw goal (point_B) as a green sphere
                draw_marker(viewer.user_scn, info["goal"], [0.2, 0.9, 0.2, 0.6])
                viewer.sync()
                if not viewer.is_running():
                    break
                next_tick += period
                time.sleep(max(0.0, next_tick - time.time()))

            if done:
                episode += 1
                successes += int(info.get("is_success", False))
                final_dist.append(float(info["distance"]))
                print(
                    f"episode {episode:>3}  success={bool(info.get('is_success'))}  "
                    f"final_dist={info['distance'] * 100:.1f} cm  "
                    f"success_rate={successes / episode:.3f}",
                    flush=True,
                )
                if args.episodes and episode >= args.episodes:
                    break
                obs, info = env.reset(seed=args.seed + episode)
                goal = info["goal"].copy()
                start = info["start"].copy()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    if final_dist:
        print(
            f"FINAL episodes={episode} success_rate={successes / episode:.3f} "
            f"mean_final_dist={np.mean(final_dist) * 100:.1f} cm "
            f"(success threshold = 2.0 cm)",
            flush=True,
        )
    if viewer is not None:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
