"""Train tabular TD learners (RL_ROADMAP Phase 2) on toy-text grid worlds.

Examples::

    python scripts/train_tabular.py --env FrozenLake-v1
    python scripts/train_tabular.py --env CliffWalking-v1 --algo sarsa
    python scripts/train_tabular.py --env CliffWalking-v1 --algo q_learning --seeds 0 1 2

Outputs (per run) land in ``outputs/tabular/<env>_<algo>_seed<seed>/``:
``curve.csv`` (per-episode telemetry), ``summary.json`` and ``policy.txt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np

from mira.rl.tabular import (
    TabularAgent,
    TDConfig,
    evaluate_greedy,
    goal_state,
    policy_map,
    render_policy,
    state_space_size,
    train_tabular,
)

DEFAULT_EPISODES = {"FrozenLake-v1": 50_000, "CliffWalking-v1": 50_000}
SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def build_env(env_id: str, args: argparse.Namespace) -> gym.Env:
    """Create the toy-text env, applying the slippery / map-size switches."""
    if env_id == "FrozenLake-v1":
        return gym.make(
            env_id,
            map_name=args.map_name,
            is_slippery=not args.no_slippery,
        )
    return gym.make(env_id)


def ascii_chart(values: list[float], width: int = 64, height: int = 10) -> str:
    """Render a series as a block chart (no plotting dependency required)."""
    if not values:
        return "(no data)"
    if len(values) > width:
        chunk = len(values) / width
        values = [float(np.mean(values[int(i * chunk) : int((i + 1) * chunk)])) for i in range(width)]
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        span = 1.0
    rows = []
    for level in range(height, 0, -1):
        threshold = lo + span * (level - 0.5) / height
        rows.append("".join(SPARK_CHARS[min(len(SPARK_CHARS) - 1, int((v - lo) / span * height))] if v >= threshold else " " for v in values))
    return "\n".join(rows)


def value_iteration(env: gymnasium.Env, gamma: float, iterations: int = 5000) -> tuple[np.ndarray, dict[int, int]]:
    """Optimal value function and greedy policy of a toy-text env.

    Uses the environment's transition model (``env.unwrapped.P``). This is a
    *reference* for "how good could a policy possibly be" — the learners above
    never look at the model.
    """
    unwrapped = env.unwrapped
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    values = np.zeros(n_states, dtype=np.float64)

    for _ in range(iterations):
        updated = values.copy()
        for state in range(n_states):
            best = -np.inf
            for action in range(n_actions):
                acc = 0.0
                for prob, next_state, reward, terminated in unwrapped.P[state][action]:
                    acc += prob * (reward if terminated else reward + gamma * values[next_state])
                best = max(best, acc)
            updated[state] = best
        values = updated

    policy: dict[int, int] = {}
    for state in range(n_states):
        best_action, best_value = 0, -np.inf
        for action in range(n_actions):
            acc = 0.0
            for prob, next_state, reward, terminated in unwrapped.P[state][action]:
                acc += prob * (reward if terminated else reward + gamma * values[next_state])
            if acc > best_value:
                best_action, best_value = action, acc
        policy[state] = best_action
    return values, policy


def optimal_success_rate(env: gymnasium.Env, policy: dict[int, int], episodes: int = 500) -> float:
    """Success rate of the optimal deterministic policy (ground truth)."""
    goal = goal_state(env)
    successes = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=20_000 + episode)
        state = int(obs)
        for _ in range(1_000):
            obs, _reward, terminated, truncated, _ = env.step(policy[state])
            state = int(obs)
            if terminated:
                successes += int(state == goal)
                break
            if truncated:
                break
    return successes / episodes if episodes else 0.0


def write_curve(path: Path, stats: Any) -> None:
    moving = stats.moving_average(100)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "return", "length", "epsilon", "moving_avg_100"])
        for i, (ret, length, eps) in enumerate(zip(stats.returns, stats.lengths, stats.epsilons)):
            writer.writerow([i, ret, length, eps, moving[i]])


def print_scaling_note() -> None:
    print("\nWhy this cannot scale to the 6-DOF arm (RL_ROADMAP Phase 2 checkpoint):")
    for joints in (1, 3, 6):
        states = state_space_size(joints, 100)
        table_bytes = states * 6 * 8
        if table_bytes >= 1024**4:
            size = f"{table_bytes / 1024**4:.1f} TiB"
        elif table_bytes >= 1024**2:
            size = f"{table_bytes / 1024**2:.1f} MiB"
        else:
            size = f"{table_bytes / 1024:.1f} KiB"
        print(f"  {joints} joint(s) x 100 bins -> {states:>16,} states   (Q-table, 6 actions: {size})")
    print("  A 6-DOF arm with 100 bins per joint needs 1e12 table entries (~48 TB).")
    print("  Tabular TD also needs *discrete* actions; MIRA's action space is continuous.")


def resolve_out_dir(path: Path) -> Path:
    """Return a writable output directory.

    ``outputs/`` is the repo convention, but it can be owned by another user
    (e.g. left behind by a root-run container). Fall back to a temp dir
    instead of crashing.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("")
        probe.unlink()
        return path
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "mira-tabular"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"warning: cannot write to {path} ({exc.strerror}); using {fallback} instead")
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="FrozenLake-v1", choices=["FrozenLake-v1", "CliffWalking-v1"])
    parser.add_argument("--algo", default="q_learning", choices=["q_learning", "sarsa"])
    parser.add_argument("--episodes", type=int, default=None, help="training episodes (default: per-env)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--map-name", default="4x4", choices=["4x4", "8x8"], help="FrozenLake map size")
    parser.add_argument("--no-slippery", action="store_true", help="disable FrozenLake slipperiness")
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "tabular"))
    parser.add_argument("--compare-optimal", action="store_true", help="also solve the env exactly (value iteration)")
    args = parser.parse_args()

    episodes = args.episodes if args.episodes is not None else DEFAULT_EPISODES[args.env]
    out_root = resolve_out_dir(Path(args.out))

    print(f"env={args.env}  algo={args.algo}  episodes={episodes}  seeds={args.seeds}")
    print(
        f"alpha={args.alpha} gamma={args.gamma} "
        f"eps: {args.epsilon_start} -> {args.epsilon_end} (decay {args.epsilon_decay}/episode)"
    )

    all_success: list[float] = []
    all_returns: list[float] = []
    all_lengths: list[float] = []
    run_dirs: list[Path] = []

    for seed in args.seeds:
        env = build_env(args.env, args)
        config = TDConfig(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            algorithm=args.algo,
        )
        agent = TabularAgent(env.observation_space.n, env.action_space.n, config, seed=seed)

        stats = train_tabular(env, agent, episodes=episodes, seed=seed)
        evaluation = evaluate_greedy(env, agent, episodes=args.eval_episodes, seed=10_000)

        run_dir = out_root / f"{args.env}_{args.algo}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_curve(run_dir / "curve.csv", stats)
        (run_dir / "policy.txt").write_text(policy_map(env, agent) + "\n")
        run_dirs.append(run_dir)

        summary: dict[str, Any] = {
            "env": args.env,
            "algorithm": args.algo,
            "seed": seed,
            "episodes": episodes,
            "config": vars(args),
            "train": {
                "total_steps": stats.total_steps,
                "mean_return_last_100": float(np.mean(stats.returns[-100:])),
                "mean_length_last_100": float(np.mean(stats.lengths[-100:])),
                "final_epsilon": stats.epsilons[-1] if stats.epsilons else None,
            },
            "eval": {
                "episodes": args.eval_episodes,
                "mean_return": evaluation.mean_return,
                "std_return": evaluation.std_return,
                "success_rate": evaluation.success_rate,
                "mean_length": evaluation.mean_length,
            },
        }

        print(f"\n--- seed {seed} ---")
        curve = stats.moving_average(100)
        print("learning curve (moving average of return, last 100 episodes):")
        print(ascii_chart(curve))
        print(
            f"  scale: min={min(curve):.3f} (ep {curve.index(min(curve))}) "
            f"max={max(curve):.3f} (ep {curve.index(max(curve))}) final={curve[-1]:.3f}"
        )
        print(f"final greedy policy (arrows = greedy action, background = tiles):")
        for line in policy_map(env, agent).splitlines():
            print("   " + line)
        print(
            f"greedy eval: success={evaluation.success_rate:.3f} "
            f"return={evaluation.mean_return:.3f}+-{evaluation.std_return:.3f} "
            f"length={evaluation.mean_length:.1f}"
        )

        if args.compare_optimal:
            _values, optimal_policy = value_iteration(env, args.gamma)
            ceiling = optimal_success_rate(env, optimal_policy)
            summary["optimal_policy_success_rate"] = ceiling
            print(f"optimal-policy reference (value iteration): success={ceiling:.3f}")
            print("optimal policy (for comparison):")
            for line in render_policy(env, env.observation_space.n, optimal_policy.__getitem__).splitlines():
                print("   " + line)

        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        all_success.append(evaluation.success_rate)
        all_returns.append(evaluation.mean_return)
        all_lengths.append(evaluation.mean_length)
        env.close()

    print("\n=== summary over seeds ===")
    print(f"seeds: {args.seeds}")
    print(f"success_rate: {np.mean(all_success):.3f} +- {np.std(all_success):.3f}")
    print(f"mean_return : {np.mean(all_returns):.3f} +- {np.std(all_returns):.3f}")
    print(f"mean_length : {np.mean(all_lengths):.1f}")
    print(f"artifacts   : {', '.join(str(d) for d in run_dirs)}")

    print_scaling_note()


if __name__ == "__main__":
    main()
