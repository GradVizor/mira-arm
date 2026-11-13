#!/usr/bin/env python3
"""Train SAC on SO-101 pick-and-place with a four-stage curriculum.

Stage 1: fixed marshmallow and cup.
Stage 2: random marshmallow yaw and xy, fixed cup.
Stage 3: random cup xy too.
Stage 4: random marshmallow mass and friction.

Stage 1 has an optional discovery curriculum (``--stage1-curriculum``) that
starts the arm at the grasp pose and anneals the pre-grasp height back to the
normal 9 cm.  The policy, observation-normalizer stats, and reward-normalizer
stats are chained between sub-runs.

Examples:
    python3 scripts/train_pick_place.py --stage 1 --total-timesteps 100000 --warm-start --pregrasp-height 0.0
    python3 scripts/train_pick_place.py --stage1-curriculum --stage-timesteps 100000
    python3 scripts/train_pick_place.py --stage 1 --init-model outputs/sac_pick_place/stage1/final_model.zip
    python3 scripts/train_pick_place.py --curriculum --stage-timesteps 200000 --promote-success 0.5

Reward normalization uses ``VecNormalize(norm_obs=False, norm_reward=True)`` so
the existing ``NormalizeObservation`` wrapper remains the single source of
observation normalization.  ``clip_reward`` defaults to 50 (not 10): with a
return std near 0.15 the old clip collapsed the +2 grasp and +10 success
bonuses to the same value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mira.envs.so101_pick_place_env import SO101PickPlaceEnv
from mira.wrappers import NormalizeObservation

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception as exc:
    print(f"Missing RL dependencies: {exc}")
    print("Install with: pip install -r requirements.txt -r requirements-rl.txt")
    sys.exit(1)


STAGE_CONFIGS: dict[int, dict[str, bool]] = {
    1: {
        "fix_cup": True,
        "randomize_box": False,
        "randomize_box_pos": False,
        "randomize_cup_pos": False,
        "randomize_physics": False,
    },
    2: {
        "fix_cup": True,
        "randomize_box": True,
        "randomize_box_pos": True,
        "randomize_cup_pos": False,
        "randomize_physics": False,
    },
    3: {
        "fix_cup": False,
        "randomize_box": True,
        "randomize_box_pos": True,
        "randomize_cup_pos": True,
        "randomize_physics": False,
    },
    4: {
        "fix_cup": False,
        "randomize_box": True,
        "randomize_box_pos": True,
        "randomize_cup_pos": True,
        "randomize_physics": True,
    },
}


def make_env(
    stage: int,
    seed: int,
    training: bool,
    warm_start: bool = False,
    pregrasp_height: float = 0.09,
    reward_kwargs: dict[str, float] | None = None,
    task_mode: str = "full",
    time_penalty: float = 0.0,
    lift_scale: float = 20.0,
    transport_scale: float = 10.0,
):
    env = Monitor(
        SO101PickPlaceEnv(
            **STAGE_CONFIGS[stage],
            warm_start=warm_start,
            pregrasp_height=pregrasp_height,
            task_mode=task_mode,
            time_penalty=time_penalty,
            lift_scale=lift_scale,
            transport_scale=transport_scale,
            **(reward_kwargs or {}),
        )
    )
    env.reset(seed=seed)
    return NormalizeObservation(env, update_running_mean=training)


def get_normalizer(env) -> NormalizeObservation:
    while True:
        if isinstance(env, NormalizeObservation):
            return env
        if isinstance(env, VecNormalize):
            env = env.venv
        elif isinstance(env, DummyVecEnv):
            env = env.envs[0]
        else:
            env = env.env


def copy_normalizer(src: NormalizeObservation, dst: NormalizeObservation) -> None:
    dst.mean = src.mean.copy()
    dst.var = src.var.copy()
    dst.count = src.count


def save_normalizer(path: Path, normalizer: NormalizeObservation) -> None:
    np.savez(
        path,
        mean=normalizer.mean,
        var=normalizer.var,
        count=np.array(normalizer.count),
    )


def load_normalizer(path: Path, normalizer: NormalizeObservation) -> None:
    if not path.exists():
        return
    data = np.load(path)
    normalizer.mean = data["mean"].copy()
    normalizer.var = data["var"].copy()
    normalizer.count = int(data["count"])


def evaluate(model: SAC, env, episodes: int) -> dict[str, float]:
    """Deterministic evaluation with the diagnostics needed to debug failures."""
    successes = 0
    returns: list[float] = []
    min_pad: list[float] = []
    min_grip: list[float] = []
    closed_at_box = 0
    both_touch = 0
    ever_grasped = 0
    max_box_z: list[float] = []
    final_xy: list[float] = []
    saturation: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False
        info: dict = {}
        ep_min_pad = float("inf")
        ep_min_grip = float("inf")
        ep_closed = False
        ep_both = False
        ep_max_z = float("-inf")
        sat_steps = 0
        n_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            a = np.asarray(action).reshape(-1)
            obs, reward, terminated, truncated, info = env.step(a)
            episode_return += float(reward)
            ep_min_pad = min(ep_min_pad, float(info.get("pad_box_dist", float("inf"))))
            ep_min_grip = min(ep_min_grip, float(info.get("gripper_q", float("inf"))))
            ep_closed = ep_closed or bool(info.get("closed_at_box", False))
            ep_both = ep_both or bool(info.get("both_touch", False))
            ep_max_z = max(ep_max_z, float(info["box_pos"][2]))
            sat_steps += int(abs(float(a[5])) > 0.95)
            n_steps += 1
            done = terminated or truncated

        returns.append(episode_return)
        successes += int(bool(info.get("is_success", False)))
        min_pad.append(ep_min_pad)
        min_grip.append(ep_min_grip)
        closed_at_box += int(ep_closed)
        both_touch += int(ep_both)
        ever_grasped += int(bool(info.get("ever_grasped", False)))
        max_box_z.append(ep_max_z)
        final_xy.append(float(info.get("box_cup_dist_xy", float("nan"))))
        saturation.append(sat_steps / max(1, n_steps))

    return {
        "success_rate": successes / max(1, episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "min_pad_box_dist": float(np.min(min_pad)) if min_pad else float("nan"),
        "min_gripper_q": float(np.min(min_grip)) if min_grip else float("nan"),
        "closed_at_box_rate": closed_at_box / max(1, episodes),
        "both_touch_rate": both_touch / max(1, episodes),
        "ever_grasped_rate": ever_grasped / max(1, episodes),
        "max_box_z": float(np.max(max_box_z)) if max_box_z else float("nan"),
        "final_xy": float(np.nanmean(final_xy)) if final_xy else float("nan"),
        "gripper_saturation": float(np.mean(saturation)) if saturation else float("nan"),
    }


DIAG_KEYS = (
    "min_pad_box_dist",
    "min_gripper_q",
    "closed_at_box_rate",
    "both_touch_rate",
    "ever_grasped_rate",
    "max_box_z",
    "final_xy",
    "gripper_saturation",
)


class EvalSuccessCallback(BaseCallback):
    """Periodic deterministic evaluation with success-rate + diagnostics logging."""

    def __init__(self, eval_env, eval_episodes: int, eval_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = max(1, eval_freq)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        copy_normalizer(get_normalizer(self.training_env), get_normalizer(self.eval_env))
        summary = evaluate(self.model, self.eval_env, self.eval_episodes)
        for key, value in summary.items():
            self.logger.record(f"eval/{key}", value)
        print(
            f"[eval @ {self.num_timesteps}] "
            f"success={summary['success_rate']:.1%} "
            f"return={summary['mean_return']:.2f} "
            f"min_pad={summary['min_pad_box_dist']:.4f} "
            f"min_grip={summary['min_gripper_q']:.3f} "
            f"closed_at_box={summary['closed_at_box_rate']:.0%} "
            f"grasped={summary['ever_grasped_rate']:.0%}"
        )
        return True


class EntropyFloorCallback(BaseCallback):
    """Bound the SAC entropy coefficient so it neither collapses nor runs away."""

    def __init__(self, floor: float, ceiling: float = 0.0, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.floor = float(floor)
        self.ceiling = float(ceiling)

    def _on_step(self) -> bool:
        # SB3 stores entropy as log_ent_coef when ent_coef="auto"; clamp that.
        log_tensor = getattr(self.model, "log_ent_coef", None)
        if log_tensor is not None:
            with torch.no_grad():
                lo = float(np.log(max(self.floor, 1e-12)))
                if self.ceiling > 0.0:
                    hi = float(np.log(self.ceiling))
                    log_tensor.data.clamp_(min=lo, max=hi)
                else:
                    log_tensor.data.clamp_(min=lo)
        tensor = getattr(self.model, "ent_coef_tensor", None)
        if tensor is not None:
            with torch.no_grad():
                if self.ceiling > 0.0:
                    tensor.data.clamp_(min=self.floor, max=self.ceiling)
                else:
                    tensor.data.clamp_(min=self.floor)
        return True


def train_stage(
    stage: int,
    timesteps: int,
    args: argparse.Namespace,
    *,
    pregrasp_height: float | None = None,
    init_dir: Path | None = None,
    init_model_path: Path | None = None,
    tag: str | None = None,
) -> tuple[float, float, dict]:
    height = args.pregrasp_height if pregrasp_height is None else float(pregrasp_height)
    name = f"stage{stage}" if tag is None else f"stage{stage}_{tag}"
    stage_dir = args.out / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed + stage * 100

    warm_start = bool(args.warm_start) and stage == 1
    if args.warm_start and stage >= 2:
        print(f"warm start disabled for stage {stage} (per-reset IK ~3 s with randomized box)")

    reward_kwargs = {
        "closure_gate_radius": args.closure_gate_radius,
        "closure_scale": args.closure_scale,
        "approach_scale": args.approach_scale,
    }

    venv = DummyVecEnv(
        [
            lambda: make_env(
                stage,
                seed,
                training=True,
                warm_start=warm_start,
                pregrasp_height=height,
                reward_kwargs=reward_kwargs,
                task_mode=args.task_mode,
                time_penalty=args.time_penalty,
                lift_scale=args.lift_scale,
                transport_scale=args.transport_scale,
            )
        ]
    )
    prev_vec = init_dir / "vecnormalize.pkl" if init_dir is not None else None
    if prev_vec is not None and prev_vec.exists():
        train_env = VecNormalize.load(str(prev_vec), venv)
        train_env.norm_reward = True
        train_env.clip_reward = args.clip_reward
        train_env.gamma = args.gamma
    else:
        train_env = VecNormalize(
            venv,
            norm_obs=False,
            norm_reward=True,
            clip_reward=args.clip_reward,
            gamma=args.gamma,
        )

    normalizer = get_normalizer(train_env)
    if init_dir is not None:
        load_normalizer(init_dir / "norm_stats.npz", normalizer)

    model_path = init_model_path
    if model_path is None and init_dir is not None:
        candidate = init_dir / "final_model.zip"
        model_path = candidate if candidate.exists() else None

    if model_path is not None and Path(model_path).exists():
        print(f"Loading model from {model_path}")
        model = SAC.load(
            str(model_path),
            env=train_env,
            device=args.device,
            tensorboard_log=str(args.out / "tb" / name),
        )
        # ``load`` restores the old gamma / target entropy; honour the new args.
        model.gamma = args.gamma
        model.target_entropy = args.target_entropy
        learn_timesteps = model.num_timesteps + timesteps
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_entropy=args.target_entropy,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=str(args.out / "tb" / name),
            seed=seed,
            device=args.device,
            verbose=1,
        )
        learn_timesteps = timesteps

    checkpoint = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(stage_dir),
        name_prefix=f"sac_stage{stage}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
    )

    eval_env = make_env(
        stage,
        seed + 9999,
        training=False,
        warm_start=warm_start,
        pregrasp_height=height,
        reward_kwargs=reward_kwargs,
        task_mode=args.task_mode,
                time_penalty=args.time_penalty,
                lift_scale=args.lift_scale,
                transport_scale=args.transport_scale,
    )
    eval_callback = EvalSuccessCallback(
        eval_env,
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
    )
    callbacks: list[BaseCallback] = [checkpoint, eval_callback]
    if args.ent_coef_floor > 0.0 or args.ent_coef_ceiling > 0.0:
        callbacks.append(
            EntropyFloorCallback(args.ent_coef_floor, args.ent_coef_ceiling)
        )
    callback_list = CallbackList(callbacks)

    print(f"\n=== Stage {stage}: {STAGE_CONFIGS[stage]} ===")
    print(f"Training {timesteps} steps -> {stage_dir}")
    print(
        f"gamma={args.gamma} pregrasp_height={height:.3f} warm_start={warm_start} "
        f"target_entropy={args.target_entropy} "
        f"reward_norm=VecNormalize(norm_reward=True, clip_reward={args.clip_reward}) "
        f"ent_coef_floor={args.ent_coef_floor} "
        f"ent_coef_ceiling={args.ent_coef_ceiling}"
    )
    print(
        f"reward: approach_scale={args.approach_scale} "
        f"closure_scale={args.closure_scale} "
        f"closure_gate_radius={args.closure_gate_radius}"
    )
    model.learn(
        total_timesteps=learn_timesteps,
        callback=callback_list,
        log_interval=args.log_interval,
        progress_bar=False,
        reset_num_timesteps=False,
    )

    model.save(stage_dir / "final_model")
    save_normalizer(stage_dir / "norm_stats.npz", normalizer)
    train_env.save(str(stage_dir / "vecnormalize.pkl"))

    copy_normalizer(normalizer, get_normalizer(eval_env))
    summary = evaluate(model, eval_env, args.eval_episodes)
    eval_env.close()

    neutral_env = make_env(
        stage,
        seed + 8888,
        training=False,
        warm_start=False,
        reward_kwargs=reward_kwargs,
        task_mode=args.task_mode,
                time_penalty=args.time_penalty,
                lift_scale=args.lift_scale,
                transport_scale=args.transport_scale,
    )
    copy_normalizer(normalizer, get_normalizer(neutral_env))
    neutral_summary = evaluate(model, neutral_env, args.eval_episodes)
    neutral_env.close()
    train_env.close()

    record = {
        "stage": stage,
        "timesteps": timesteps,
        "total_timesteps": int(model.num_timesteps),
        "pregrasp_height": height,
        "warm_start": warm_start,
        "gamma": args.gamma,
        "target_entropy": args.target_entropy,
        "clip_reward": args.clip_reward,
        "ent_coef_floor": args.ent_coef_floor,
        "ent_coef_ceiling": args.ent_coef_ceiling,
        "approach_scale": args.approach_scale,
        "closure_scale": args.closure_scale,
        "closure_gate_radius": args.closure_gate_radius,
        "config": STAGE_CONFIGS[stage],
        "warm_eval": summary,
        "neutral_eval": neutral_summary,
    }
    (stage_dir / "summary.json").write_text(json.dumps(record, indent=2))
    print(
        f"Stage {stage}: warm success={summary['success_rate']:.1%} "
        f"neutral success={neutral_summary['success_rate']:.1%} "
        f"min_pad={summary['min_pad_box_dist']:.4f} "
        f"grasped={summary['ever_grasped_rate']:.0%}"
    )
    return summary["success_rate"], neutral_summary["success_rate"], record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--stage-timesteps", type=int, default=100_000)
    parser.add_argument("--promote-success", type=float, default=0.5)
    parser.add_argument("--stage1-curriculum", action="store_true")
    parser.add_argument(
        "--stage1-heights",
        default="0.0,0.03,0.09",
        help="comma-separated pre-grasp heights for the stage-1 discovery curriculum",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--target-entropy", type=float, default=-3.0)
    parser.add_argument("--clip-reward", type=float, default=50.0)
    parser.add_argument("--ent-coef-floor", type=float, default=0.05)
    parser.add_argument("--ent-coef-ceiling", type=float, default=0.5)
    parser.add_argument("--closure-gate-radius", type=float, default=0.02)
    parser.add_argument("--closure-scale", type=float, default=4.0)
    parser.add_argument("--approach-scale", type=float, default=5.0)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument(
        "--pregrasp-height",
        type=float,
        default=0.09,
        help="reset height above the grasp pose (0.0 = grasp pose)",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="continue training from this .zip (also loads sibling norm stats)",
    )
    parser.add_argument("--task-mode", choices=("full", "lift_only"), default="full",
                        help="full pick-and-place or grasp+lift bootstrap")
    parser.add_argument("--time-penalty", type=float, default=0.001,
                        help="per-step time penalty (default 0.001)")
    parser.add_argument("--lift-scale", type=float, default=20.0,
                        help="dense lift shaping scale while grasped (default 20.0)")
    parser.add_argument("--transport-scale", type=float, default=10.0,
                        help="transport shaping scale while grasped (default 10.0)")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("outputs/sac_pick_place"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.stage1_curriculum:
        args.warm_start = True
        heights = [float(value) for value in args.stage1_heights.split(",") if value]
        init_dir: Path | None = None
        for height in heights:
            tag = f"h{height:.2f}"
            train_stage(
                1,
                args.stage_timesteps,
                args,
                pregrasp_height=height,
                init_dir=init_dir,
                tag=tag,
            )
            init_dir = args.out / f"stage1_{tag}"
        return

    if not args.curriculum:
        init_dir = Path(args.init_model).parent if args.init_model is not None else None
        train_stage(
            args.stage,
            args.total_timesteps,
            args,
            init_model_path=args.init_model,
            init_dir=init_dir,
        )
        return

    for stage in range(1, 5):
        success_rate, _, _ = train_stage(stage, args.stage_timesteps, args)
        if success_rate < args.promote_success:
            print(
                f"Stopping curriculum: stage {stage} success "
                f"{success_rate:.1%} < {args.promote_success:.1%}"
            )
            break


if __name__ == "__main__":
    main()
