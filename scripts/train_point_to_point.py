#!/usr/bin/env python3
"""CleanRL-style PPO on the SO-101 point-to-point task.

Task: move the end effector from fixed point_A to fixed point_B
(`scene_with_markers.xml`). Single-file implementation so every term of the
PPO loss is visible. Run:

    python scripts/train_point_to_point.py --total-timesteps 500000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mira  # noqa: F401  (registers MIRA/SO101PointToPoint-v0)
from mira.envs.so101_point_to_point_env import SO101PointToPointEnv
from mira.wrappers import NormalizeObservation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value = nn.Linear(hidden, 1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(self.critic(x)).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        h = self.actor(x)
        mean = self.mean(h)
        std = self.log_std.exp().clamp(1e-3, 1.0)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(self.critic(x)).squeeze(-1)
        return action, log_prob, entropy, value


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def save_norm_stats(envs, path: Path) -> None:
    """Pool the per-env Welford stats of NormalizeObservation into `path` (.npz).

    The policy is trained on normalized observations, so evaluation/visualization
    needs the same mean/variance. `NormalizeObservation.var` holds M2 (sum of
    squared deviations), not the variance, so convert before writing.
    """
    means = np.asarray(envs.get_attr("mean"), dtype=np.float64)
    m2 = np.asarray(envs.get_attr("var"), dtype=np.float64)
    counts = np.asarray(envs.get_attr("count"), dtype=np.float64)
    if counts.sum() <= 0:
        raise RuntimeError("no observation statistics collected")

    total = counts.sum()
    mean = (means * counts[:, None]).sum(axis=0) / total
    m2_total = m2.sum(axis=0) + (counts[:, None] * (means - mean) ** 2).sum(axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, var=m2_total / total)


def make_env(seed: int, idx: int):
    def thunk():
        env = SO101PointToPointEnv()
        env = NormalizeObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=500000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")
    num_envs = args.num_envs
    num_steps = args.num_steps

    envs = gym.vector.SyncVectorEnv([make_env(args.seed + i, i) for i in range(num_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    agent = ActorCritic(obs_dim, act_dim).to(device)
    for m in agent.modules():
        if isinstance(m, nn.Linear):
            layer_init(m)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    global_step = 0
    num_updates = args.total_timesteps // (num_envs * num_steps)
    t0 = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)

    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            nobs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(nobs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs) * (1.0 - next_done)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten
        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, act_dim)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = num_envs * num_steps
        minibatch = args.minibatch_size
        inds = np.arange(batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch):
                mb = inds[start:start + minibatch]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb])
                logratio = new_logprob - b_logprobs[mb]
                ratio = logratio.exp()

                pg_loss1 = -b_advantages[mb] * ratio
                pg_loss2 = -b_advantages[mb] * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                v_loss_unclipped = (new_value - b_returns[mb]) ** 2
                v_clipped = b_values[mb] + torch.clamp(new_value - b_values[mb],
                                                      -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % 5 == 0 or update == num_updates:
            sps = int(global_step / (time.time() - t0))
            print(f"update={update:>4} global_step={global_step:>8} sps={sps} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

        if args.eval_every > 0 and update % args.eval_every == 0:
            eval_env = SO101PointToPointEnv()
            eval_env = NormalizeObservation(eval_env, update_running_mean=False)
            try:
                means = np.asarray(envs.get_attr("mean"))
                vars_ = np.asarray(envs.get_attr("var"))
                counts = np.asarray(envs.get_attr("count"))
                eval_env.mean = means[0].copy()
                eval_env.var = vars_[0].copy()
                eval_env.count = int(counts[0])
            except Exception:
                pass
            successes = 0
            final_dists = []
            for _ in range(args.eval_episodes):
                obs_e, info_e = eval_env.reset()
                done_e = False
                while not done_e:
                    obs_t = torch.as_tensor(obs_e, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(obs_t)
                    action = action.squeeze(0).cpu().numpy()
                    obs_e, reward_e, term_e, trunc_e, info_e = eval_env.step(action)
                    done_e = term_e or trunc_e
                if info_e.get("is_success", False):
                    successes += 1
                final_dists.append(info_e.get("distance", float("nan")))
            success_rate = successes / args.eval_episodes if args.eval_episodes > 0 else 0.0
            mean_final_dist_cm = float(np.nanmean(final_dists)) * 100.0 if final_dists else float("nan")
            print(f"eval update={update} success_rate={success_rate:.3f} mean_final_dist_cm={mean_final_dist_cm:.2f}", flush=True)
            eval_env.close()

    save_path = REPO_ROOT / "outputs" / "ppo_point_to_point.pt"
    norm_path = save_path.with_name(f"{save_path.stem}_norm.npz")
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), save_path)
        save_norm_stats(envs, norm_path)
        print(f"saved {save_path} + {norm_path.name}")
    except (OSError, RuntimeError, PermissionError):
        fallback = Path("/tmp/opencode/ppo_point_to_point.pt")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), fallback)
        save_norm_stats(envs, fallback.with_name(f"{fallback.stem}_norm.npz"))
        print(f"outputs/ not writable; saved {fallback} + ppo_point_to_point_norm.npz")
    envs.close()


if __name__ == "__main__":
    main()
