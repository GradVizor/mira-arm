"""Tabular temporal-difference control from scratch (RL_ROADMAP Phase 2).

No RL library is used here on purpose. The point of this module is that the
*whole* algorithm fits in one readable file, so the mechanics of TD learning
stay visible instead of hiding behind an API.

Two learners share one update rule and differ only in the bootstrap target::

    Q-learning (off-policy):   target = r + gamma * max_a' Q(s', a')
    SARSA      (on-policy):    target = r + gamma * Q(s', a')
                               (a' = the action actually taken at s')

The environment is any Gymnasium env with a ``Discrete`` observation space and
a ``Discrete`` action space. ``FrozenLake-v1`` and ``CliffWalking-v1`` ship with
Gymnasium and are used in ``scripts/train_tabular.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import gymnasium
import numpy as np

Algorithm = Literal["q_learning", "sarsa"]

__all__ = [
    "Algorithm",
    "TDConfig",
    "TabularAgent",
    "EpisodeStats",
    "EvalResult",
    "train_tabular",
    "evaluate_greedy",
    "goal_state",
    "policy_map",
    "render_policy",
    "state_space_size",
]


@dataclass(frozen=True)
class TDConfig:
    """Hyper-parameters of a tabular TD learner.

    The defaults are the ones validated in ``docs/TABULAR_RL.md``: on
    slippery ``FrozenLake-v1`` they reach ≈ 0.72 greedy success (the proven
    optimum for a deterministic policy is ≈ 0.733) on every seed tried.

    Attributes:
        alpha: Step size of the TD update, in (0, 1].
        gamma: Discount factor, in [0, 1].
        epsilon_start: Exploration probability of the first episode.
        epsilon_end: Lower bound the exploration probability decays towards.
        epsilon_decay: Multiplicative per-episode decay of epsilon.
        algorithm: ``"q_learning"`` (off-policy) or ``"sarsa"`` (on-policy).
    """

    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.999
    algorithm: Algorithm = "q_learning"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon_start}")
        if not 0.0 <= self.epsilon_end <= 1.0:
            raise ValueError(f"epsilon_end must be in [0, 1], got {self.epsilon_end}")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}")
        if self.algorithm not in ("q_learning", "sarsa"):
            raise ValueError(f"unknown algorithm {self.algorithm!r}")


class TabularAgent:
    """A Q-table agent for discrete-state / discrete-action MDPs.

    The table is ``Q[s, a]``: the estimated return of taking action ``a`` in
    state ``s`` and then following the current policy.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: TDConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config if config is not None else TDConfig()
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.epsilon = self.config.epsilon_start
        self.n_updates = 0

    @property
    def is_on_policy(self) -> bool:
        """True for SARSA, which learns about the policy it is actually running."""
        return self.config.algorithm == "sarsa"

    def select_action(self, state: int, epsilon: float | None = None) -> int:
        """Pick an action with the epsilon-greedy rule.

        With probability ``epsilon`` act uniformly at random (exploration),
        otherwise act greedily with respect to ``Q`` (exploitation).
        """
        eps = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def greedy_action(self, state: int) -> int:
        """The exploitation action: argmax_a Q[state, a] (ties -> lowest index)."""
        return int(np.argmax(self.q[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        next_action: int | None = None,
    ) -> float:
        """Apply one TD(0) update and return the TD error ``delta``.

        ``terminated`` (not ``truncated``) decides whether the bootstrap term
        is dropped: a time-limit truncation must still bootstrap, a real
        terminal state must not.

        ``next_action`` is the action actually taken in ``next_state``. It is
        required for SARSA and ignored by Q-learning.
        """
        if terminated:
            target = reward
        elif self.is_on_policy:
            if next_action is None:
                raise ValueError("SARSA needs the action actually taken in next_state")
            target = reward + self.config.gamma * float(self.q[next_state, next_action])
        else:
            target = reward + self.config.gamma * float(self.q[next_state].max())

        delta = target - float(self.q[state, action])
        self.q[state, action] += self.config.alpha * delta
        self.n_updates += 1
        return delta

    def decay_epsilon(self) -> None:
        """Shrink the exploration probability by one episode's worth of decay."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)


@dataclass
class EpisodeStats:
    """Per-episode training telemetry."""

    returns: list[float]
    lengths: list[int]
    epsilons: list[float]
    total_steps: int

    def moving_average(self, window: int = 100) -> list[float]:
        """Mean return over a trailing window (shorter at the start)."""
        out: list[float] = []
        for i in range(len(self.returns)):
            lo = max(0, i + 1 - window)
            chunk = self.returns[lo : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out


@dataclass
class EvalResult:
    """Greedy-policy evaluation summary."""

    mean_return: float
    std_return: float
    success_rate: float
    mean_length: float
    returns: list[float]


def train_tabular(
    env: gymnasium.Env,
    agent: TabularAgent,
    episodes: int = 10_000,
    seed: int = 0,
) -> EpisodeStats:
    """Run ``episodes`` training episodes against ``env``.

    Each episode is seeded with ``seed + episode_index`` so a full run is
    reproducible from the two integers alone.
    """
    returns: list[float] = []
    lengths: list[int] = []
    epsilons: list[float] = []
    total_steps = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = int(obs)
        action = agent.select_action(state)
        episode_return = 0.0
        episode_length = 0

        while True:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_obs)

            # The next action is always drawn from the behaviour policy
            # (epsilon-greedy). SARSA *learns from* it; Q-learning ignores it and
            # bootstraps from max_a' Q[next_state, a'] instead.
            next_action = None if terminated else agent.select_action(next_state)

            agent.update(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                terminated=bool(terminated),
                next_action=next_action,
            )

            episode_return += float(reward)
            episode_length += 1
            total_steps += 1

            if terminated or truncated:
                break

            state = next_state
            action = next_action if next_action is not None else 0

        returns.append(episode_return)
        lengths.append(episode_length)
        epsilons.append(agent.epsilon)
        agent.decay_epsilon()

    return EpisodeStats(
        returns=returns,
        lengths=lengths,
        epsilons=epsilons,
        total_steps=total_steps,
    )


def evaluate_greedy(
    env: gymnasium.Env,
    agent: TabularAgent,
    episodes: int = 100,
    seed: int = 10_000,
) -> EvalResult:
    """Evaluate the greedy policy (no exploration) over ``episodes`` episodes."""
    goal = goal_state(env)
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = int(obs)
        episode_return = 0.0
        episode_length = 0
        reached_goal = False

        while True:
            action = agent.greedy_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_obs)
            episode_return += float(reward)
            episode_length += 1
            state = next_state
            if terminated and next_state == goal:
                reached_goal = True
            if terminated or truncated:
                break

        returns.append(episode_return)
        lengths.append(episode_length)
        successes += int(reached_goal)

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    return EvalResult(
        mean_return=mean_return,
        std_return=std_return,
        success_rate=successes / episodes if episodes else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        returns=returns,
    )


def goal_state(env: gymnasium.Env) -> int:
    """Index of the goal state of a toy-text grid env.

    ``FrozenLake-v1`` exposes its tile map as ``desc``; ``CliffWalking-v1``
    puts the goal in the last cell.
    """
    unwrapped = env.unwrapped
    desc = getattr(unwrapped, "desc", None)
    if desc is not None:
        for row_index, row in enumerate(desc.tolist()):
            for col_index, cell in enumerate(row):
                letter = cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)
                if letter == "G":
                    return row_index * int(unwrapped.ncol) + col_index
    return int(env.observation_space.n) - 1


def policy_map(env: gymnasium.Env, agent: TabularAgent) -> str:
    """Render the greedy policy of ``agent`` as an ASCII grid."""
    return render_policy(env, agent.n_states, agent.greedy_action)


def render_policy(env: gymnasium.Env, n_states: int, action_for: Callable[[int], int]) -> str:
    """Render a policy as an ASCII grid of arrows over the env's tile map.

    ``action_for(state)`` returns the action to draw in that state.
    """
    unwrapped = env.unwrapped
    base, blocked = _tile_map(env)

    if hasattr(unwrapped, "desc"):
        # FrozenLake: 0=left, 1=down, 2=right, 3=up
        arrows = ("<", "v", ">", "^")
    else:
        # CliffWalking: 0=up, 1=right, 2=down, 3=left
        arrows = ("^", ">", "v", "<")

    grid: list[list[str]] = []
    for row_index, row in enumerate(base):
        out_row: list[str] = []
        for col_index, cell in enumerate(row):
            state = row_index * len(row) + col_index
            if cell in blocked or state >= n_states:
                out_row.append(cell)
                continue
            out_row.append(arrows[action_for(state)])
        grid.append(out_row)

    return "\n".join(" ".join(row) for row in grid)


def _tile_map(env: gymnasium.Env) -> tuple[list[list[str]], set[str]]:
    """Background tiles of a toy-text grid env plus the set of blocked letters."""
    unwrapped = env.unwrapped
    desc = getattr(unwrapped, "desc", None)
    if desc is not None:
        grid: list[list[str]] = []
        for row in desc.tolist():
            out_row: list[str] = []
            for cell in row:
                out_row.append(cell.decode("utf-8") if isinstance(cell, bytes) else str(cell))
            grid.append(out_row)
        return grid, {"H", "G"}

    shape = tuple(int(v) for v in unwrapped.shape)
    cliff = getattr(unwrapped, "_cliff")
    start = int(unwrapped.start_state_index)
    grid = [["." for _ in range(shape[1])] for _ in range(shape[0])]
    for row_index in range(shape[0]):
        for col_index in range(shape[1]):
            if cliff[row_index, col_index]:
                grid[row_index][col_index] = "C"
    start_row, start_col = divmod(start, shape[1])
    grid[start_row][start_col] = "S"
    goal_row, goal_col = shape[0] - 1, shape[1] - 1
    grid[goal_row][goal_col] = "G"
    return grid, {"C", "G"}


def state_space_size(n_joints: int, bins_per_joint: int) -> int:
    """Number of joint configurations after per-joint discretisation.

    This is the quantity that makes tabular RL hopeless for a 6-DOF arm:
    ``bins_per_joint ** n_joints``.
    """
    if n_joints < 1:
        raise ValueError(f"n_joints must be >= 1, got {n_joints}")
    if bins_per_joint < 1:
        raise ValueError(f"bins_per_joint must be >= 1, got {bins_per_joint}")
    return int(bins_per_joint) ** int(n_joints)
