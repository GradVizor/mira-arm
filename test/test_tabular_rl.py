"""Self-checks for the Phase 2 tabular RL learners.

Run with::

    python test/test_tabular_rl.py

Exits non-zero if any check fails. The learning checks are deliberately
tolerant: RL is noisy, so the assertions bound the *outcome* (the agent solves
the task, the greedy policy is sane) rather than an exact curve.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gymnasium as gym
import numpy as np

from mira.rl.tabular import (
    TabularAgent,
    TDConfig,
    evaluate_greedy,
    goal_state,
    policy_map,
    state_space_size,
    train_tabular,
)

FROZEN_LAKE_EPISODES = 50_000
CLIFF_WALKING_EPISODES = 50_000
FROZEN_LAKE_SUCCESS_FLOOR = 0.65


def make_agent(env, algorithm="q_learning", alpha=0.1, gamma=0.95, seed=0):
    config = TDConfig(
        alpha=alpha,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999,
        algorithm=algorithm,
    )
    return TabularAgent(env.observation_space.n, env.action_space.n, config, seed=seed)


def test_q_learning_update_matches_hand_computation():
    agent = make_agent(gym.make("FrozenLake-v1"))
    agent.q[1] = [2.0, 3.0, 0.0, 0.0]

    delta = agent.update(state=0, action=1, reward=1.0, next_state=1, terminated=False)

    expected_target = 1.0 + 0.95 * 3.0
    expected_delta = expected_target - 0.0
    assert np.isclose(delta, expected_delta), f"delta {delta} != {expected_delta}"
    assert np.isclose(agent.q[0, 1], agent.config.alpha * expected_delta), f"q {agent.q[0, 1]}"


def test_sarsa_update_uses_the_action_actually_taken():
    agent = make_agent(gym.make("FrozenLake-v1"), algorithm="sarsa")
    agent.q[1] = [2.0, 3.0, 0.0, 0.0]

    agent.update(state=0, action=1, reward=1.0, next_state=1, terminated=False, next_action=0)

    expected_target = 1.0 + 0.95 * 2.0
    assert np.isclose(agent.q[0, 1], agent.config.alpha * expected_target), f"q {agent.q[0, 1]}"


def test_terminal_state_does_not_bootstrap():
    agent = make_agent(gym.make("FrozenLake-v1"))
    agent.q[1] = [2.0, 3.0, 0.0, 0.0]

    agent.update(state=0, action=1, reward=1.0, next_state=1, terminated=True)

    assert np.isclose(agent.q[0, 1], agent.config.alpha * 1.0), f"q {agent.q[0, 1]}"


def test_sarsa_rejects_a_missing_next_action():
    agent = make_agent(gym.make("FrozenLake-v1"), algorithm="sarsa")
    try:
        agent.update(state=0, action=1, reward=1.0, next_state=1, terminated=False)
    except ValueError:
        return
    raise AssertionError("SARSA should require next_action")


def test_epsilon_greedy_is_greedy_at_zero_and_uniform_at_one():
    agent = make_agent(gym.make("FrozenLake-v1"), seed=3)
    agent.q[:] = 0.0
    agent.q[:, 2] = 1.0

    for _ in range(200):
        assert agent.select_action(0, epsilon=0.0) == 2

    counts = np.zeros(4, dtype=int)
    for _ in range(4000):
        counts[agent.select_action(0, epsilon=1.0)] += 1
    assert counts.min() > 700, f"epsilon=1 should be roughly uniform, got {counts.tolist()}"


def test_training_is_deterministic_for_a_seed():
    tables = []
    for _ in range(2):
        env = gym.make("CliffWalking-v1")
        agent = make_agent(env, seed=7)
        train_tabular(env, agent, episodes=500, seed=7)
        tables.append(agent.q.copy())
    assert np.array_equal(tables[0], tables[1]), "same seed must give the same Q-table"


def test_training_improves_the_return():
    env = gym.make("CliffWalking-v1")
    agent = make_agent(env, seed=0)
    stats = train_tabular(env, agent, episodes=5_000, seed=0)
    first = float(np.mean(stats.returns[:100]))
    last = float(np.mean(stats.returns[-100:]))
    assert last > first, f"return did not improve: {first:.2f} -> {last:.2f}"


def test_q_learning_solves_frozen_lake():
    env = gym.make("FrozenLake-v1")
    agent = make_agent(env, seed=0)
    train_tabular(env, agent, episodes=FROZEN_LAKE_EPISODES, seed=0)
    evaluation = evaluate_greedy(env, agent, episodes=500, seed=10_000)
    assert evaluation.success_rate >= FROZEN_LAKE_SUCCESS_FLOOR, (
        f"success rate {evaluation.success_rate:.3f} below floor {FROZEN_LAKE_SUCCESS_FLOOR}"
    )


def test_q_learning_finds_the_optimal_cliff_walking_path():
    env = gym.make("CliffWalking-v1")
    agent = make_agent(env, seed=0)
    train_tabular(env, agent, episodes=CLIFF_WALKING_EPISODES, seed=0)
    evaluation = evaluate_greedy(env, agent, episodes=100, seed=10_000)
    assert evaluation.success_rate == 1.0, f"success rate {evaluation.success_rate}"
    assert evaluation.mean_length == 13.0, (
        f"expected the 13-step optimal path, got {evaluation.mean_length}"
    )


def test_sarsa_is_more_conservative_than_q_learning_on_cliff_walking():
    env = gym.make("CliffWalking-v1")
    q_agent = make_agent(env, seed=0)
    train_tabular(env, q_agent, episodes=CLIFF_WALKING_EPISODES, seed=0)
    q_eval = evaluate_greedy(env, q_agent, episodes=100, seed=10_000)

    sarsa_agent = make_agent(env, algorithm="sarsa", seed=0)
    train_tabular(env, sarsa_agent, episodes=CLIFF_WALKING_EPISODES, seed=0)
    sarsa_eval = evaluate_greedy(env, sarsa_agent, episodes=100, seed=10_000)

    assert sarsa_eval.success_rate == 1.0, f"sarsa success rate {sarsa_eval.success_rate}"
    assert sarsa_eval.mean_length > q_eval.mean_length, (
        f"SARSA ({sarsa_eval.mean_length}) should take the longer, safer path "
        f"than Q-learning ({q_eval.mean_length})"
    )


def test_state_space_size_grows_exponentially():
    assert state_space_size(1, 100) == 100
    assert state_space_size(3, 100) == 1_000_000
    assert state_space_size(6, 100) == 10**12


def test_policy_map_matches_the_grid_shape():
    for env_id, rows, cols in (("FrozenLake-v1", 4, 4), ("CliffWalking-v1", 4, 12)):
        env = gym.make(env_id)
        agent = make_agent(env, seed=0)
        lines = policy_map(env, agent).splitlines()
        assert len(lines) == rows, f"{env_id}: {len(lines)} rows != {rows}"
        for line in lines:
            assert len(line.split()) == cols, f"{env_id}: row has {len(line.split())} cells != {cols}"


def test_goal_state_is_identified():
    assert goal_state(gym.make("FrozenLake-v1")) == 15
    assert goal_state(gym.make("CliffWalking-v1")) == 47


TESTS = [
    test_q_learning_update_matches_hand_computation,
    test_sarsa_update_uses_the_action_actually_taken,
    test_terminal_state_does_not_bootstrap,
    test_sarsa_rejects_a_missing_next_action,
    test_epsilon_greedy_is_greedy_at_zero_and_uniform_at_one,
    test_training_is_deterministic_for_a_seed,
    test_training_improves_the_return,
    test_q_learning_solves_frozen_lake,
    test_q_learning_finds_the_optimal_cliff_walking_path,
    test_sarsa_is_more_conservative_than_q_learning_on_cliff_walking,
    test_state_space_size_grows_exponentially,
    test_policy_map_matches_the_grid_shape,
    test_goal_state_is_identified,
]


def main() -> int:
    failures = []
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures.append((test.__name__, str(exc)))
            print(f"FAIL  {test.__name__}: {exc}")
        else:
            print(f"PASS  {test.__name__}")

    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
