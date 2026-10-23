# Tabular RL — Temporal-Difference Learning from Scratch

**RL_ROADMAP Phase 2.** This is the one phase that is *not* about the robot. It exists to make
temporal-difference (TD) learning concrete before function approximation makes the maths fuzzy.

> **Code archived.** Phase 2 is complete and the code that carried it (`mira/rl/tabular.py`,
> `scripts/train_tabular.py`, `test/test_tabular_rl.py`) has been removed from the repo: a Q-table
> cannot represent a 6-DOF continuous arm, so none of it was reachable from the training path.
> Recover it with `git show 8174a96` when you want the runnable reference — the CLI examples below
> refer to that archived code and will not run on the current tree. This document is kept because
> the theory is the part that transfers to DQN/PPO.

---

## 1. The MDP, written out explicitly

Do this for every task before writing any code. "Most RL bugs are a mis-specified MDP"
(`RL_ROADMAP.md`).

**FrozenLake-v1** (4×4, slippery — the default):

```
S F F F        states:  s = row * 4 + col,  s ∈ {0..15}
F H F H        actions: 0 = left, 1 = down, 2 = right, 3 = up
F F F H        reward:  +1 on reaching G, 0 everywhere else
H F F G        transition (slippery): intended direction w.p. 1/3,
                                           each perpendicular w.p. 1/3
              terminal: G (goal) or H (hole); truncated at 100 steps
```

**CliffWalking-v1** (4×12, deterministic):

```
row 3:  S C C C C C C C C C C G       states: s = row * 12 + col, s ∈ {0..47}
row 2:  . . . . . . . . . . . .       actions: 0 = up, 1 = right, 2 = down, 3 = left
row 1:  . . . . . . . . . . . .       reward:  −1 per step, −100 for falling in the cliff
row 0:  . . . . . . . . . . . .       falling in the cliff returns you to the start
```

We use γ = 0.95: a future reward is worth 0.95 of it now, and 0.95² ≈ 0.90 the step after.

---

## 2. From the Bellman equation to the TD update

The Bellman **optimality** equation says the best possible value of a state is the best action's
one-step-lookahead:

```
V*(s) = max_a  Σ_{s'} P(s'|s,a) [ r + γ V*(s') ]
```

and the action-value function follows from it:

```
Q*(s,a) = Σ_{s'} P(s'|s,a) [ r + γ V*(s') ]
```

Neither is computable without the transition model. But we do not need the model — we need
*one sample* of what happened. Replacing the expectation by a sample and the true value by the
current estimate gives the **TD(0) update**:

```
        ┌                                        ┐
        │  r + γ · max_a' Q(s',a')  −  Q(s,a)    │   =  δ   (the TD error)
        └                                        ┘

   Q(s,a)  ←  Q(s,a) + α · δ
```

Three things to internalise:

1. **The target is bootstrapped.** `max_a' Q(s',a')` is an *estimate* of the future, not the future.
   That is what makes the update one sample of experience instead of a full episode — and it is
   also the source of most instabilities in deep RL.
2. **α is a step size**, and the update is a weighted average of the old estimate and the new
   target: `Q ← (1−α)·Q + α·target`.
3. **The target must not bootstrap past a real terminal state.** If the episode ended because the
   agent reached the goal, the target is just `r`. If it ended because of the 100-step time limit,
   the target *must* still bootstrap — that is the difference between `terminated` and `truncated`,
   and getting it wrong silently biases every value in the table.

### The whole algorithm, in ~50 lines

This is the condensed version; the archived `mira/rl/tabular.py` (`git show 8174a96`) was the same
thing with types, docstrings and telemetry.

```python
def q_learning(env, episodes=50_000, alpha=0.1, gamma=0.95, epsilon_start=1.0,
               epsilon_end=0.1, epsilon_decay=0.999, seed=0):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(seed)
    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = int(obs)

        while True:
            # epsilon-greedy: explore with prob epsilon, else act greedily
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(q[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_obs)

            # the TD(0) update -- no bootstrap past a real terminal state
            target = reward if terminated else reward + gamma * q[next_state].max()
            q[state, action] += alpha * (target - q[state, action])

            state = next_state
            if terminated or truncated:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q
```

That is the whole of tabular Q-learning. Everything else in this phase is measurement.

---

## 3. Q-learning vs SARSA: on-policy vs off-policy

The two algorithms differ in **one term** of the target:

| | target | behaviour |
|---|---|---|
| **Q-learning** (off-policy) | `r + γ · max_a' Q(s',a')` | learns about the *greedy* policy; data may come from any policy |
| **SARSA** (on-policy) | `r + γ · Q(s',a')` where `a'` is the action actually taken | learns about the policy it is actually running |

SARSA's name comes from the transition tuple **S-A-R-S-A**: state, action, reward, next state,
next action. The "next action" is the whole difference.

### The cliff-walking evidence

Run both on `CliffWalking-v1` and compare the greedy policies (archived CLI, `git show 8174a96`):
`--algo q_learning` vs `--algo sarsa`.

**Q-learning** — 13 steps, return −13 (the *optimal* path, hugging the cliff):

```
> > > > > > > > > > > v
> > > > > > > > > > > v
> > > > > > > > > > > v
^ C C C C C C C C C C G      C = cliff,  G = goal,  S = start
```

**SARSA** — 17 steps, return −17 (the *safe* path, two rows above the cliff):

```
> > > > > > > > > > > v
^ ^ ^ ^ ^ ^ ^ ^ ^ ^ > v
^ < ^ ^ ^ ^ ^ ^ ^ > v
^ C C C C C C C C C C G
```

Both reach the goal 100% of the time. The difference is *how much risk they take*.

**Why?** SARSA's target uses `Q(s',a')` where `a'` is the action the ε-greedy policy will actually
take next. Near the cliff, "the action actually taken" has a 5% chance of being a random action
that walks off the edge — and SARSA's target *charges it for that risk*. Q-learning's target uses
`max_a'`, which assumes the next action is the best one, so it happily takes the risky route.

This is the practical meaning of on-policy vs off-policy: **SARSA learns the value of the policy it
is running (including its exploration accidents); Q-learning learns the value of the best possible
policy, using data from a sloppier one.** With ε = 0 the two become identical — the difference
exists only because we are still exploring.

---

## 4. Results

All numbers below are reproducible with the commands shown; every run is seeded.

### FrozenLake-v1 (slippery 4×4)

The environment is stochastic: every intended move has a 1/3 chance of slipping sideways, and three
of the sixteen cells are holes. That puts a hard ceiling on any *deterministic* policy. The archived
CLI solved the MDP exactly with value iteration (`--compare-optimal`, `git show 8174a96`) to measure
that ceiling:

* The **optimal deterministic policy reaches 0.733 success**. Q-learning with 50k episodes
  recovers a policy with a success rate of ≈ 0.72 on *every* seed tried, and on good seeds the
  learned policy map is *character-for-character identical* to the value-iteration optimum:

```
< ^ < ^        learned == optimal
< H < H
^ v < H
H > v G
```

  Measured over seeds 0/1/2 with the shipped defaults (α = 0.1, ε floored at 0.1, 50k episodes):
  **0.730 / 0.716 / 0.716** — within two points of the proven ceiling on every seed.

* The learned policy looks odd because it is optimal: `←` at (0,0) and `↓` at (3,2) are **"stay
  put"** moves. When the ice is slippery, the best action is often to wait and try again rather
  than to move and risk a slip. A "sensible-looking" policy that always moves is *worse*.
* **You cannot reach 1.0 on this map.** The residual ~27% failure is the environment, not the
  learner. If a tutorial claims 100% on slippery FrozenLake, it is measuring something else.
* **Run-to-run spread is real — and it is a hyper-parameter story.** With the tuned defaults the
  spread is tiny (0.716–0.730), but the *same code* with α = 0.2 lands anywhere in ≈ 0.28–0.73,
  because a small error in a Q-value can flip one action in one state, and one wrong action on the
  slippery ice means falling in a hole. Always report ≥ 3 seeds
  (`RL_ROADMAP.md`: "a single RL run means nothing").

### CliffWalking-v1 (deterministic 4×12)

| Algorithm | Greedy success | Steps | Return | Path |
|---|---|---|---|---|
| Q-learning | 1.000 | 13 | −13.00 | optimal, hugs the cliff (row 2) |
| SARSA | 1.000 | 17 | −17.00 | safe, two rows above the cliff |

Q-learning's 13-step path is identical to the value-iteration optimum.

---

## 5. Checkpoint answers

### "Explain TD learning."

TD learning estimates the value of a state (or state-action pair) by *bootstrapping*: the target
for the current estimate is built from the agent's own estimate of the future, corrected by the
error it just observed. Formally, the TD error

```
δ = r + γ · max_a' Q(s',a') − Q(s,a)
```

is the difference between "what actually happened plus what I think will happen" and "what I
thought would happen". A fraction α of that error is folded back into the estimate. Because the
target is a one-step lookahead rather than a full trajectory, TD learning needs no model of the
transition dynamics, needs only one sample per update, and updates *online* — it learns while it
acts. Its weaknesses are exactly the flip side: the bootstrapped target can be self-confirming
(the agent can be optimistic about a state it has not visited), and the estimate of a state depends
on the estimate of its neighbours, so learning propagates backwards from the goal only as fast as
experience arrives.

### "Why can't this scale to a 6-DOF arm?"

Four independent reasons, and the first one alone is fatal.

**1. The state space is not enumerable.** The reach task's observation is 18 continuous numbers
(6 joint angles, 6 joint velocities, 3 end-effector positions, 3 target coordinates). Even after
discretising each joint into 100 bins:

| Discretisation | States | Q-table (6 actions, float64) |
|---|---|---|
| 1 joint × 100 bins | 100 | 4.7 KiB |
| 3 joints × 100 bins | 1,000,000 | 45.8 MiB |
| **6 joints × 100 bins** | **1,000,000,000,000** | **43.7 TiB (~48 TB)** |

The table would not fit in RAM, let alone be learned from samples. And 100 bins is already far too
coarse: it resolves each joint to ~1°, while the reward is a function of the *continuous*
end-effector position — a 1 cm success threshold is far below the bin width, so the discretised
reward is not even the reward of the task. The archived learner computed this as `bins ** n_joints`
(`state_space_size`, `git show 8174a96`) — that one function was the whole Phase 2 checkpoint.

**2. The action space is continuous.** A tabular agent stores one number per (state, action) pair,
which requires a *finite* action set. MIRA's action is a 6-dimensional continuous vector of joint
targets. You cannot enumerate `Box(-1, 1, 6)`, so there is no table to fill — the table would have
to become a *function* `Q(s, a)`.

**3. No generalisation.** A table learns each (s, a) pair from its own experience. Two joint
configurations that differ by 0.1° are different rows, so every one of them needs its own data.
The arm's useful behaviour is smooth across nearby configurations — precisely the structure a table
throws away. Sample complexity scales as |S|·|A| instead of scaling with the *dimensionality* of
the problem.

**4. The state contains a continuous target.** Even the *goal* is continuous, so the "state" is not
a fixed-size set of rows at all — it is a continuum that a finite table can only approximate.

**What fixes it:** replace the table with a function approximator (Phase 3 onwards). A network
generalises across similar states, handles continuous actions by *producing* them rather than
indexing them, and shares statistical strength between nearby configurations. That is the entire
reason deep RL exists — and the reason this phase is worth doing on a 16-state grid world first,
where you can see every single number in the table and check it against the exact solution.

---

## 6. Lab notebook — the things that actually bit me

* **The next action must come from the behaviour policy.** My first training loop sampled the next
  action only for SARSA and fell back to action 0 for Q-learning. The agent then only ever tried
  "left". It looked like a learning problem and was a control-flow bug. Symptom to watch for:
  learning that improves for a while and then freezes.
* **`terminated` vs `truncated`.** Bootstrapping past a real terminal state inflates every value
  in the table; failing to bootstrap past a *truncation* deflates them. One line, two ways to be
  wrong.
* **Do not chase 1.0 on a stochastic environment.** 0.73 on slippery FrozenLake is optimal, and
  the script proves it by solving the MDP exactly.
* **One seed is noise.** The same configuration produced 0.28 and 0.73 on two seeds (with α = 0.2).
  The `--seeds 0 1 2` flag exists for that reason.
* **The exploration floor matters more than the step size.** Sweeping (α, ε floor) over 3 seeds on
  slippery FrozenLake gave: (0.1, 0.1) → 0.716–0.730 · (0.05, 0.1) → 0.688–0.730 ·
  (0.1, 0.05) → 0.506–0.730 · (0.2, 0.1) → 0.280–0.716. The winning combination keeps a
  **10% exploration floor** (so no state is ever frozen out of the table) and a **small α**
  (so Q-estimate noise cannot flip an action next to a hole). Those two, not the episode count,
  are what turned a coin-flip into a reliable run.

---

## 7. Where to go next

Phase 3 replaces the table with a network: DQN on CartPole, then the deliberate failure —
discretise one MIRA joint and watch the state space explode. Keep the archived learner
(`git show 8174a96`) as the reference implementation you diff your neural Q-function against: when
the DQN loss diverges, the 15-line `update()` that provably works is the fastest debugging aid you
have.
