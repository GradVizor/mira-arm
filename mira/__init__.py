from __future__ import annotations

import gymnasium.envs.registration as registration

from mira.envs.so101_reach_env import SO101ReachEnv

# Idempotent: re-importing the package is cached, and gymnasium overrides a
# duplicate id with a warning rather than raising.
registration.register(
    id="MIRA/SO101Reach-v0",
    entry_point="mira.envs.so101_reach_env:SO101ReachEnv",
)

__all__ = ["SO101ReachEnv"]
