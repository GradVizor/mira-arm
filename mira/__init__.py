from __future__ import annotations

import gymnasium.envs.registration as registration

from mira.envs.so101_reach_env import SO101ReachEnv
from mira.envs.so101_point_to_point_env import SO101PointToPointEnv

# Idempotent: re-importing the package is cached, and gymnasium overrides a
# duplicate id with a warning rather than raising.
registration.register(
    id="MIRA/SO101Reach-v0",
    entry_point="mira.envs.so101_reach_env:SO101ReachEnv",
)
registration.register(
    id="MIRA/SO101PointToPoint-v0",
    entry_point="mira.envs.so101_point_to_point_env:SO101PointToPointEnv",
)

__all__ = ["SO101ReachEnv", "SO101PointToPointEnv"]
