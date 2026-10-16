from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces


class NormalizeObservation(gymnasium.ObservationWrapper):
    """Normalize observations using running mean and variance (Welford).

    Normalizes by (obs - mean) / sqrt(var + eps), clips to [-clip, clip].
    """

    def __init__(
        self,
        env: gymnasium.Env,
        eps: float = 1e-8,
        clip: float = 10.0,
        update_running_mean: bool = True,
    ) -> None:
        super().__init__(env)
        if not isinstance(self.observation_space, spaces.Box):
            raise TypeError("NormalizeObservation only supports Box observation spaces")
        self.eps = eps
        self.clip = clip
        self.update_running_mean = update_running_mean
        self.mean = np.zeros(self.observation_space.shape, dtype=np.float64)
        self.var = np.ones(self.observation_space.shape, dtype=np.float64)
        self.count = 0

    def _update(self, obs: np.ndarray) -> None:
        if not self.update_running_mean:
            return
        obs64 = obs.astype(np.float64, copy=False)
        self.count += 1
        delta = obs64 - self.mean
        self.mean += delta / self.count
        delta2 = obs64 - self.mean
        self.var += delta * delta2

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs64 = observation.astype(np.float64, copy=False)
        if self.update_running_mean:
            self._update(obs64)
        std = np.sqrt(self.var / max(1, self.count) + self.eps) if self.count > 0 else np.ones_like(self.mean)
        norm = (obs64 - (self.mean if self.count > 0 else 0.0)) / (std + self.eps)
        norm = np.clip(norm, -self.clip, self.clip)
        return norm.astype(np.float32)


class ActionScale(gymnasium.ActionWrapper):
    """Linearly rescale actions from [-1,1] to a target Box space."""

    def __init__(self, env: gymnasium.Env, target_space: spaces.Box) -> None:
        super().__init__(env)
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("ActionScale expects Box action space")
        if not isinstance(target_space, spaces.Box):
            raise TypeError("target_space must be Box")
        if self.action_space.shape != target_space.shape:
            raise ValueError("action_space and target_space shapes must match")
        self.target_low = target_space.low.astype(np.float64)
        self.target_high = target_space.high.astype(np.float64)
        self._src_low = self.action_space.low.astype(np.float64)
        self._src_high = self.action_space.high.astype(np.float64)

    def action(self, action: np.ndarray) -> np.ndarray:
        action64 = action.astype(np.float64, copy=False)
        # map [-1,1] to [target_low, target_high]
        scaled = self.target_low + (action64 - self._src_low) * (self.target_high - self.target_low) / (
            self._src_high - self._src_low
        )
        return scaled.astype(np.float64)


__all__ = ["NormalizeObservation", "ActionScale"]
