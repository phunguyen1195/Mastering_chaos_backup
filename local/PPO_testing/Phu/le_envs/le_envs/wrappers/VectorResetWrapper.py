"""Vectorizes observation wrappers to works for `VectorEnv`."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Sequence

import numpy as np

from gymnasium import Space
from gymnasium.core import Env, ObsType
from gymnasium.experimental.vector import VectorEnv, VectorWrapper
from gymnasium.experimental.vector.utils import batch_space, concatenate, iterate
from gymnasium.experimental.wrappers import lambda_observation
from gymnasium.vector.utils import create_empty_array

class VectorResetWrapperBase (VectorWrapper):
    """Wraps the vectorized environment to allow a modular transformation of the observation. Equivalent to :class:`gym.ObservationWrapper` for vectorized environments."""

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Modifies the observation returned from the environment ``reset`` using the :meth:`observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.vector_observation(obs), info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Modifies the observation returned from the environment ``step`` using the :meth:`observation`."""
        observation, reward, termination, truncation, info = self.env.step(actions)
        return (
            observation,
            reward,
            termination,
            truncation,
            self.update_final_obs(info),
        )

class VectorResetWrapper(VectorResetWrapperBase):
    """Transforms an observation via a function provided to the wrapper.

    The function :attr:`func` will be applied to all vector observations.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s observation space, provide an :attr:`observation_space`.
    """

    def __init__(
        self,
        env: VectorEnv,
        observation_space: Space | None = None,
    ):
        """Constructor for the lambda observation wrapper.

        Args:
            env: The vector environment to wrap
            vector_func: A function that will transform the vector observation. If this transformed observation is outside the observation space of ``env.observation_space`` then provide an ``observation_space``.
            single_func: A function that will transform an individual observation.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as ``env.observation_space``.
        """
        super().__init__(env)

        if observation_space is not None:
            self.observation_space = observation_space


    def vector_observation(self, observation: ObsType, delta: float | None = None) -> ObsType:
        """Apply function to the vector observation."""
        if delta is None:
            delta = 0.001
        dim = observation.shape[1]
        base_obs = observation[0]
        low = base_obs - delta
        high = base_obs + delta
        observation = np.array([self.np_random.uniform(low=low, high=high) for i in range(observation.shape[0])])
        return observation

    def single_observation(self, observation: ObsType) -> ObsType:
        """Apply function to the single observation."""
        return observation