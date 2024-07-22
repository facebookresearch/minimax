"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Union, Optional

from .env_wrapper import EnvWrapper
from minimax.envs.environment import EnvState


class MonitorEpisodicMetricsWrapper(EnvWrapper):
    """
    Tracks episodic metrics about environment instances.
    """

    def __init__(self, env):
        super().__init__(env)

        base_env = env.base_env if hasattr(env, "base_env") else env
        _env = base_env.env if hasattr(base_env, "env") else base_env

        self.metrics = ()
        if hasattr(_env, "get_episodic_metrics"):
            reset_tuple = _env.reset(jax.random.PRNGKey(0))
            dummy_state = reset_tuple[1]

            self.metrics = tuple(
                {
                    k: jnp.zeros_like(v)
                    for k, v in _env.get_episodic_metrics(dummy_state).items()
                }.keys()
            )

    @classmethod
    def is_compatible(cls, env):
        _env = env.env if hasattr(env, "env") else env
        return hasattr(_env, "get_episodic_metrics")

    def get_monitored_metrics(self):
        metrics = tuple(f"ep/{m}" for m in self.metrics)
        if self._wrap_level > 1:
            return self._env.get_monitored_metrics() + metrics
        else:
            return self.metrics

    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        reset_state: Optional[chex.ArrayTree] = None,
        extra: dict = None,
    ) -> Tuple[chex.Array, EnvState, float, bool]:
        step_kwargs = dict(reset_state=reset_state)
        if self._wrap_level > 1:
            step_kwargs.update(dict(extra=extra))

        step = self._env.step(key, state, action, **step_kwargs)

        if len(step) == 5:
            obs, state, reward, done, info = step
        else:
            obs, state, reward, done, info, extra = step

        if len(self.metrics) > 0:
            for m in self.metrics:
                info[f"ep/{m}"] = info[m]
                del info[m]

        return obs, state, reward, done, info, extra
