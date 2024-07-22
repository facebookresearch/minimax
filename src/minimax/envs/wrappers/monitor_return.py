"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import jax
import chex
from typing import Tuple, Union, Optional

from .env_wrapper import EnvWrapper
from minimax.envs.environment import EnvState


class MonitorReturnWrapper(EnvWrapper):
    """
    Tracks episodic returns and, optionally, environment metrics.
    """

    def reset_extra(self):
        if self._wrap_level > 1:
            extra = self._env.reset_extra()
        else:
            extra = {}

        extra.update(
            {
                "ep_return": 0.0,
            }
        )

        return extra

    def get_monitored_metrics(self):
        return super().get_monitored_metrics() + ("return",)

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

        # Track returns
        extra["ep_return"] += reward
        info["return"] = done * extra["ep_return"]
        extra["ep_return"] *= 1 - done

        return obs, state, reward, done, info, extra
