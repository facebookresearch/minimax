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

from minimax.envs.environment import Environment, EnvState


class UEDEnvWrapper:
    """
    Abstract base class for an env wrapper.
    """

    def __init__(self, env):
        self._env = env

        self._wrap_level = 1
        while hasattr(env, "_env"):
            if not isinstance(env, Environment):
                self._wrap_level += 1

            env = env._env

    @classmethod
    def is_compatible(cls, env):
        return True

    @property
    def base_env(self):
        env = self
        for i in range(self._wrap_level):
            env = env._env

        return env

    def reset_extra(self):
        return {}

    def get_monitored_metrics(self):
        if self._wrap_level > 1:
            return self._env.get_monitored_metrics()
        return ()

    def _append_extra_to_tuple(self, _tuple, extra=None):
        if extra is None:
            extra = self.reset_extra()

        if self._wrap_level > 1:
            _tuple[-1].update(extra)
        else:
            _tuple = _tuple + (extra,)

        return _tuple

    def reset_teacher(self, rng: chex.PRNGKey) -> Tuple[chex.ArrayTree, EnvState]:
        _tuple = self._env.reset_teacher(rng)

        return self._append_extra_to_tuple(_tuple)

    def step_teacher(
        self,
        rng: chex.PRNGKey,
        ued_state: EnvState,
        action: Union[int, float],
        extra: dict = None,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        if self._wrap_level > 1:
            return self._env.step_teacher(rng, ued_state, action, extra)
        else:
            _tuple = self._env.step_teacher(rng, ued_state, action)
            return self._append_extra_to_tuple(_tuple)

    def __getattr__(self, attr):
        return getattr(self._env, attr)
