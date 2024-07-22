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

from minimax.envs.environment import EnvState


class EnvWrapper:
    """
    Abstract base class for an env wrapper.
    """

    def __init__(self, env):
        self._env = env

        self._wrap_level = 1
        while hasattr(env, "_env"):
            if isinstance(env, EnvWrapper):
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

    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        reset_state: Optional[chex.ArrayTree] = None,
        extra: dict = None,
    ) -> Tuple[chex.Array, EnvState, float, bool]:
        if self._wrap_level > 1:
            return self._env.step(key, state, action, reset_state, extra)
        else:
            _tuple = self._env.step(key, state, action, reset_state=reset_state)
            return self._append_extra_to_tuple(_tuple, extra)

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, EnvState, chex.ArrayTree]:
        _tuple = self._env.reset(key)

        return self._append_extra_to_tuple(_tuple)

    def set_state(
        self,
        state: EnvState,
    ) -> Tuple[chex.ArrayTree, EnvState, chex.ArrayTree]:
        _tuple = self._env.set_state(state)

        return self._append_extra_to_tuple(_tuple)

    def set_env_instance(
        self,
        encoding: chex.ArrayTree,
    ) -> Tuple[chex.ArrayTree, EnvState, chex.ArrayTree]:
        _tuple = self._env.set_env_instance(encoding)

        return self._append_extra_to_tuple(_tuple)

    def reset_student(
        self,
        key: chex.PRNGKey,
        state: chex.ArrayTree,
    ) -> Tuple[chex.ArrayTree, EnvState, chex.ArrayTree]:
        _tuple = self._env.reset_student(key, state)

        return self._append_extra_to_tuple(_tuple)

    def __getattr__(self, attr):
        return getattr(self._env, attr)
