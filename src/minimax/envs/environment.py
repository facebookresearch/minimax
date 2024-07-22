"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This file extends the Environment class from
https://github.com/RobertTLange/gymnax/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax
import chex
from typing import Tuple, Union, Optional
from functools import partial
from flax import struct


@struct.dataclass
class EnvState:
    time: int


@struct.dataclass
class EnvParams:
    max_episode_steps: int


class Environment(object):
    """Jittable abstract base class for all basic environments."""

    def __init__(self):
        self.eval_solved_rate = self.get_eval_solved_rate_fn()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def align_kwargs(kwargs, other_kwargs):
        """
        Return kwargs that are consistent with other_kwargs,
        e.g. in the case of the student env, other_kwargs may be
        those for the paired teacher env, and in the case of the
        teacher env, the paired student env.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        reset_on_done: bool = True,
        reset_state: Optional[chex.ArrayTree] = None,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if hasattr(self, "params"):
            params = self.params
        else:
            params = self.default_params

        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action)

        if reset_on_done:
            if reset_state is not None:
                state_re = reset_state
                obs_re = self.get_obs(reset_state)
            else:
                if hasattr(params, "singleton_seed") and params.singleton_seed >= 0:
                    key_reset = jax.random.PRNGKey(params.singleton_seed)

                obs_re, state_re = self.reset_env(key_reset)

            # Auto-reset environment based on termination
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        else:
            obs, state = obs_st, state_st

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[chex.ArrayTree, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if hasattr(self, "params"):
            params = self.params
        else:
            params = self.default_params

        if hasattr(params, "singleton_seed") and params.singleton_seed >= 0:
            key = jax.random.PRNGKey(params.singleton_seed)
        obs, state = self.reset_env(key)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, EnvState]:
        """Environment-specific reset."""
        raise NotImplementedError

    def set_state(self, state: EnvState) -> Tuple[chex.ArrayTree, EnvState]:
        """
        Implemented for basic envs.
        """
        return self.get_obs(state), state

    def set_env_instance(
        self, encoding: chex.ArrayTree
    ) -> Tuple[chex.ArrayTree, EnvState]:
        """
        Implemented for basic envs.
        """
        raise NotImplementedError

    def get_env_instance(self, key: chex.PRNGKey, state: EnvState) -> chex.ArrayTree:
        """
        Implemented for UED envs.
        """
        raise NotImplementedError

    def get_obs(self, state: EnvState) -> chex.ArrayTree:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state: EnvState) -> bool:
        """Check whether state is terminal."""
        raise NotImplementedError

    def get_eval_solved_rate_fn(self):
        return None

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self):
        """State space of the environment."""
        raise NotImplementedError

    def max_episode_steps(self):
        """Maximum number of time steps in environment."""
        raise NotImplementedError

    def get_env_metrics(self, state: EnvState):
        """Environment-specific metrics, e.g. number of walls."""
        raise NotImplementedError
