"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import jax
import jax.numpy as jnp

import minimax.envs as envs


class BatchEnv:
    def __init__(
        self, env_name, n_parallel, n_eval, env_kwargs, wrappers=["monitor_return"]
    ):
        self.env_name = env_name
        self.env, self.env_params = envs.make(
            env_name,
            env_kwargs=env_kwargs,
            wrappers=wrappers,
        )
        self.n_parallel = n_parallel
        self.n_eval = n_eval

        self.sub_batch_size = n_parallel * n_eval

        self.step = jax.vmap(self._step, in_axes=0)
        self.get_env_metrics = jax.vmap(self._get_env_metrics, in_axes=0)
        self.set_state = jax.vmap(self._set_state, in_axes=0)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def reset(self, rng, n_parallel=None, n_eval=None):
        return jax.vmap(self._reset, in_axes=(0, None, None))(rng, n_parallel, n_eval)

    def _reset(self, rng, n_parallel=None, n_eval=None):
        # Create n_parallel envs, repeated n_eval times
        if n_parallel is None:
            n_parallel = self.n_parallel

        if n_eval is None:
            n_eval = self.n_eval

        brngs = jnp.repeat(jax.random.split(rng, n_parallel), n_eval, axis=0)

        obs, state, extra = jax.vmap(self.env.reset, in_axes=(0,))(brngs)

        return obs, state, extra

    @partial(jax.jit, static_argnums=0)
    def _step(self, rng, state, action, extra):
        brngs = jax.random.split(rng, self.sub_batch_size)
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, 0, 0))(
            brngs, state, action, None, extra
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_metrics(self, state):
        return jax.vmap(self.env.get_env_metrics, in_axes=(0,))(state)

    @partial(jax.jit, static_argnums=(0,))
    def _set_state(self, state):
        # Need to repeat the state
        state = jax.tree_map(lambda x: x.repeat(self.n_eval, axis=0), state)

        return jax.vmap(self.env.set_state)(state)
