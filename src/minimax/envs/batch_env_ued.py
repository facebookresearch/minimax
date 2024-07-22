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


class BatchUEDEnv:
    """
    Wraps and batches a UEDEnvironment in
    its private methods as follows:

    For student MDP:
            Manages a batch of n_parallel x n_eval envs

    For teacher MDP:
            Manages a batch of n_parallel envs.

    The public interface vmaps the private methods over
    an additional agent population dimension.
    """

    def __init__(
        self,
        env_name,
        n_parallel,
        n_eval,
        env_kwargs,
        ued_env_kwargs,
        wrappers=["monitor_return"],
        ued_wrappers=None,
    ):
        self.env, self.env_params, self.ued_params = envs.make(
            env_name,
            env_kwargs=env_kwargs,
            ued_env_kwargs=ued_env_kwargs,
            wrappers=wrappers,
            ued_wrappers=ued_wrappers,
        )

        self.n_parallel = n_parallel
        self.n_eval = n_eval
        self.sub_batch_size = n_parallel * n_eval

        self.reset_student = jax.vmap(self._reset_student, in_axes=(0, 0, None))
        self.step_teacher = jax.vmap(self._step_teacher, in_axes=0)
        self.step_student = jax.vmap(self._step_student, in_axes=0)

        self.set_env_instance = jax.vmap(self._set_env_instance, in_axes=0)
        self.get_env_metrics = jax.vmap(self._get_env_metrics, in_axes=0)

    partial(jax.jit, static_argnums=(2,))

    def reset(self, rng, sub_batch_size=None):
        if sub_batch_size is None:
            sub_batch_size = self.sub_batch_size

        return jax.vmap(self._reset, in_axes=(0, None))(rng, sub_batch_size)

    def _reset(self, rng, sub_batch_size):
        brngs = jax.random.split(rng, sub_batch_size)
        return jax.vmap(self.env.reset)(brngs)

    partial(jax.jit, static_argnums=(2,))

    def reset_teacher(self, rng, n_parallel=None):
        if n_parallel is None:
            n_parallel = self.n_parallel

        return jax.vmap(self._reset_teacher, in_axes=(0, None))(rng, n_parallel)

    def _reset_teacher(self, rng, n_parallel):
        """
        Reset n_parallel envs
        """
        brngs = jax.random.split(rng, n_parallel)
        return jax.vmap(self.env.reset_teacher)(brngs)

    def _step_teacher(self, rng, ued_state, action, extra=None):
        """
        Step n_parallel envs
        """
        brngs = jax.random.split(rng, self.n_parallel)
        step_args = (brngs, ued_state, action)
        if extra is not None:
            step_args += (extra,)

        return jax.vmap(self.env.step_teacher)(*step_args)

    def _reset_student(self, rng, ued_state, n_students):
        """
        Reset the student MDP based on the state of the teacher MDP.
        """
        brngs = jax.random.split(rng, self.n_parallel)
        obs, state, extra = jax.vmap(self.env.reset_student)(brngs, ued_state)

        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(jnp.repeat(x, self.n_eval, 0), 0), n_students, 0
            ),
            obs,
        )

        state = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(jnp.repeat(x, self.n_eval, 0), 0), n_students, 0
            ),
            state,
        )

        extra = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(jnp.repeat(x, self.n_eval, 0), 0), n_students, 0
            ),
            extra,
        )

        return obs, state, extra

    def _step_student(self, rng, state, action, reset_state, extra=None):
        """
        Step the student MDP.
        """
        brngs = jax.random.split(rng, self.sub_batch_size)
        return jax.vmap(self.env.step)(brngs, state, action, reset_state, extra)

    def _set_env_instance(self, instance):
        """
        Reset the student MDP to a particular configuration,
        captured by state argument. Used for PLR.
        """
        return jax.vmap(self.env.set_env_instance)(instance)

    def _get_env_metrics(self, state):
        return jax.vmap(self.env.get_env_metrics)(state)
