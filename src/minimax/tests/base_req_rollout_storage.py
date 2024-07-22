"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from util.rl import RolloutStorage
import envs
import models
import agents


class RequiresRolloutStorageTestClass:
    def setup_class(self):
        # Create maze object
        env_name = "Maze"
        self.env, self.env_params = envs.make(env_name)

        self.n_steps = 32
        self.n_agents = 2
        self.n_envs = 3
        self.n_eval = 5
        self.rnn_dim = 4
        self.discount = 0.995
        self.gae_lambda = 0.95

        self.batch_shape = (
            self.n_agents,
            self.n_steps,
            self.n_envs * self.n_eval,
        )

        self.t_batch_shape = (
            self.n_agents,
            self.n_envs * self.n_eval,
        )

        # Create dummy agent
        self.agent_model = models.make(
            env_name=env_name,
            model_name="default_student_cnn",
            recurrent_arch="lstm",
            recurrent_hidden_dim=self.rnn_dim,
        )

        self.agent = agents.PPOAgent(
            model=self.agent_model,
        )

        dummy_rng = jax.random.PRNGKey(0)
        self.zero_carry_t = self.agent.init_carry(
            dummy_rng, batch_dims=(self.n_agents, self.n_envs * self.n_eval)
        )

        # Initialize RolloutStorage obj
        self.rollout_mgr = RolloutStorage(
            discount=self.discount,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            n_eval=self.n_eval,
            n_steps=self.n_steps,
            action_space=self.env.action_space(),
            obs_space=self.env.observation_space(),
            agent=self.agent,
            n_agents=self.n_agents,
        )

    def setup_method(self):
        pass

    def _get_dummy_step_components(self):
        t_batch_shape = self.t_batch_shape

        dummy_rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(dummy_rng)
        obs = jax.tree_util.tree_map(
            lambda x: x.repeat(np.prod(t_batch_shape)).reshape(
                *t_batch_shape, *x.shape
            ),
            obs,
        )
        action = jnp.ones(
            (*t_batch_shape, *self.env.action_space().shape),
            dtype=self.env.action_space().dtype,
        )

        done = jnp.ones(
            (*t_batch_shape, *self.env.action_space().shape), dtype=jnp.uint8
        )

        reward = jnp.ones(t_batch_shape, dtype=jnp.float32)

        log_pis_old = jnp.ones(t_batch_shape, dtype=jnp.float32)

        values_old = jnp.ones(t_batch_shape, dtype=jnp.float32)

        carry = self.zero_carry_t

        return (obs, action, reward, done, log_pis_old, values_old, carry)
