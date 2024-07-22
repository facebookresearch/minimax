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

from tests.base_req_rollout_storage import RequiresRolloutStorageTestClass


class TestRolloutStorage(RequiresRolloutStorageTestClass):
    def test_reset(self):
        rollout = self.rollout_mgr.reset()

        batch_shape = self.batch_shape
        obs_space = self.env.observation_space()
        for k, v in rollout["obs"].items():
            assert rollout["obs"][k].shape == (*batch_shape, *obs_space.spaces[k].shape)
            assert (rollout["obs"][k] > 0).sum() == 0

        assert rollout["actions"].shape == (
            *batch_shape,
            *self.env.action_space().shape,
        )

        assert rollout["rewards"].shape == batch_shape
        assert (rollout["rewards"] > 0).sum() == 0

        assert rollout["dones"].shape == batch_shape
        assert (rollout["dones"] > 0).sum() == 0

        assert rollout["log_pis_old"].shape == batch_shape
        assert (rollout["log_pis_old"] > 0).sum() == 0

        assert rollout["values_old"].shape == batch_shape
        assert (rollout["values_old"] > 0).sum() == 0

        assert rollout["_t"].shape == (self.n_agents,)
        assert (rollout["_t"] > 0).sum() == 0

        assert rollout["carry"][0].shape == (*batch_shape, self.rnn_dim)
        assert rollout["carry"][1].shape == (*batch_shape, self.rnn_dim)
        assert jax.tree_util.tree_structure(
            rollout["carry"][0].at[0, 0, 0].get()
        ) == jax.tree_util.tree_structure(self.zero_carry_t[0])
        assert jax.tree_util.tree_structure(
            rollout["carry"][1].at[0, 0, 0].get()
        ) == jax.tree_util.tree_structure(self.zero_carry_t[1])
        assert (rollout["carry"][0] > 0).sum() == 0
        assert (rollout["carry"][1] > 0).sum() == 0

    def test_append(self):
        # Make sure appending the full rollout length looks right
        n_appends = 10

        t_batch_shape = self.t_batch_shape

        rollout = self.rollout_mgr.reset()

        step = self._get_dummy_step_components()

        for i in range(n_appends):
            rollout = self.rollout_mgr.append(rollout, *step)

        t_batch_size = np.prod(t_batch_shape)

        assert rollout["actions"].sum() == t_batch_size * n_appends
        assert rollout["rewards"].sum() == t_batch_size * n_appends
        assert rollout["dones"].sum() == t_batch_size * n_appends
        assert rollout["log_pis_old"].sum() == t_batch_size * n_appends
        assert rollout["values_old"].sum() == t_batch_size * n_appends
        assert rollout["_t"].mean() == n_appends

        t_overshoot = 2
        for t in range(self.n_steps - n_appends + t_overshoot):
            rollout = self.rollout_mgr.append(rollout, *step)

        assert rollout["_t"].mean() == t_overshoot

    def test_compute_gae(self):
        # Set up placeholder values
        (obs, action, _, _, log_pi, _, carry) = self._get_dummy_step_components()

        # Mark episode done every 8 steps
        batch_shape = self.batch_shape
        done = jnp.zeros(batch_shape, dtype=jnp.uint8)
        done = done.at[:, jnp.arange(4, self.n_steps, 4), :].set(1)

        # Reward of 10 at the end of every episode
        reward = done * 10

        # Predict 0.1 at every time step
        value = jnp.ones(batch_shape) * 0.1

        # Last value is 1
        last_value = jnp.ones(self.t_batch_shape)

        advantages, targets = jax.vmap(self.rollout_mgr.compute_gae)(
            value, reward, done, last_value
        )

        _advantages = jnp.zeros(batch_shape)

        next_value = last_value
        next_advantage = np.zeros_like(advantages.at[:, 0, :].get())
        for t in np.arange(self.n_steps)[::-1]:
            _done = done.at[:, t, :].get()
            cur_value = value.at[:, t, :].get()
            td = (
                reward.at[:, t, :].get()
                + self.discount * next_value * (1 - _done)
                - cur_value
            )
            _advantages = _advantages.at[:, t, :].set(
                td + self.discount * self.gae_lambda * (1 - _done) * next_advantage
            )
            next_advantage = _advantages.at[:, t, :].get()
            next_value = cur_value

        _targets = _advantages + value

        assert (_advantages != advantages).sum() == 0
        assert (_targets != targets).sum() == 0

    def test_get_batch(self):
        rollout = self.rollout_mgr.reset()
        step = self._get_dummy_step_components()
        for i in range(self.n_steps):
            rollout = self.rollout_mgr.append(rollout, *step)

        last_value = jnp.ones(self.t_batch_shape)

        batch = self.rollout_mgr.get_batch(rollout, last_value)

        for k, v in batch.obs.items():
            assert (batch.obs[k] != rollout["obs"][k]).sum() == 0

        assert (batch.actions != rollout["actions"]).sum() == 0
        assert (batch.dones != rollout["dones"]).sum() == 0
        assert (batch.rewards != rollout["rewards"]).sum() == 0
        assert (batch.log_pis != rollout["log_pis_old"]).sum() == 0
        assert (batch.values != rollout["values_old"]).sum() == 0

        assert ((batch.advantages + batch.values) != batch.targets).sum() == 0

        assert (batch.carry[0] != rollout["carry"][0]).sum() == 0
        assert (batch.carry[1] != rollout["carry"][1]).sum() == 0

    def test_set_final_reward(self):
        rollout = self.rollout_mgr.reset()

        final_reward = jnp.ones(self.t_batch_shape) * 3
        rollout = self.rollout_mgr.set_final_reward(rollout, final_reward)

        assert (rollout["rewards"].at[:, -1, :].get() != final_reward).sum() == 0
