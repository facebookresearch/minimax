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

import util.rl.ued_scores as _ued_scores

from tests.base_req_rollout_storage import RequiresRolloutStorageTestClass


class TestUEDScores(RequiresRolloutStorageTestClass):
    def test_compute_ued_scores_returns(self):
        (obs, action, done, reward, log_pi, value, carry) = (
            self._get_dummy_step_components()
        )

        # Mark episode done every 8 steps
        batch_shape = self.batch_shape
        dones = jnp.zeros(batch_shape, dtype=jnp.uint8)
        dones = dones.at[:, jnp.arange(4, self.n_steps, 4), :].set(1)

        # Reward of 10 at the end of every episode
        rewards = jnp.zeros_like(dones, dtype=jnp.float32)
        rewards = rewards.at[:, jnp.arange(4, self.n_steps, 4), :].set(
            jnp.arange(1, 8, dtype=jnp.float32).reshape(1, 7, 1)
        )

        rollout = self.rollout_mgr.reset()
        for t in range(self.n_steps):
            rollout = self.rollout_mgr.append(
                rollout,
                obs,
                action,
                rewards.at[:, t, :].get(),
                dones.at[:, t, :].get(),
                log_pi,
                value,
                carry,
            )

        score_name = _ued_scores.UEDScore.RETURN
        last_value = jnp.zeros(self.t_batch_shape)
        batch = self.rollout_mgr.get_batch(rollout, last_value)
        ued_score, _ = _ued_scores.compute_ued_scores(
            score_name, batch, n_eval=self.n_eval
        )

        n_agents, n_steps, flat_batch_size = batch.dones.shape
        pop_batch_shape = (
            n_agents,
            n_steps,
            flat_batch_size // self.n_eval,
            self.n_eval,
        )
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape(*pop_batch_shape, *x.shape[3:]), batch
        )
        mean_env_returns_per_agent, max_env_returns_per_agent, _ = jax.vmap(
            _ued_scores._compute_ued_scores, in_axes=(None, 0)
        )(score_name, batch)

        mean_return = mean_env_returns_per_agent.mean(0)
        max_return = max_env_returns_per_agent.max(0)

        assert (mean_return != ued_score).sum() == 0

        batch_size = self.n_agents * self.n_envs
        assert (
            mean_env_returns_per_agent.sum(0).sum(0)
            == jnp.arange(1, 8).mean() * batch_size
        )
        assert max_env_returns_per_agent.sum(0).sum(0) == 7 * batch_size
