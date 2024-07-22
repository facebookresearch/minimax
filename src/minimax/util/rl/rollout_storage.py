"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp

import minimax.util.pytree as _tree_util
from .ued_scores import compute_episodic_stats


RolloutBatch = namedtuple(
    "RolloutBatch",
    (
        "obs",
        "actions",
        "rewards",
        "dones",
        "log_pis",
        "values",
        "targets",
        "advantages",
        "carry",
    ),
)


class RolloutStorage:
    def __init__(
        self,
        discount,
        gae_lambda,
        n_envs,
        n_eval,
        n_steps,
        action_space,
        obs_space,
        agent,
        n_agents=1,
    ):
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_evals = n_eval
        self.flat_batch_size = n_envs * self.n_evals
        self.action_space = action_space
        self.value_ensemble_size = agent.model.value_ensemble_size

        dummy_rng = jax.random.PRNGKey(0)
        self.empty_obs = jax.jax.tree_util.tree_map(
            lambda x: jnp.empty(
                (n_agents, n_steps, self.flat_batch_size) + x.shape, dtype=x.dtype
            ),
            obs_space.sample(dummy_rng),
        )

        self.empty_action = jax.jax.tree_util.tree_map(
            lambda x: jnp.empty(
                (n_agents, n_steps, self.flat_batch_size) + x.shape, dtype=x.dtype
            ),
            action_space.sample(dummy_rng),
        )

        if agent.is_recurrent:
            self.empty_carry = agent.init_carry(
                dummy_rng, batch_dims=(n_agents, self.n_steps, self.flat_batch_size)
            )
        else:
            self.empty_carry = None

        if agent.is_recurrent:
            self.append = jax.vmap(self._append_with_carry, in_axes=0)
        else:
            self.append = jax.vmap(self._append_without_carry, in_axes=0)
        self.get_batch = jax.vmap(self._get_batch)
        self.get_return_stats = jax.vmap(self._get_return_stats, in_axes=(0, None))

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        """
        Maintains a pytree of rollout transitions and metadata
        """
        if self.empty_carry is None:
            carry_buffer = None
        else:
            carry_buffer = self.empty_carry

        value_batch_size = (self.flat_batch_size,)
        if self.value_ensemble_size > 1:
            value_batch_size += (self.value_ensemble_size,)

        return {
            "obs": self.empty_obs,
            "actions": self.empty_action,
            "rewards": jnp.empty(
                (self.n_agents, self.n_steps, self.flat_batch_size), dtype=jnp.float32
            ),
            "dones": jnp.empty(
                (self.n_agents, self.n_steps, self.flat_batch_size), dtype=jnp.uint8
            ),
            "log_pis_old": jnp.empty(
                (self.n_agents, self.n_steps, self.flat_batch_size), dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_agents, self.n_steps, *value_batch_size), dtype=jnp.float32
            ),
            "carry": carry_buffer,
            "_t": jnp.zeros((self.n_agents,), dtype=jnp.uint32),  # for vmap
        }

    @partial(jax.jit, static_argnums=0)
    def _append(self, buffer, obs, action, reward, done, log_pi, value, carry):
        if carry is not None:
            carry_buffer = _tree_util.pytree_set_array_at(
                buffer["carry"], buffer["_t"], carry
            )
        else:
            carry_buffer = None

        return {
            "obs": _tree_util.pytree_set_struct_at(buffer["obs"], buffer["_t"], obs),
            "actions": _tree_util.pytree_set_struct_at(
                buffer["actions"], buffer["_t"], action
            ),
            "rewards": buffer["rewards"].at[buffer["_t"]].set(reward.squeeze()),
            "dones": buffer["dones"].at[buffer["_t"]].set(done.squeeze()),
            "log_pis_old": buffer["log_pis_old"].at[buffer["_t"]].set(log_pi),
            "values_old": buffer["values_old"].at[buffer["_t"]].set(value.squeeze()),
            "carry": carry_buffer,
            "_t": (buffer["_t"] + 1) % self.n_steps,
        }

    @partial(jax.jit, static_argnums=0)
    def _append_with_carry(
        self, buffer, obs, action, reward, done, log_pi, value, carry
    ):
        return self._append(buffer, obs, action, reward, done, log_pi, value, carry)

    @partial(jax.jit, static_argnums=0)
    def _append_without_carry(self, buffer, obs, action, reward, done, log_pi, value):
        return self._append(buffer, obs, action, reward, done, log_pi, value, None)

    @partial(jax.jit, static_argnums=(0,))
    def _get_batch(self, buffer, last_value):
        _dones = buffer["dones"]
        rewards = buffer["rewards"]

        gae, target = self.compute_gae(
            value=buffer["values_old"],
            reward=rewards,
            done=_dones,
            last_value=last_value,
        )

        # T x N x E x M --> N x T x EM if recurrent or N x TEM if not
        if self.empty_carry is not None:
            carry = buffer["carry"]
        else:
            carry = None

        batch_kwargs = dict(
            obs=buffer["obs"],
            actions=buffer["actions"],
            rewards=rewards,
            dones=_dones,
            log_pis=buffer["log_pis_old"],
            values=buffer["values_old"],
            targets=target,
            advantages=gae,
            carry=carry,
        )

        return RolloutBatch(**batch_kwargs)

    def compute_gae(self, value, reward, done, last_value):
        def _compute_gae(carry, step):
            (discount, gae_lambda, gae, value_next) = carry
            value, reward, done = step

            value_diff = discount * value_next * (1 - done) - value
            delta = reward + value_diff

            gae = delta + discount * gae_lambda * (1 - done) * gae

            return (discount, gae_lambda, gae, value), gae

        value, reward, done = jnp.flip(value, 0), jnp.flip(reward, 0), jnp.flip(done, 0)

        # Handle ensemble values, which have an extra ensemble dim at index -1
        if value.shape != done.shape:
            reward = jnp.expand_dims(reward, -1)
            done = jnp.expand_dims(done, -1)

        gae = jnp.zeros(value.shape[1:])
        _, advantages = jax.lax.scan(
            _compute_gae,
            (self.discount, self.gae_lambda, gae, last_value),
            (value, reward, done),
            length=len(reward),
        )
        advantages = jnp.flip(advantages, 0)
        targets = advantages + jnp.flip(value, 0)

        return advantages, targets

    def _get_return_stats(self, rollout, control_idxs=None):
        if control_idxs is not None:
            positive_signs = control_idxs == 0
            reward_signs = -1 * (
                positive_signs.astype(jnp.float32)
                - (~positive_signs).astype(jnp.float32)
            )
            rewards = rollout["rewards"] * reward_signs
        else:
            rewards = rollout["rewards"]

        pop_batch_shape = (self.n_steps, self.n_envs, self.n_evals)
        rewards = jnp.flip(rewards.reshape(*pop_batch_shape), 0)
        dones = jnp.flip(rollout["dones"].reshape(*pop_batch_shape), 0)

        return compute_episodic_stats(rewards, dones)

    def set_final_reward(self, rollout, reward):
        rollout["rewards"] = rollout["rewards"].at[:, -1, :].set(reward)

        return rollout
