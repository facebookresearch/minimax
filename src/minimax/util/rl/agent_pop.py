"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


class AgentPop:
    """
    Manages multiple agents of the same architecture
    """

    def __init__(self, agent, n_agents):
        """
        Maintains a set of model parameters.
        """
        self.agent = agent
        self.n_agents = n_agents

    def _reshape_to_pop(self, x):
        return jax.tree_map(
            lambda x: jnp.reshape(
                x, newshape=(self.n_agents, x.shape[0] // self.n_agents, *x.shape[1:])
            ),
            x,
        )

    def _flatten(self, x):
        return jax.tree_map(
            lambda x: jnp.reshape(
                x, newshape=(self.n_agents * x.shape[1], -1)
            ).squeeze(),
            x,
        )

    def init_params(self, rng, obs):
        if self.agent.is_recurrent:
            obs = jax.tree_map(lambda x: x[jnp.newaxis, :], obs)  # Make time first dim

        vrngs = jax.random.split(rng, self.n_agents)
        return jax.vmap(self.agent.init_params, in_axes=(0, None))(vrngs, obs)

    @partial(jax.jit, static_argnums=(0,))
    def init_carry(self, rng, obs):
        agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[1]
        return self.agent.init_carry(
            rng=rng, batch_dims=(self.n_agents, agent_batch_dim)
        )

    @partial(jax.jit, static_argnums=(0,))
    def act(self, params, obs, carry, reset=None):
        # If recurrent, add time axis to support scanned rollouts
        if self.agent.is_recurrent:
            obs = jax.tree_map(
                lambda x: x[:, jnp.newaxis, :], obs
            )  # Add time dim after agent dim

            if reset is None:
                agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[2]
                reset = jnp.zeros((self.n_agents, 1, agent_batch_dim), dtype=jnp.bool_)
            else:
                reset = reset[:, jnp.newaxis, :]

        value, pi_params, next_carry = jax.vmap(self.agent.act)(
            params, obs, carry, reset
        )

        if self.agent.is_recurrent:  # Remove time dim
            value = value.squeeze(1)
            pi_params = jax.tree_map(lambda x: x.squeeze(1), pi_params)

        return value, pi_params, next_carry

    def get_action_dist(self, dist_params, dtype=jnp.uint32):
        return self.agent.get_action_dist(dist_params, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def get_value(self, params, obs, carry):
        return self.act(params, obs, carry)[0].squeeze(-1)

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def update(self, rng, train_state, batch, prefix_steps=0, fake=False):
        if fake:
            return train_state, jax.vmap(
                lambda *_: self.agent.get_empty_update_stats()
            )(np.arange(self.n_agents))

        rng, *vrngs = jax.random.split(rng, self.n_agents + 1)
        vrngs = jnp.array(vrngs)

        return jax.vmap(self.agent.update)(vrngs, train_state, batch)
