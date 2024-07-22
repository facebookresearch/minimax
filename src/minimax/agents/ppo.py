"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from typing import Any, Callable, Tuple
from collections import defaultdict, OrderedDict

import numpy as np
import jax
import jax.numpy as jnp
import flax
import optax
from flax.training.train_state import TrainState
from tensorflow_probability.substrates import jax as tfp

from .agent import Agent


class PPOAgent(Agent):
    def __init__(
        self,
        model,
        n_epochs=5,
        n_minibatches=1,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        clip_eps=0.2,
        clip_value_loss=True,
        track_grad_norm=False,
        n_unroll_update=1,
        n_devices=1,
    ):

        self.model = model
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.clip_eps = clip_eps
        self.clip_value_loss = clip_value_loss
        self.track_grad_norm = track_grad_norm
        self.n_unroll_update = n_unroll_update
        self.n_devices = n_devices

        self.grad_fn = jax.value_and_grad(self._loss, has_aux=True)

    @property
    def is_recurrent(self):
        return self.model.is_recurrent

    def init_params(self, rng, obs):
        """
        Returns initialized parameters and RNN hidden state for a specific
        observation shape.
        """
        rng, subrng = jax.random.split(rng)
        if self.model.is_recurrent:
            batch_size = jax.tree_util.tree_leaves(obs)[0].shape[1]
            carry = self.model.initialize_carry(rng=subrng, batch_dims=(batch_size,))
            reset = jnp.zeros((1, batch_size), dtype=jnp.bool_)

            rng, subrng = jax.random.split(rng)
            params = self.model.init(subrng, obs, carry, reset)
        else:
            params = self.model.init(subrng, obs)

        return params

    def init_carry(self, rng, batch_dims=(1,)):
        return self.model.initialize_carry(rng=rng, batch_dims=batch_dims)

    @partial(jax.jit, static_argnums=(0,))
    def act(self, params, obs, carry=None, reset=None):
        value, logits, carry = self.model.apply(params, obs, carry, reset)

        return value, logits, carry

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, action, obs, carry=None, reset=None):
        value, dist_params, carry = self.model.apply(params, obs, carry, reset)
        dist = self.get_action_dist(dist_params, dtype=action.dtype)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return value.squeeze(), log_prob.squeeze(), entropy.squeeze(), carry

    def get_action_dist(self, dist_params, dtype=jnp.uint32):
        return tfp.distributions.Categorical(logits=dist_params, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, rng, train_state, batch):
        rngs = jax.random.split(rng, self.n_epochs)

        def _scan_epoch(carry, rng):
            brng, urng = jax.random.split(rng)
            batch, train_state = carry
            minibatches = self._get_minibatches(brng, batch)
            train_state, stats = self._update_epoch(urng, train_state, minibatches)

            return (batch, train_state), stats

        (_, train_state), stats = jax.lax.scan(
            _scan_epoch, (batch, train_state), rngs, length=len(rngs)
        )

        stats = jax.tree_util.tree_map(lambda x: x.mean(), stats)
        train_state = train_state.increment_updates()

        return train_state, stats

    @partial(jax.jit, static_argnums=(0,))
    def get_empty_update_stats(self):
        keys = [
            "total_loss",
            "actor_loss",
            "value_loss",
            "entropy",
            "mean_value",
            "mean_target",
            "mean_gae",
            "grad_norm",
        ]

        return OrderedDict({k: -jnp.inf for k in keys})

    @partial(jax.jit, static_argnums=(0,))
    def _update_epoch(self, rng, train_state: TrainState, minibatches):

        def _update_minibatch(carry, step):
            rng, minibatch = step
            train_state = carry

            (loss, aux_info), grads = self.grad_fn(
                train_state.params,
                train_state.apply_fn,
                minibatch,
                rng,
            )

            loss_info = (loss,) + aux_info
            loss_info = loss_info + (optax.global_norm(grads),)

            if self.n_devices > 1:
                loss_info = jax.tree_map(
                    lambda x: jax.lax.pmean(x, "device"), loss_info
                )
                grads = jax.tree_map(lambda x: jax.lax.pmean(x, "device"), grads)

            train_state = train_state.apply_gradients(grads=grads)

            stats_def = jax.tree_util.tree_structure(
                OrderedDict(
                    {
                        k: 0
                        for k in [
                            "total_loss",
                            "actor_loss",
                            "value_loss",
                            "entropy",
                            "mean_value",
                            "mean_target",
                            "mean_gae",
                            "grad_norm",
                        ]
                    }
                )
            )

            loss_stats = jax.tree_util.tree_unflatten(
                stats_def, jax.tree_util.tree_leaves(loss_info)
            )

            return train_state, loss_stats

        rngs = jax.random.split(rng, self.n_minibatches)
        train_state, loss_stats = jax.lax.scan(
            _update_minibatch,
            train_state,
            (rngs, minibatches),
            length=self.n_minibatches,
            unroll=self.n_unroll_update,
        )

        loss_stats = jax.tree_util.tree_map(lambda x: x.mean(axis=0), loss_stats)

        return train_state, loss_stats

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def _loss(self, params, apply_fn, batch, rng=None):
        carry = None

        if self.is_recurrent:
            """
            Elements have batch shape of n_rollout_steps x n_envs//n_minibatches
            """
            carry = jax.tree_util.tree_map(lambda x: x[0, :], batch.carry)
            (
                obs,
                action,
                rewards,
                dones,
                log_pi_old,
                value_old,
                target,
                gae,
                carry_old,
            ) = batch

            if self.is_recurrent:
                dones = dones.at[1:, :].set(dones[:-1, :])
                dones = dones.at[0, :].set(False)
                _batch = batch._replace(dones=dones)

                # Returns LxB and LxBxH tensors
                obs, action, _, done, _, _, _, _, _ = _batch
                value, log_pi, entropy, carry = apply_fn(
                    params, action, obs, carry, done
                )
            else:
                value, log_pi, entropy, carry = apply_fn(params, action, obs, carry_old)
        else:
            obs, action, rewards, dones, log_pi_old, value_old, target, gae, _ = batch
            value, log_pi, entropy, _ = apply_fn(params, action, obs, carry)

        if self.clip_value_loss:
            value_pred_clipped = value_old + (value - value_old).clip(
                -self.clip_eps, self.clip_eps
            )
            value_losses = jnp.square(value - target)
            value_losses_clipped = jnp.square(value_pred_clipped - target)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        else:
            value_loss = optax.huber_loss(value, target).mean()

        if self.model.value_ensemble_size > 1:
            gae = gae.at[..., 0].get()

        ratio = jnp.exp(log_pi - log_pi_old)
        norm_gae = (gae - gae.mean()) / (gae.std() + 1e-5)
        loss_actor1 = ratio * norm_gae
        loss_actor2 = (
            jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * norm_gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

        entropy = entropy.mean()

        total_loss = (
            loss_actor + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        )

        return total_loss, (
            loss_actor,
            value_loss,
            entropy,
            value.mean(),
            target.mean(),
            gae.mean(),
        )

    @partial(jax.jit, static_argnums=0)
    def _get_minibatches(self, rng, batch):
        n_rollout_steps, n_envs = batch.dones.shape[0:2]  # get dims based on dones
        if self.is_recurrent:
            """
            Reshape elements into a batch shape of
            n_minibatches x n_envs//n_minibatches x n_rollout_steps.
            """
            assert (
                n_envs % self.n_minibatches == 0
            ), "Number of environments must be divisible into number of minibatches."

            n_env_per_minibatch = n_envs // self.n_minibatches
            shuffled_idx = jax.random.permutation(rng, jnp.arange(n_envs))

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, shuffled_idx, axis=1), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: x.swapaxes(0, 1)
                .reshape(
                    self.n_minibatches,
                    n_env_per_minibatch,
                    n_rollout_steps,
                    *x.shape[2:],
                )
                .swapaxes(1, 2),
                shuffled_batch,
            )
        else:
            n_txns = n_envs * n_rollout_steps
            assert n_envs * n_rollout_steps % self.n_minibatches == 0

            shuffled_idx = jax.random.permutation(rng, jnp.arange(n_txns))
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(
                    x.reshape(n_txns, *x.shape[2:]), shuffled_idx, axis=0
                ),
                batch,
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(self.n_minibatches, -1, *x.shape[1:]),
                shuffled_batch,
            )

        return minibatches
