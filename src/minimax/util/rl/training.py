"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import core
from flax import struct
import optax
import chex

from .plr import PLRBuffer


class VmapTrainState(struct.PyTreeNode):
    n_iters: chex.Array
    n_updates: chex.Array  # per agent
    n_grad_updates: chex.Array  # per agent
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    plr_buffer: PLRBuffer = None

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            n_grad_updates=self.n_updates + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = jax.vmap(tx.init)(params)
        return cls(
            n_iters=jnp.array(jax.vmap(lambda x: 0)(params), dtype=jnp.uint32),
            n_updates=jnp.array(jax.vmap(lambda x: 0)(params), dtype=jnp.uint32),
            n_grad_updates=jnp.array(jax.vmap(lambda x: 0)(params), dtype=jnp.uint32),
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def increment(self):
        return self.replace(
            n_iters=self.n_iters + 1,
        )

    def increment_updates(self):
        return self.replace(
            n_updates=self.n_updates + 1,
        )

    @property
    def state_dict(self):
        return dict(
            n_iters=self.n_iters,
            n_updates=self.n_updates,
            n_grad_updates=self.n_grad_updates,
            params=self.params,
            opt_state=self.opt_state,
        )

    def load_state_dict(self, state):
        return self.replace(
            n_iters=state["n_iters"],
            n_updates=state["n_updates"],
            n_grad_updates=state["n_grad_updates"],
            params=state["params"],
            opt_state=state["opt_state"],
        )
