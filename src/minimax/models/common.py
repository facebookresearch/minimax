"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from typing import Tuple, Callable

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex

import minimax.envs as envs
from .rnn import CustomOptimizedLSTMCell


def calc_gain(kind):
    if kind == "linear":
        return 1.0
    elif kind == "conv":
        return 1.0
    elif kind == "sigmoid":
        return 1.0
    elif kind == "tanh":
        return np.sqrt(2)
    elif kind == "relu":
        return np.sqrt(2)
    elif kind == "leaky_relu":
        return np.sqrt(2 / (1 + 0.01))
    elif kind == "selu":
        return 0.75
    elif kind == "gelu":
        return 0.75
    elif kind == "crelu":
        return np.sqrt(2)


def crelu(x):
    return jnp.concatenate((nn.relu(x), nn.relu(-x)), axis=-1)


def get_activation(name):
    if name == "crelu":
        return crelu
    else:
        return getattr(nn, name)


def default_bias_init(scale=1.0):
    return nn.initializers.zeros


def init_orth(scale=1.0):
    return nn.initializers.orthogonal(scale)


def make_fc_layers(
    name=None,
    n_layers=1,
    hidden_dim=32,
    activation=None,
    kernel_init=None,
    use_layernorm=False,
):
    if kernel_init is None:
        kernel_init = init_orth(scale=calc_gain("linear"))

    layers = []
    for i in range(n_layers):
        layer_name = None
        if name:
            layer_name = f"{name}_{i+1}"

        layers.append(
            nn.Dense(
                hidden_dim,
                kernel_init=kernel_init,
                name=layer_name,
            )
        )

        if use_layernorm:
            layers.append(nn.LayerNorm())

        if activation is not None:
            layers.append(activation)

    return nn.Sequential(layers)


def make_rnn(
    arch="lstm", kernel_init=init_orth(), recurrent_kernel_init=init_orth(), name=None
):
    if arch == "lstm":
        rnn = CustomOptimizedLSTMCell(
            kernel_init=init_orth(), recurrent_kernel_init=init_orth(), name=name
        )
    elif arch == "gru":
        rnn = nn.GRUCell(
            kernel_init=init_orth(), recurrent_kernel_init=init_orth(), name=name
        )
    else:
        rnn = None

    return rnn


class RecurrentModuleBase(nn.Module):
    def initialize_carry(
        self, rng: chex.PRNGKey, batch_dims: Tuple[int] = ()
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """Initialize hidden state of LSTM."""
        if self.recurrent_arch == "lstm":
            return nn.OptimizedLSTMCell.initialize_carry(
                rng, batch_dims, self.recurrent_hidden_dim
            )
        elif self.recurrent_arch == "gru":
            return nn.GRUCell.initialize_carry(
                rng, batch_dims, self.recurrent_hidden_dim
            )
        else:
            raise ValueError("Model is not recurrent.")

    @property
    def is_recurrent(self):
        return self.recurrent_arch is not None


class ScannedRNN(nn.Module):
    """
    Scanned RNN.
    Inputs:
            carry: time-major input hidden states, LxBxH and optional
            resets: Reset flags of shape LxB, where 1 indicates reset (equivalent to done==True).
    """

    recurrent_arch: str = "lstm"
    recurrent_hidden_dim: int = 256
    kernel_init: Callable = init_orth()
    recurrent_kernel_init: Callable = init_orth()

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, step):
        x, reset = step
        rnn_state = carry

        rnn_state = jax.tree_map(
            lambda x, y: jax.vmap(jax.lax.select)(reset, x, y),
            ScannedRNN.initialize_carry(
                jax.random.PRNGKey(0),
                (x.shape[0],),
                self.recurrent_hidden_dim,
                self.recurrent_arch,
            ),
            rnn_state,
        )

        rnn_kwargs = dict(
            features=self.recurrent_hidden_dim,
            kernel_init=self.kernel_init,
            recurrent_kernel_init=self.recurrent_kernel_init,
        )
        if self.recurrent_arch == "lstm":
            rnn_cell = nn.OptimizedLSTMCell(**rnn_kwargs)  # defaults to orth init
        elif self.recurrent_arch == "gru":
            rnn_cell = nn.GRUCell(**rnn_kwargs)
        else:
            raise ValueError(f"Unsupported recurrent_arch={self.recurrent_arch}")

        new_rnn_state, y = rnn_cell(rnn_state, x)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(rng, batch_dims, recurrent_hidden_dim, recurrent_arch):
        init_args = (rng, (*batch_dims, recurrent_hidden_dim))
        if recurrent_arch == "lstm":
            return nn.OptimizedLSTMCell(
                recurrent_hidden_dim, parent=None
            ).initialize_carry(
                *init_args
            )  # defaults to orth init
        elif recurrent_arch == "gru":
            return nn.GRUCell(recurrent_hidden_dim, parent=None).initialize_carry(
                *init_args
            )
        else:
            raise ValueError(f"Unsupported recurrent_arch={recurrent_arch}")


class ValueHead(nn.Module):
    n_hidden_layers: int = 1
    hidden_dim: int = 256
    activation: Callable = nn.tanh
    kernel_init: Callable = init_orth(calc_gain("tanh"))
    last_layer_kernel_init: Callable = init_orth(calc_gain("linear"))
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                make_fc_layers(
                    n_layers=self.n_hidden_layers,
                    hidden_dim=self.hidden_dim,
                    activation=self.activation,
                    kernel_init=self.kernel_init,
                    use_layernorm=self.use_layernorm,
                ),
                nn.Dense(
                    1, kernel_init=self.last_layer_kernel_init, name="fc_value_final"
                ),
            ]
        )(x)


class EnsembleValueHead(nn.Module):
    n: int = 2

    n_hidden_layers: int = 1
    hidden_dim: int = 256
    activation: Callable = nn.tanh
    kernel_init: Callable = init_orth(calc_gain("tanh"))
    last_layer_kernel_init: Callable = init_orth(calc_gain("linear"))

    @nn.compact
    def __call__(self, x):
        """
        Assumes x is batch
        """
        VmapValue = nn.vmap(
            ValueHead,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=1,
            axis_size=self.n,
        )
        vs = VmapValue(
            n_hidden_layers=self.n_hidden_layers,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            kernel_init=self.kernel_init,
            last_layer_kernel_init=self.last_layer_kernel_init,
        )(x)

        return vs


def clean_init_kwargs_prefix(prefix):
    def class_decorator(cls):
        old_init = cls.__init__

        def new_init(self, *args, **kwargs):
            kwargs = {k.removeprefix(prefix): v for k, v in kwargs.items()}
            old_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return class_decorator
