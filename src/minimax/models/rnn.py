"""
Copyright 2018 The JAX Authors.

This file is based on the OptimizedLSTMCell class from
https://github.com/google/jax

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from functools import partial
from typing import Any, Tuple, Mapping

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.module import compact
from flax.linen.recurrent import DenseParams


Array = Any


class CustomOptimizedLSTMCell(nn.OptimizedLSTMCell):
    @compact
    def __call__(
        self, carry: Tuple[Array, Array], inputs: Array
    ) -> Tuple[Tuple[Array, Array], Array]:
        r"""An optimized long short-term memory (LSTM) cell.

        Args:
          carry: the hidden state of the LSTM cell, initialized using
                `LSTMCell.initialize_carry`.
          inputs: an ndarray with the input for the current time step. All
                dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """
        c, h = carry
        hidden_features = h.shape[-1]

        def _concat_dense(
            inputs: Array,
            params: Mapping[str, Tuple[Array, Array]],
            use_bias: bool = True,
        ) -> Array:
            # Concatenates the individual kernels and biases, given in params, into a
            # single kernel and single bias for efficiency before applying them using
            # dot_general.
            kernels, biases = zip(*params.values())
            kernel = jnp.concatenate(kernels, axis=-1)
            if use_bias:
                bias = jnp.concatenate(biases, axis=-1)
            else:
                bias = None
            inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
            y = jnp.dot(inputs, kernel)
            if use_bias:
                y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

            # Split the result back into individual (i, f, g, o) outputs.
            split_indices = np.cumsum([kernel.shape[-1] for kernel in kernels[:-1]])
            ys = jnp.split(y, split_indices, axis=-1)
            return dict(zip(params.keys(), ys))

        # Create params with the same names/shapes as `LSTMCell` for compatibility.
        dense_params_h = {}
        dense_params_i = {}
        for component in ["i", "f", "g", "o"]:
            dense_params_i[component] = DenseParams(
                features=hidden_features,
                use_bias=True,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"i{component}",
            )(inputs)
            dense_params_h[component] = DenseParams(
                features=hidden_features,
                use_bias=True,
                param_dtype=self.param_dtype,
                kernel_init=self.recurrent_kernel_init,
                bias_init=self.bias_init,
                name=f"h{component}",
            )(h)
        dense_h = _concat_dense(h, dense_params_h, use_bias=True)
        dense_i = _concat_dense(inputs, dense_params_i, use_bias=True)

        i = self.gate_fn(dense_h["i"] + dense_i["i"])
        f = self.gate_fn(dense_h["f"] + dense_i["f"])
        g = self.activation_fn(dense_h["g"] + dense_i["g"])
        o = self.gate_fn(dense_h["o"] + dense_i["o"])

        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h
