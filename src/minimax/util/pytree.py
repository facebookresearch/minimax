"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import collections

import jax


def pytree_set_array_at(pytree, i, value):
	return jax.tree_util.tree_map(lambda x,y: x.at[i].set(y), pytree, value)

def pytree_set_struct_at(pytree, i, value):
	return jax.tree_util.tree_map(lambda x,y: x.at[i].set(y), pytree, value)

def pytree_at(pytree, start, end=None):
	return jax.tree_util.tree_map(lambda x: x.at[start:end].get(), pytree)

def pytree_select(pred, on_true, on_false):
	vselect = jax.vmap(jax.lax.select, in_axes=(0, 0, 0))
	return jax.tree_util.tree_map(lambda x,y: vselect(pred, x, y), on_true, on_false)

def pytree_expand_batch_dim(pytree, batch_shape, n_batch_axes=2):
	"""
	Expands a single batch dimension into a multi-dim batch shape 
	"""
	return jax.tree_util.tree_map(lambda x: x.reshape(*batch_shape, *x.shape[n_batch_axes:]), pytree)

def pytree_transform(pytree, transform):
	return jax.tree_util.tree_map(lambda x: transform(x), pytree)

def pytree_merge(dst, src, start_idx, src_len):
	return jax.tree_map(lambda x,y: x.at[start_idx:start_idx+src_len].set(y.at[:src_len].get()), dst, src)