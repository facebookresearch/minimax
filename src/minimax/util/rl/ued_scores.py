"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from enum import Enum
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import chex


class UEDScore(Enum):
	RELATIVE_REGRET = 1
	MEAN_RELATIVE_REGRET = 2
	POPULATION_REGRET = 3
	RETURN = 4
	NEG_RETURN = 5
	L1_VALUE_LOSS = 6
	POSITIVE_VALUE_LOSS = 7
	MAX_MC = 8
	VALUE_DISAGREEMENT = 9


@partial(jax.jit, static_argnums=(2,3))
def compute_episodic_stats(
	metrics, 
	dones, 
	time_average=False, 
	partial_metrics=0,
	partial_steps=0,
	return_partial=False):
	env_batch_shape = dones.shape[1:]
	n_episodes = jnp.zeros(env_batch_shape, dtype=jnp.uint32)
	sum_ep_metrics = jnp.zeros(env_batch_shape, dtype=jnp.float32)
	partial_metrics = jnp.zeros(env_batch_shape, dtype=jnp.float32)
	max_metrics = jnp.zeros(env_batch_shape, dtype=jnp.float32)
	steps = jnp.zeros(env_batch_shape, dtype=jnp.float32)
	partial_steps = jnp.zeros(env_batch_shape, dtype=jnp.float32)

	def _compute_metrics(carry, step):
		(n_episodes, 
		 sum_ep_metrics, 
		 max_metrics,
		 partial_metrics, 
		 partial_steps) = carry

		_metrics, _dones = step

		partial_metrics += _metrics
		partial_steps += 1

		if time_average:
			ep_metric = partial_metrics/partial_steps
		else:
			ep_metric = partial_metrics

		sum_ep_metrics += _dones*ep_metric
		max_metrics = _dones*jnp.maximum(max_metrics, ep_metric) + (1-_dones)*max_metrics

		n_episodes += _dones

		partial_metrics = (1-_dones)*partial_metrics
		partial_steps = (1-_dones)*partial_steps

		return (
			n_episodes, 
			sum_ep_metrics, 
			max_metrics,
			partial_metrics,
			partial_steps
		), None

	# @todo: Track and return partial means + partial n_steps
	(n_episodes, sum_ep_metrics, max_metrics, partial_metrics, partial_steps), _  = jax.lax.scan(
		_compute_metrics,
		(n_episodes, sum_ep_metrics, max_metrics, partial_metrics, partial_steps),
		(metrics, dones),
		length=len(metrics)
	)

	# Take mean over eval dimension
	total_metrics_per_env = sum_ep_metrics.sum(-1)
	n_episodes_per_env = n_episodes.sum(-1)
	n_episodes_per_env = jnp.maximum(n_episodes_per_env, 1)

	# Take max over eval dimension
	max_metrics_per_env = max_metrics.max(-1)

	return total_metrics_per_env/n_episodes_per_env, max_metrics_per_env

@partial(jax.jit, static_argnums=(0,))
def _compute_ued_scores(score_name: UEDScore, batch: namedtuple, info=None):
	"""
	Compute UED score from a rollout batch.
	Individual score functions return a tuple of mean_scores and max_scores,
	where each is of dimension n_agents x n_envs.
	"""
	if score_name in [UEDScore.RELATIVE_REGRET, UEDScore.MEAN_RELATIVE_REGRET, UEDScore.POPULATION_REGRET]:
		mean_scores, max_scores, score_info = compute_return(batch)

	elif score_name == UEDScore.RETURN:
		mean_scores, max_scores, score_info = compute_return(batch)

	elif score_name == UEDScore.NEG_RETURN:
		batch = batch._replace(rewards=-batch.rewards)
		mean_scores, max_scores, score_info = compute_return(batch)

	elif score_name == UEDScore.MAX_MC:
		mean_scores, max_scores, score_info = compute_max_mc(batch, info)

	elif score_name == UEDScore.L1_VALUE_LOSS:
		mean_scores, max_scores, score_info = compute_l1_value_loss(batch)

	elif score_name == UEDScore.POSITIVE_VALUE_LOSS:
		mean_scores, max_scores, score_info = compute_positive_value_loss(batch)

	elif score_name == UEDScore.VALUE_DISAGREEMENT:
		mean_scores, max_scores, score_info = compute_value_disagreement(batch)

	return mean_scores, max_scores, score_info

@partial(jax.jit, static_argnums=(0,2,4,5))
def compute_ued_scores(score_name: UEDScore, batch: namedtuple, n_eval: int, info: dict = None, ignore_val=None, per_agent=False):
	n_agents, n_steps, flat_batch_size = batch.dones.shape
	pop_batch_shape = (n_agents, n_steps, flat_batch_size//n_eval, n_eval)
	batch = jax.tree_util.tree_map(lambda x: x.reshape(*pop_batch_shape, *x.shape[3:]), batch)

	mean_env_returns_per_agent, max_env_returns_per_agent, score_info = \
		jax.vmap(_compute_ued_scores, in_axes=(None, 0, 0))(
			score_name, batch, info
		)

	if score_name in [UEDScore.RELATIVE_REGRET, UEDScore.MEAN_RELATIVE_REGRET]:
		assert len(mean_env_returns_per_agent) == 2, \
			"Standard PAIRED requires exactly 2 agents."		

	if score_name == UEDScore.RELATIVE_REGRET:
		scores = jnp.clip(max_env_returns_per_agent[1] \
				- mean_env_returns_per_agent[0], 0)

	elif score_name == UEDScore.MEAN_RELATIVE_REGRET:
		scores = jnp.clip(mean_env_returns_per_agent[1] \
				- mean_env_returns_per_agent[0], 0)

	elif score_name == UEDScore.POPULATION_REGRET:
		max_env_returns = max_env_returns_per_agent.max(0)
		mean_env_returns = mean_env_returns_per_agent.mean(0)
		scores = max_env_returns - mean_env_returns
	else:
		if per_agent:
			scores = mean_env_returns_per_agent
			max_scores = max_env_returns_per_agent
		else:
			scores = mean_env_returns_per_agent.mean(0)
			max_scores = max_env_returns_per_agent.max(0)

	if ignore_val is not None:
		if per_agent:
			incomplete_idxs = batch.dones.sum(axis=(1,-1)) == 0
		else:
			incomplete_idxs = batch.dones.sum(axis=(0, 1,-1)) == 0

		scores = jnp.where(incomplete_idxs, ignore_val, scores)

	return scores, score_info

# ======== UED score computations ========
def compute_return(batch):
	mean_scores, max_scores = compute_episodic_stats(
		batch.rewards, batch.dones, time_average=False)

	return mean_scores, max_scores, None


def compute_l1_value_loss(batch):
	mean_scores, max_scores = compute_episodic_stats(
		jnp.abs(batch.advantages), batch.dones, time_average=True)

	return mean_scores, max_scores, None


def compute_positive_value_loss(batch):
	mean_scores, max_scores = compute_episodic_stats(
		jnp.clip(batch.advantages, 0), batch.dones, time_average=True)

	return mean_scores, max_scores, None


def compute_max_mc(batch, info):
	_, max_env_returns_per_agent = \
		compute_episodic_stats(batch.rewards, batch.dones, time_average=False)

	max_returns = jnp.maximum(max_env_returns_per_agent, info['max_returns'])
	mean_scores, max_scores = compute_episodic_stats(
		max_returns[jnp.newaxis,:,jnp.newaxis] - batch.values, # Can be negative
		batch.dones, 
		time_average=True
	)

	score_info = {'max_returns': max_env_returns_per_agent}

	return mean_scores, max_scores, score_info


def compute_value_disagreement(batch):
	mean_scores, max_scores = compute_episodic_stats(
		batch.values.std(-1), batch.dones, time_average=True
	)

	return mean_scores, max_scores, None 

