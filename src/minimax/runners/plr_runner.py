"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from enum import Enum
from typing import Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import optax
import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import minimax.envs as envs
from minimax.runners.dr_runner import DRRunner
from minimax.util import pytree as _tree_util
from minimax.util.rl import (
	AgentPop,
	VmapTrainState,
	RolloutStorage,
	RollingStats,
	UEDScore,
	compute_ued_scores,
	PopPLRManager
)


class MutationCriterion(Enum):
	BATCH = 0
	EASY = 1
	HARD = 2


class PLRRunner(DRRunner):
	def __init__(
		self, 
		*,
		replay_prob=0.5,
		buffer_size=100,
		staleness_coef=0.3,
		use_score_ranks=True,
		temp=1.0,
		min_fill_ratio=0.5,
		use_robust_plr=False,
		use_parallel_eval=False,
		ued_score='l1_value_loss',
		force_unique=False, # Slower if True
		mutation_fn=None,
		n_mutations=0,
		mutation_criterion='batch',
		mutation_subsample_size=1,
		**kwargs):
		use_mutations = mutation_fn is not None
		if use_parallel_eval:
			replay_prob = 1.0 # Replay every rollout cycle
			mutation_criterion = 'batch' # Force batch mutations (no UED scores)
			self._n_parallel_batches = 3 if use_mutations else 2
			kwargs['n_parallel'] *= self._n_parallel_batches

		super().__init__(**kwargs)

		self.replay_prob = replay_prob
		self.buffer_size = buffer_size
		self.staleness_coef = staleness_coef
		self.temp = temp
		self.use_score_ranks = use_score_ranks
		self.min_fill_ratio = min_fill_ratio
		self.use_robust_plr = use_robust_plr
		self.use_parallel_eval = use_parallel_eval
		self.ued_score = UEDScore[ued_score.upper()]

		self.use_mutations = use_mutations
		if self.use_mutations:
			self.mutation_fn = envs.get_mutator(self.benv.env_name, mutation_fn)
		else:
			self.mutation_fn = None
		self.n_mutations = n_mutations
		self.mutation_criterion = MutationCriterion[mutation_criterion.upper()]
		self.mutation_subsample_size = mutation_subsample_size

		self.force_unique = force_unique
		if force_unique:
			self.comparator_fn = envs.get_comparator(self.benv.env_name)
		else:
			self.comparator_fn = None

		if mutation_fn is not None and mutation_criterion != 'batch':
			assert self.n_parallel % self.mutation_subsample_size == 0, \
				'Number of parallel envs must be divisible by mutation subsample size.'

	def reset(self, rng):
		runner_state = list(super().reset(rng))
		rng = runner_state[0]
		runner_state[0], subrng = jax.random.split(rng)
		example_state = self.benv.env.reset(rng)[1]

		self.plr_mgr = PopPLRManager(
			n_agents=self.n_students,
			example_level=example_state,
			ued_score=self.ued_score,
			replay_prob=self.replay_prob,
			buffer_size=self.buffer_size,
			staleness_coef=self.staleness_coef,
			temp=self.temp,
			use_score_ranks=self.use_score_ranks,
			min_fill_ratio=self.min_fill_ratio,
			use_robust_plr=self.use_robust_plr,
			use_parallel_eval=self.use_parallel_eval,
			comparator_fn=self.comparator_fn,
			n_devices=self.n_devices
		)
		plr_buffer = self.plr_mgr.reset(self.n_students)

		train_state = runner_state[1]
		train_state = train_state.replace(plr_buffer=plr_buffer)
		if self.n_devices == 1:
			runner_state[1] = train_state
		else:
			plr_buffer = jax.tree_map(lambda x: x.repeat(self.n_devices, 1), plr_buffer) # replicate plr buffer
			runner_state += (plr_buffer,) # Return PLR buffer directly to make shmap easier

		self.dummy_eval_output = self._create_dummy_eval_output(train_state)

		return tuple(runner_state)

	def _create_dummy_eval_output(self, train_state):
		rng, *vrngs = jax.random.split(jax.random.PRNGKey(0), self.n_students+1)
		obs, state, extra = self.benv.reset(jnp.array(vrngs))

		ep_stats = self.rolling_stats.reset_stats(
			batch_shape=(self.n_students, self.n_parallel*self.n_eval))

		ued_scores = jnp.zeros((self.n_students, self.n_parallel))

		if self.student_pop.agent.is_recurrent:
			carry = self.zero_carry
		else:
			carry = None
		rollout = self.student_rollout.reset()

		train_batch = self.student_rollout.get_batch(
			rollout, 
			self.student_pop.get_value(
				jax.lax.stop_gradient(train_state.params), 
				obs, 
				carry,
			)
		)

		return (
			rng,
			train_state, 
			state, 
			state,
			obs, 
			carry, 
			extra, 
			ep_stats,
			state,
			train_batch,
			ued_scores
		)

	@partial(jax.jit, static_argnums=(0,8))
	def _eval_and_update_plr(
			self,
			rng,
			levels,
			level_idxs, 
			train_state,    
			update_plr,
			parent_idxs=None,
			dupe_mask=None,
			fake=False):
		# Collect rollout and optionally update plr buffer
		# Returns train_batch and ued_scores
		# Perform rollout: @todo: pmap this
		if fake:
			dummy_eval_output = list(self.dummy_eval_output)
			dummy_eval_output[1] = train_state
			return tuple(dummy_eval_output)

		rollout_batch_shape = (self.n_students, self.n_parallel*self.n_eval)
		obs, state, extra = self.benv.set_state(levels)
		ep_stats = self.rolling_stats.reset_stats(
			batch_shape=rollout_batch_shape)

		rollout_start_state = state

		done = jnp.zeros(rollout_batch_shape, dtype=jnp.bool_)
		if self.student_pop.agent.is_recurrent:
			carry = self.zero_carry
		else:
			carry = None

		rng, subrng = jax.random.split(rng)
		start_state = state
		reset_state = state
		rollout, state, start_state, obs, carry, extra, ep_stats, train_state = \
			self._rollout_students(
				subrng, 
				train_state, 
				state, 
				start_state,
				obs, 
				carry, 
				done,
				reset_state,
				extra, 
				ep_stats
			)

		train_batch = self.student_rollout.get_batch(
			rollout, 
			self.student_pop.get_value(
				jax.lax.stop_gradient(train_state.params), 
				obs, 
				carry
			)
		)

		# Update PLR buffer
		if self.ued_score == UEDScore.MAX_MC:
			max_returns = jax.vmap(lambda x,y: x.at[y].get())(train_state.plr_buffer.max_returns, level_idxs)
			max_returns = jnp.where(
				jnp.greater_equal(level_idxs, 0),
				max_returns,
				jnp.full_like(max_returns, -jnp.inf)
			)
			ued_info = {'max_returns': max_returns}
		else:
			ued_info = None
		ued_scores, ued_score_info = compute_ued_scores(
			self.ued_score, train_batch, self.n_eval, info=ued_info, ignore_val=-jnp.inf, per_agent=True)
		next_plr_buffer = self.plr_mgr.update(
			train_state.plr_buffer, 
			levels=levels, 
			level_idxs=level_idxs, 
			ued_scores=ued_scores,
			dupe_mask=dupe_mask, 
			info=ued_score_info, 
			ignore_val=-jnp.inf,
			parent_idxs=parent_idxs)

		next_plr_buffer = jax.vmap(
			lambda update, new, prev: jax.tree_map(
				lambda x, y: jax.lax.select(update, x, y), new, prev)
		)(update_plr, next_plr_buffer, train_state.plr_buffer)

		train_state = train_state.replace(plr_buffer=next_plr_buffer)

		return (
			rng,
			train_state, 
			state, 
			start_state, 
			obs, 
			carry, 
			extra, 
			ep_stats,
			rollout_start_state,
			train_batch,
			ued_scores,
		)

	@partial(jax.jit, static_argnums=(0,))
	def _mutate_levels(self, rng, levels, level_idxs, ued_scores=None):
		if not self.use_mutations:
			return levels, level_idxs, jnp.full_like(level_idxs, -1)

		def upsample_levels(levels, level_idxs, subsample_idxs):
			subsample_idxs = subsample_idxs.repeat(self.n_parallel//self.mutation_subsample_size, -1)
			parent_idxs = level_idxs.take(subsample_idxs)
			levels = jax.vmap(
				lambda x, y: jax.tree_map(lambda _x: jnp.array(_x).take(y, 0), x)
			)(levels, parent_idxs)
			
			return levels, parent_idxs

		if self.mutation_criterion == MutationCriterion.BATCH:
			parent_idxs = level_idxs

		if self.mutation_criterion == MutationCriterion.EASY:
			_, top_level_idxs = jax.lax.approx_min_k(ued_scores, self.mutation_subsample_size)
			levels, parent_idxs = upsample_levels(levels, level_idxs, top_level_idxs)

		elif self.mutation_criterion == MutationCriterion.HARD:
			_, top_level_idxs = jax.lax.approx_max_k(ued_scores, self.mutation_subsample_size)
			levels, parent_idxs = upsample_levels(levels, level_idxs, top_level_idxs)

		n_parallel = level_idxs.shape[-1]
		vrngs = jax.vmap(lambda subrng: jax.random.split(subrng, n_parallel))(
			jax.random.split(rng, self.n_students)
		)

		mutated_levels = jax.vmap(
			lambda *args: jax.vmap(self.mutation_fn, in_axes=(0,None,0,None))(*args),
			in_axes=(0,None,0,None)
		)(vrngs, self.benv.env_params, levels, self.n_mutations)

		# Mutated levels do not have existing idxs in the PLR buffer.
		mutated_level_idxs = jnp.full((self.n_students, n_parallel), -1)

		return mutated_levels, mutated_level_idxs, parent_idxs

	def _efficient_grad_update(self, rng, train_state, train_batch, is_replay):
		# PPOAgent vmaps over the train state and batch. Batch must be N x EM
		skip_grad_update = jnp.logical_and(self.use_robust_plr, ~is_replay)

		if self.n_students == 1:
			train_state, stats = jax.lax.cond(
				skip_grad_update[0],
				partial(self.student_pop.update, fake=True),
				self.student_pop.update,
				*(rng, train_state, train_batch)
			)
		elif self.n_students > 1: # Have to vmap all students + take only students that need updates
			_, dummy_stats = jax.vmap(lambda *_: self.student_pop.agent.get_empty_update_stats())(np.arange(self.n_students))
			_train_state, stats = self.student.update(rng, train_state, train_batch)
			train_state, stats = jax.vmap(lambda cond,x,y: \
					jax.tree_map(lambda _cond,_x,_y: jax.lax.select(_cond,_x,_y), cond, x, y))(
						is_replay, (train_state, stats), (_train_state, dummy_stats)
					)

		return train_state, stats

	@partial(jax.jit, static_argnums=(0,))
	def _compile_stats(self, update_stats, ep_stats, is_replay, env_metrics=None, plr_stats=None):
		stats = super()._compile_stats(update_stats, ep_stats, env_metrics, mask_passable=False)

		if self.track_env_metrics:
			if self.use_parallel_eval:
				env_passable_mask = env_metrics.pop('env/passable')
				env_env_metrics = {k:v for k,v in env_metrics.items() if k.startswith('env/')}
				env_env_metrics = jax.vmap(lambda info, mask: jax.tree.map(lambda x: (x*mask).sum()/mask.sum(), info))(env_env_metrics, env_passable_mask)
				env_env_metrics.update({'env/passable_ratio': jax.vmap(lambda x: x.mean())(env_passable_mask)})
				env_metrics.update(env_env_metrics)

				plr_passable_mask = env_metrics.pop('plr/passable')
				plr_env_metrics = {k:v for k,v in env_metrics.items() if k.startswith('plr/')}
				plr_env_metrics = jax.vmap(lambda info, mask: jax.tree.map(lambda x: (x*mask).sum()/mask.sum(), info))(plr_env_metrics, plr_passable_mask)
				plr_env_metrics.update({'plr/passable_ratio': jax.vmap(lambda x: x.mean())(plr_passable_mask)})
				env_metrics.update(plr_env_metrics)
			else:
				passable_mask = env_metrics.pop('passable')
				env_metrics = jax.vmap(lambda info, mask: jax.tree.map(lambda x: (x*mask).sum()/mask.sum(), info))(env_metrics, passable_mask)
				env_metrics.update({'passable_ratio': jax.vmap(lambda x: x.mean())(passable_mask)})

			if not self.use_parallel_eval:
				_env_metrics = jax.vmap(lambda cond: jax.lax.cond(
					cond,
					lambda *_: {f'env/{k}':-jnp.inf for k,_ in env_metrics.items()},
					lambda *_: {f'env/{k}':v.mean() for k,v in env_metrics.items()}
				))(is_replay)

				plr_env_metrics = jax.vmap(lambda cond: jax.lax.cond(
					cond,
					lambda *_: {f'plr/{k}':v.mean() for k,v in env_metrics.items()},
					lambda *_: {f'plr/{k}':-jnp.inf for k,_ in env_metrics.items()},
				))(is_replay)

				stats.update(jax.tree.map(lambda x: x[0], plr_env_metrics))
				stats.update(jax.tree.map(lambda x: x[0], _env_metrics))
			else:
				stats = {k:v for k,v in stats.items() if not k.startswith('env/')}
				env_metrics = {k:v.mean() for k,v in env_metrics.items()}
				stats.update(env_metrics)

		if plr_stats is not None:
			plr_stats = jax.vmap(lambda info: jax.tree_util.tree_map(lambda x: x.mean(), info))(plr_stats)
			plr_stats = jax.tree.map(lambda x: x[0], plr_stats)
			stats.update({f'plr/{k}':v for k,v in plr_stats.items()})

		if self.n_devices > 1:
			stats = jax.tree.map(lambda x: jax.lax.pmean(x, 'device'), stats)

		return stats


	@partial(jax.jit, static_argnums=(0,))
	def run(
		self, 
		rng, 
		train_state, 
		state, 
		start_state,
		obs, 
		carry=None, 
		extra=None, 
		ep_stats=None,
		plr_buffer=None):
		# If device sharded, load sharded PLR buffer into train state
		if self.n_devices > 1:
			rng = jax.random.fold_in(rng, jax.lax.axis_index('device'))
			train_state = train_state.replace(plr_buffer=plr_buffer)

		# Sample next training levels via PLR
		rng, *vrngs = jax.random.split(rng, self.n_students+1)
		obs, state, extra = self.benv.reset(jnp.array(vrngs), self.n_parallel, 1)

		if self.use_parallel_eval:
			n_level_samples = self.n_parallel//self._n_parallel_batches
			new_levels = jax.tree_map(lambda x: x.at[:,n_level_samples:2*n_level_samples].get(), state)
		else:
			n_level_samples = self.n_parallel
			new_levels = state

		rng, subrng = jax.random.split(rng)
		levels, level_idxs, is_replay, next_plr_buffer = \
			self.plr_mgr.sample(subrng, train_state.plr_buffer, new_levels, n_level_samples)
		train_state = train_state.replace(plr_buffer=next_plr_buffer)

		# If use_parallel_eval=True, need to combine replay and non-replay levels together
		# Need to mutate levels as well
		parent_idxs = jnp.full((self.n_students, self.n_parallel), -1)
		if self.use_parallel_eval: # Parallel ACCEL
			new_level_idxs = jnp.full_like(parent_idxs, -1)

			_all_levels = jax.vmap(
				lambda x,y: _tree_util.pytree_merge(x,y, start_idx=n_level_samples, src_len=n_level_samples),
				)(state, levels)
			_all_level_idxs = jax.vmap(
				lambda x,y: _tree_util.pytree_merge(x,y, start_idx=n_level_samples, src_len=n_level_samples),
				)(new_level_idxs, level_idxs)

			if self.use_mutations:
				rng, subrng = jax.random.split(rng)
				mutated_levels, mutated_level_idxs, _parent_idxs = self._mutate_levels(subrng, levels, level_idxs)
				
				fallback_levels = jax.tree_map(lambda x: x.at[:,-n_level_samples:].get(), state)
				fallback_level_idxs = jnp.full_like(mutated_level_idxs, -1)

				mutated_levels = jax.vmap(
					lambda cond,x,y: jax.tree_map(
						lambda _x,_y: jax.lax.select(cond,_x,_y), x, y
					))(is_replay, mutated_levels, fallback_levels)

				mutated_level_idxs = jax.vmap(
					lambda cond,x,y: jax.tree_map(
						lambda _x,_y: jax.lax.select(cond,_x,_y), x, y
					))(is_replay, mutated_level_idxs, fallback_level_idxs)

				_parent_idxs = jax.vmap(
					lambda cond,x,y: jax.tree_map(
						lambda _x,_y: jax.lax.select(cond,_x,_y), x, y
					))(is_replay, _parent_idxs, fallback_level_idxs)
		
				mutated_levels_start_idx = 2*n_level_samples
				_all_levels = jax.vmap(
					lambda x,y: _tree_util.pytree_merge(x,y, start_idx=mutated_levels_start_idx, src_len=n_level_samples),
					)(_all_levels, mutated_levels)
				_all_level_idxs = jax.vmap(
					lambda x,y: _tree_util.pytree_merge(x,y, start_idx=mutated_levels_start_idx, src_len=n_level_samples),
					)(_all_level_idxs, mutated_level_idxs)
				parent_idxs = jax.vmap(
					lambda x,y: _tree_util.pytree_merge(x,y, start_idx=mutated_levels_start_idx, src_len=n_level_samples),
					)(parent_idxs, _parent_idxs)

			levels = _all_levels
			level_idxs = _all_level_idxs

		# dedupe levels, move into PLR buffer logic
		if self.force_unique:
			level_idxs, dupe_mask = self.plr_mgr.dedupe_levels(next_plr_buffer, levels, level_idxs)
		else:
			dupe_mask = None 

		# Evaluate levels + update PLR
		result = self._eval_and_update_plr(
			rng, levels, level_idxs, train_state, update_plr=jnp.array([True]*self.n_students), parent_idxs=parent_idxs, dupe_mask=dupe_mask)
		rng, train_state, state, start_state, obs, carry, extra, ep_stats, \
			rollout_start_state, train_batch, ued_scores = result

		if self.use_parallel_eval:
			replay_start_idx = self.n_eval*n_level_samples
			replay_end_idx = 2*replay_start_idx
			train_batch = jax.vmap(
				lambda x: jax.tree_map(
					lambda _x: _x.at[:,replay_start_idx:replay_end_idx].get(), x)
				)(train_batch)

		# Gradient update
		rng, subrng = jax.random.split(rng)
		train_state, update_stats = self._efficient_grad_update(subrng, train_state, train_batch, is_replay)

		# Mutation step
		use_mutations = jnp.logical_and(self.use_mutations, is_replay)
		use_mutations = jnp.logical_and(use_mutations, not self.use_parallel_eval) # Already mutated above in parallel
		rng, arng, brng = jax.random.split(rng, 3)

		mutated_levels, mutated_level_idxs, parent_idxs = jax.lax.cond(
			use_mutations.any(),
			self._mutate_levels,
			lambda *_: (levels, level_idxs, jnp.full_like(level_idxs, -1)),
			*(arng, levels, level_idxs, ued_scores)
		)

		mutated_dupe_mask = jnp.zeros_like(mutated_level_idxs, dtype=jnp.bool_)
		if self.force_unique: # Should move into update plr logic
			mutated_level_idxs, mutated_dupe_mask = jax.lax.cond(
				use_mutations.any(),
				self.plr_mgr.dedupe_levels,
				lambda *_: (mutated_level_idxs, mutated_dupe_mask),
				*(next_plr_buffer, mutated_levels, mutated_level_idxs)
			)

		mutation_eval_result = jax.lax.cond(
			use_mutations.any(),
			self._eval_and_update_plr,
			partial(self._eval_and_update_plr, fake=True),
			*(brng, mutated_levels, mutated_level_idxs, train_state, use_mutations, parent_idxs, mutated_dupe_mask)
		)
		train_state = mutation_eval_result[1]

		# Collect training env metrics
		if self.track_env_metrics:
			if self.use_parallel_eval:
				new_levels = jax.tree.map(lambda x: x.at[:,:n_level_samples].get(), levels)
				replay_levels = jax.tree.map(lambda x: x.at[:,n_level_samples:2*n_level_samples].get(), levels)
				new_level_metrics = self.benv.get_env_metrics(new_levels)
				replay_level_metrics = self.benv.get_env_metrics(replay_levels)
				env_metrics = {f'env/{k}':v for k,v in new_level_metrics.items()}
				env_metrics.update({
						f'plr/{k}':v for k,v in replay_level_metrics.items()
				})
			else:
				env_metrics = self.benv.get_env_metrics(levels)
		else:
			env_metrics = None

		plr_stats = self.plr_mgr.get_metrics(train_state.plr_buffer)

		stats = self._compile_stats(update_stats, ep_stats, is_replay, env_metrics, plr_stats)

		if self.n_devices > 1:
			plr_buffer = train_state.plr_buffer
			train_state = train_state.replace(plr_buffer=None)

		train_state = train_state.increment()
		stats.update(dict(n_updates=train_state.n_updates[0]))

		return (
			stats, 
			rng, 
			train_state, 
			state, 
			start_state, 
			obs, 
			carry, 
			extra, 
			ep_stats,
			plr_buffer
		)
		