"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from typing import Tuple, Optional
import inspect

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import optax
import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import minimax.envs as envs
from minimax.util import pytree as _tree_util
from minimax.util.rl import (
	AgentPop,
	VmapTrainState,
	RolloutStorage,
	RollingStats
)


class DRRunner:
	"""
	Orchestrates rollouts across one or more students. 
	The main components at play:
	- AgentPop: Manages train state and batched inference logic 
		for a population of agents.
	- BatchEnv: Manages environment step and reset logic, using a 
		populaton of agents.
	- RolloutStorage: Manages the storing and sampling of collected txns.
	- PPO: Handles PPO updates, which take a train state + batch of txns.
	"""
	def __init__(
		self, 
		env_name,
		env_kwargs,
		student_agents,
		n_students=1,
		n_parallel=1,
		n_eval=1,
		n_rollout_steps=256,
		lr=1e-4,
		lr_final=None,
		lr_anneal_steps=0,
		max_grad_norm=0.5,
		discount=0.99,
		gae_lambda=0.95,
		adam_eps=1e-5,
		normalize_return=False,
		track_env_metrics=False,
		n_unroll_rollout=1,
		n_devices=1,
		render=False):

		assert len(student_agents) == 1, 'Only one type of student supported.'
		assert n_parallel % n_devices == 0, 'Num envs must be divisible by num devices.'

		self.n_students = n_students
		self.n_parallel = n_parallel // n_devices
		self.n_eval = n_eval
		self.n_devices = n_devices
		self.step_batch_size = n_students*n_eval*n_parallel
		self.n_rollout_steps = n_rollout_steps
		self.n_updates = 0
		self.lr = lr
		self.lr_final = lr if lr_final is None else lr_final
		self.lr_anneal_steps = lr_anneal_steps
		self.max_grad_norm = max_grad_norm
		self.adam_eps = adam_eps
		self.normalize_return = normalize_return
		self.track_env_metrics = track_env_metrics
		self.n_unroll_rollout = n_unroll_rollout
		self.render = render

		self.student_pop = AgentPop(student_agents[0], n_agents=n_students)

		self.env, self.env_params = envs.make(
			env_name, 
			env_kwargs=env_kwargs
		)
		self._action_shape = self.env.action_space().shape

		self.benv = envs.BatchEnv(
			env_name=env_name,
			n_parallel=self.n_parallel,
			n_eval=self.n_eval,
			env_kwargs=env_kwargs,
			wrappers=['monitor_return', 'monitor_ep_metrics']
		)
		self.action_dtype = self.benv.env.action_space().dtype

		self.student_rollout = RolloutStorage(
			discount=discount,
			gae_lambda=gae_lambda,
			n_steps=n_rollout_steps,
			n_agents=n_students,
			n_envs=self.n_parallel,
			n_eval=self.n_eval,
			action_space=self.env.action_space(),
			obs_space=self.env.observation_space(),
			agent=self.student_pop.agent,
		)

		monitored_metrics = self.benv.env.get_monitored_metrics()
		self.rolling_stats = RollingStats(
			names=monitored_metrics,
			window=10,
		)
		self._update_ep_stats = jax.vmap(jax.vmap(self.rolling_stats.update_stats))

		if self.render:
			from envs.viz.grid_viz import GridVisualizer
			self.viz = GridVisualizer()
			self.viz.show()

	def reset(self, rng):
		self.n_updates = 0

		n_parallel = self.n_parallel*self.n_devices

		rngs, *vrngs = jax.random.split(rng, self.n_students+1)
		obs, state, extra = self.benv.reset(jnp.array(vrngs), n_parallel=n_parallel)
		dummy_obs = jax.tree_util.tree_map(lambda x: x[0], obs) # for one agent only

		rng, subrng = jax.random.split(rng)
		if self.student_pop.agent.is_recurrent:
			carry = self.student_pop.init_carry(subrng, obs)
			self.zero_carry = jax.tree_map(lambda x: x.at[:,:self.n_parallel].get(), carry)
		else:
			carry = None

		rng, subrng = jax.random.split(rng)
		params = self.student_pop.init_params(subrng, dummy_obs)

		schedule_fn = optax.linear_schedule(
			init_value=-float(self.lr),
			end_value=-float(self.lr_final),
			transition_steps=self.lr_anneal_steps,
		)

		tx = optax.chain(
			optax.clip_by_global_norm(self.max_grad_norm),
			optax.adam(learning_rate=float(self.lr), eps=self.adam_eps)
		)

		train_state = VmapTrainState.create(
			apply_fn=self.student_pop.agent.evaluate,
			params=params,
			tx=tx
		)

		ep_stats = self.rolling_stats.reset_stats(
			batch_shape=(self.n_students, n_parallel*self.n_eval))

		start_state = state

		return (
			rng, 
			train_state, 
			state,
			start_state, # Used to track metrics from starting state
			obs, 
			carry, 
			extra, 
			ep_stats
		)

	def get_checkpoint_state(self, state):
		_state = list(state)
		_state[1] = state[1].state_dict

		return _state

	def load_checkpoint_state(self, runner_state, state):
		runner_state = list(runner_state)
		runner_state[1] = runner_state[1].load_state_dict(state[1])

		return tuple(runner_state)

	@partial(jax.jit, static_argnums=(0,2))
	def _get_transition(
		self, 
		rng, 
		pop, 
		params, 
		rollout, 
		state, 
		start_state, 
		obs, 
		carry, 
		done,
		extra=None):
		# Sample action
		value, pi_params, next_carry = pop.act(params, obs, carry, done)

		pi = pop.get_action_dist(pi_params, dtype=self.action_dtype)
		rng, subrng = jax.random.split(rng)
		action = pi.sample(seed=subrng)
		log_pi = pi.log_prob(action)

		rng, *vrngs = jax.random.split(rng, self.n_students+1)
		(next_obs, 
		 next_state, 
		 reward, 
		 done, 
		 info, 
		 extra) = self.benv.step(jnp.array(vrngs), state, action, extra)

		next_start_state = jax.vmap(_tree_util.pytree_select)(
			done, next_state, start_state
		)

		# Add transition to storage
		step = (obs, action, reward, done, log_pi, value)
		if carry is not None:
			step += (carry,)

		rollout = self.student_rollout.append(rollout, *step)

		if self.render:
			self.viz.render(
				self.benv.env.params, 
				jax.tree_util.tree_map(lambda x: x[0][0], state))

		return (
			rollout, 
			next_state,
			next_start_state, 
			next_obs, 
			next_carry, 
			done, 
			info, 
			extra
		)

	@partial(jax.jit, static_argnums=(0,))
	def _rollout_students(
		self, 
		rng, 
		train_state, 
		state, 
		start_state, 
		obs, 
		carry, 
		done,
		extra=None, 
		ep_stats=None):
		rollout = self.student_rollout.reset()

		rngs = jax.random.split(rng, self.n_rollout_steps)

		def _scan_rollout(scan_carry, rng):
			rollout, state, start_state, obs, carry, done, extra, ep_stats, train_state = scan_carry 

			next_scan_carry = \
				self._get_transition(
					rng, 
					self.student_pop, 
					jax.lax.stop_gradient(train_state.params), 
					rollout, 
					state,
					start_state, 
					obs, 
					carry,
					done, 
					extra)
			(rollout, 
			 next_state,
			 next_start_state, 
			 next_obs, 
			 next_carry, 
			 done, 
			 info, 
			 extra) = next_scan_carry

			ep_stats = self._update_ep_stats(ep_stats, done, info)

			return (
				rollout, 
				next_state,
				next_start_state,
				next_obs, 
				next_carry,
				done,
				extra, 
				ep_stats,
				train_state), None

		(rollout, 
		 state, 
		 start_state, 
		 obs, 
		 carry, 
		 done,
		 extra, 
		 ep_stats,
		 train_state), _ = jax.lax.scan(
			_scan_rollout,
			(rollout, 
			 state, 
			 start_state,
			 obs, 
			 carry, 
			 done,
			 extra, 
			 ep_stats,
			 train_state),
			rngs,
			length=self.n_rollout_steps,
		)

		return rollout, state, start_state, obs, carry, extra, ep_stats, train_state

	@partial(jax.jit, static_argnums=(0,))
	def _compile_stats(self, update_stats, ep_stats, env_metrics=None):
		stats = jax.vmap(lambda info: jax.tree_map(lambda x: x.mean(), info))(
			{k:ep_stats[k] for k in self.rolling_stats.names}
		)
		stats.update(update_stats)

		if self.n_students > 1:
			_stats = {}
			for i in range(self.n_students):
				_student_stats = jax.tree_util.tree_map(lambda x: x[i], stats) # for agent0
				_stats.update({f'a{i}/{k}':v for k,v in _student_stats.items()})
			stats = _stats

		if self.track_env_metrics:
			mean_env_metrics = jax.vmap(lambda info: jax.tree_map(lambda x: x.mean(), info))(env_metrics)
			mean_env_metrics = {f'env/{k}':v for k,v in mean_env_metrics.items()}

			if self.n_students > 1:
				_env_metrics = {}
				for i in range(self.n_students):
					_student_env_metrics = jax.tree_util.tree_map(lambda x: x[i], mean_env_metrics) # for agent0
					_env_metrics.update({f'{k}_a{i}':v for k,v in _student_env_metrics.items()})
				mean_env_metrics = _env_metrics

			stats.update(mean_env_metrics)

		if self.n_students == 1:
			stats = jax.tree_map(lambda x: x[0], stats)

		if self.n_devices > 1:
			stats = jax.tree_map(lambda x: jax.lax.pmean(x, 'device'), stats)

		return stats

	def get_shmap_spec(self):
		runner_state_size = len(inspect.signature(self.run).parameters)
		in_spec = [P(None,'device'),]*(runner_state_size)
		out_spec = [P(None,'device'),]*(runner_state_size)

		in_spec[:2] = [P(None),]*2
		in_spec = tuple(in_spec)
		out_spec = (P(None),) + in_spec

		return in_spec, out_spec

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
		ep_stats=None):
		"""
		Perform one update step: rollout all students and teachers + update with PPO
		"""
		if self.n_devices > 1:
			rng = jax.random.fold_in(rng, jax.lax.axis_index('device'))

		rng, *vrngs = jax.random.split(rng, self.n_students+1)
		rollout_batch_shape = (self.n_students, self.n_parallel*self.n_eval)

		obs, state, extra = self.benv.reset(jnp.array(vrngs))
		ep_stats = self.rolling_stats.reset_stats(
			batch_shape=rollout_batch_shape)

		rollout_start_state = state

		done = jnp.zeros(rollout_batch_shape, dtype=jnp.bool_)
		rng, subrng = jax.random.split(rng)
		rollout, state, start_state, obs, carry, extra, ep_stats, train_state = \
			self._rollout_students(
				subrng, 
				train_state, 
				state, 
				start_state,
				obs, 
				carry, 
				done,
				extra, 
				ep_stats
			)

		train_batch = self.student_rollout.get_batch(
			rollout, 
			self.student_pop.get_value(
				jax.lax.stop_gradient(train_state.params), 
				obs, 
				carry,
			)
		)

		# PPOAgent vmaps over the train state and batch. Batch must be N x EM
		rng, subrng = jax.random.split(rng)
		train_state, update_stats = self.student_pop.update(subrng, train_state, train_batch)

		# Collect env metrics
		if self.track_env_metrics:
			env_metrics = self.benv.get_env_metrics(rollout_start_state)
		else:
			env_metrics = None

		stats = self._compile_stats(update_stats, ep_stats, env_metrics)
		stats.update(dict(n_updates=train_state.n_updates[0]))

		train_state = train_state.increment()
		self.n_updates += 1

		return (
			stats, 
			rng, 
			train_state, 
			state, 
			start_state, 
			obs, 
			carry, 
			extra, 
			ep_stats
		)
