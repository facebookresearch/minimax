"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from typing import Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

import minimax.envs as envs
from minimax.util.rl import AgentPop, RollingStats
import minimax.util.pytree as _tree_util


def generate_all_kwargs_combos(arg_choices):
    def update_kwargs_with_choices(prev_combos, k, choices):
        updated_combos = []
        for v in choices:
            for p in prev_combos:
                updated = p.copy()
                updated[k] = v
                updated_combos.append(updated)

        return updated_combos

    all_combos = [{}]
    for k, choices in arg_choices.items():
        all_combos = update_kwargs_with_choices(all_combos, k, choices)

    return all_combos


def create_envs_for_kwargs(env_names, kwargs):
    # Check for csv kwargs
    arg_choices = {}
    varied_args = []
    for k, v in kwargs.items():
        if isinstance(v, str) and "," in v:
            vs = eval(v)
            arg_choices[k] = vs
            varied_args.append(k)
        elif isinstance(v, str):
            arg_choices[k] = [eval(v)]
        else:
            arg_choices[k] = [v]

    # List of kwargs
    kwargs_combos = generate_all_kwargs_combos(arg_choices)

    env_infos = []
    incl_ext = len(varied_args) > 0
    for name in env_names:
        for kwargs in kwargs_combos:
            if incl_ext and len(kwargs) > 0:
                ext = ":".join([f"{k}={kwargs[k]}" for k in varied_args])
                ext_name = f"{name}:{ext}"
            else:
                ext_name = name
            env_infos.append((name, ext_name, kwargs))

    return env_infos


class EvalRunner:
    def __init__(
        self,
        pop,
        env_names,
        env_kwargs=None,
        n_episodes=10,
        agent_idxs="*",
        render_mode=None,
    ):

        self.pop = pop

        if isinstance(agent_idxs, str):
            if "*" in agent_idxs:
                self.agent_idxs = np.arange(pop.n_agents)
            else:
                self.agent_idxs = np.array([int(x) for x in agent_idxs.split(",")])
        else:
            self.agent_idxs = agent_idxs  # assume array

        assert np.max(self.agent_idxs) < pop.n_agents, "Agent index is out of bounds."

        if isinstance(env_names, str):
            env_names = [x.strip() for x in env_names.split(",")]

        self.n_episodes = n_episodes
        env_infos = create_envs_for_kwargs(env_names, env_kwargs)
        env_names = []
        self.ext_env_names = []
        env_kwargs = []
        for name, ext_name, kwargs in env_infos:
            env_names.append(name)
            self.ext_env_names.append(ext_name)
            env_kwargs.append(kwargs)
        self.n_envs = len(env_names)

        self.benvs = []
        self.env_params = []
        self.env_has_solved_rate = []
        for env_name, kwargs in zip(env_names, env_kwargs):
            benv = envs.BatchEnv(
                env_name=env_name,
                n_parallel=n_episodes,
                n_eval=1,
                env_kwargs=kwargs,
                wrappers=["monitor_return", "monitor_ep_metrics"],
            )
            self.benvs.append(benv)
            self.env_params.append(benv.env.params)
            self.env_has_solved_rate.append(benv.env.eval_solved_rate is not None)

        self.action_dtype = self.benvs[0].env.action_space().dtype

        monitored_metrics = self.benvs[0].env.get_monitored_metrics()
        self.rolling_stats = RollingStats(names=monitored_metrics, window=1)
        self._update_ep_stats = jax.vmap(
            jax.vmap(self.rolling_stats.update_stats, in_axes=(0, 0, 0, None)),
            in_axes=(0, 0, 0, None),
        )

        self.test_return_pre = "test_return"
        self.test_solved_rate_pre = "test_solved_rate"

        self.render_mode = render_mode
        if render_mode:
            from minimax.envs.viz.grid_viz import GridVisualizer

            self.viz = GridVisualizer()
            self.viz.show()

            if render_mode == "ipython":
                from IPython import display

                self.ipython_display = display

    def load_checkpoint_state(self, runner_state, state):
        runner_state = list(runner_state)
        runner_state[1] = runner_state[1].load_state_dict(state[1])

        return tuple(runner_state)

    @partial(jax.jit, static_argnums=(0, 2))
    def _get_transition(self, rng, benv, params, state, obs, carry, zero_carry, extra):
        value, pi_params, next_carry = self.pop.act(params, obs, carry)
        pi = self.pop.get_action_dist(pi_params, dtype=self.action_dtype)
        rng, subrng = jax.random.split(rng)
        action = pi.sample(seed=subrng)
        log_pi = pi.log_prob(action)

        rng, *vrngs = jax.random.split(rng, self.pop.n_agents + 1)

        step_args = (jnp.array(vrngs), state, action, extra)
        (next_obs, next_state, reward, done, info, extra) = benv.step(*step_args)

        # Add transition to storage
        step = (obs, action, reward, done, log_pi, value)
        if carry is not None:
            step += (carry,)

        # Zero carry if needed
        if carry is not None:
            next_carry = jax.vmap(_tree_util.pytree_select)(
                done, zero_carry, next_carry
            )

        if self.render_mode:
            self.viz.render(
                benv.env.params, jax.tree_util.tree_map(lambda x: x[0][0], state)
            )
            if self.render_mode == "ipython":
                self.ipython_display.display(self.viz.window.fig)
                self.ipython_display.clear_output(wait=True)

        return next_state, next_obs, next_carry, done, info, extra

    @partial(jax.jit, static_argnums=(0, 2))
    def _rollout_benv(
        self,
        rng,
        benv,
        params,
        env_params,
        state,
        obs,
        carry,
        zero_carry,
        extra,
        ep_stats,
    ):

        def _scan_rollout(scan_carry, rng):
            (state, obs, carry, extra, ep_stats) = scan_carry

            step = self._get_transition(
                rng, benv, params, state, obs, carry, zero_carry, extra
            )

            (next_state, next_obs, next_carry, done, info, extra) = step

            ep_stats = self._update_ep_stats(ep_stats, done, info, 1)

            return (next_state, next_obs, next_carry, extra, ep_stats), None

        n_steps = benv.env.max_episode_steps()
        rngs = jax.random.split(rng, n_steps)
        (state, obs, carry, extra, ep_stats), _ = jax.lax.scan(
            _scan_rollout, (state, obs, carry, extra, ep_stats), rngs, length=n_steps
        )

        return ep_stats

    @partial(jax.jit, static_argnums=(0,))
    def run(self, rng, params):
        """
        Rollout agents on each env.

        For each env, run n_eval episodes in parallel,
        where each is indexed to return in order.
        """
        eval_stats = self.fake_run(rng, params)
        rng, *rollout_rngs = jax.random.split(rng, self.n_envs + 1)
        for i, (benv, env_param) in enumerate(zip(self.benvs, self.env_params)):
            rng, *reset_rngs = jax.random.split(rng, self.pop.n_agents + 1)
            obs, state, extra = benv.reset(jnp.array(reset_rngs))

            if self.pop.agent.is_recurrent:
                rng, subrng = jax.random.split(rng)
                zero_carry = self.pop.init_carry(subrng, obs)
            else:
                zero_carry = None

            # Reset episodic stats
            ep_stats = self.rolling_stats.reset_stats(
                batch_shape=(self.pop.n_agents, self.n_episodes)
            )

            ep_stats = self._rollout_benv(
                rollout_rngs[i],
                benv,
                jax.lax.stop_gradient(params),
                env_param,
                state,
                obs,
                zero_carry,
                zero_carry,
                extra,
                ep_stats,
            )

            env_name = self.ext_env_names[i]
            mean_return = ep_stats["return"].mean(1)

            if self.env_has_solved_rate[i]:
                mean_solved_rate = jax.vmap(jax.vmap(benv.env.eval_solved_rate))(
                    ep_stats
                ).mean(1)

            for idx in self.agent_idxs:
                eval_stats[f"eval/a{idx}:{self.test_return_pre}:{env_name}"] = (
                    mean_return[idx].squeeze()
                )
                if self.env_has_solved_rate[i]:
                    eval_stats[
                        f"eval/a{idx}:{self.test_solved_rate_pre}:{env_name}"
                    ] = mean_solved_rate[idx].squeeze()

        return eval_stats

    def fake_run(self, rng, params):
        eval_stats = {}
        for i, env_name in enumerate(self.ext_env_names):
            for idx in self.agent_idxs:
                eval_stats.update(
                    {f"eval/a{idx}:{self.test_return_pre}:{env_name}": 0.0}
                )
                if self.env_has_solved_rate[i]:
                    eval_stats.update(
                        {
                            f"eval/a{idx}:{self.test_solved_rate_pre}:{env_name}": 0.0,
                        }
                    )

        return eval_stats
