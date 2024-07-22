"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass
from collections import namedtuple, OrderedDict
from functools import partial
from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict

from .common import EnvInstance, make_maze_map
from minimax.envs import environment, spaces
from minimax.envs.registration import register_ued


class SequentialActions(IntEnum):
    skip = 0
    wall = 1
    goal = 2
    agent = 3


@struct.dataclass
class EnvState:
    encoding: chex.Array
    time: int
    terminal: bool


@struct.dataclass
class EnvParams:
    height: int = 15
    width: int = 15
    n_walls: int = 25
    noise_dim: int = 50
    replace_wall_pos: bool = False
    fixed_n_wall_steps: bool = False
    first_wall_pos_sets_budget: bool = False
    use_seq_actions: bool = (False,)
    set_agent_dir: bool = False
    normalize_obs: bool = False
    singleton_seed: int = -1


class UEDMaze(environment.Environment):
    def __init__(
        self,
        height=13,
        width=13,
        n_walls=25,
        noise_dim=16,
        replace_wall_pos=False,
        fixed_n_wall_steps=False,
        first_wall_pos_sets_budget=False,
        use_seq_actions=False,
        set_agent_dir=False,
        normalize_obs=False,
    ):
        """
        Using the original action space requires ensuring proper handling
        of a sequence with trailing dones, e.g. dones: 0 0 0 0 1 1 1 1 1 ... 1.
        Advantages and value losses should only be computed where ~dones[0].
        """
        assert not (
            first_wall_pos_sets_budget and fixed_n_wall_steps
        ), "Setting first_wall_pos_sets_budget=True requires fixed_n_wall_steps=False."

        super().__init__()

        self.n_tiles = height * width
        self.action_set = jnp.array(
            jnp.arange(self.n_tiles)
        )  # go straight, turn left, turn right, take action

        self.params = EnvParams(
            height=height,
            width=width,
            n_walls=n_walls,
            noise_dim=noise_dim,
            replace_wall_pos=replace_wall_pos,
            fixed_n_wall_steps=fixed_n_wall_steps,
            first_wall_pos_sets_budget=first_wall_pos_sets_budget,
            use_seq_actions=False,
            set_agent_dir=set_agent_dir,
            normalize_obs=normalize_obs,
        )

    @staticmethod
    def align_kwargs(kwargs, other_kwargs):
        kwargs.update(
            dict(
                height=other_kwargs["height"],
                width=other_kwargs["width"],
            )
        )

        return kwargs

    def _add_noise_to_obs(self, rng, obs):
        if self.params.noise_dim > 0:
            noise = jax.random.uniform(rng, (self.params.noise_dim,))
            obs.update(dict(noise=noise))

        return obs

    def reset_env(self, key: chex.PRNGKey):
        """
        Prepares the environment state for a new design
        from a blank slate.
        """
        params = self.params
        noise_rng, dir_rng = jax.random.split(key)
        encoding = jnp.zeros((self._get_encoding_dim(),), dtype=jnp.uint32)

        if not params.set_agent_dir:
            rand_dir = jax.random.randint(
                dir_rng, (), minval=0, maxval=4
            )  # deterministic
            tile_scale_dir = jnp.ceil((rand_dir / 4) * self.n_tiles).astype(jnp.uint32)
            encoding = encoding.at[-1].set(tile_scale_dir)

        state = EnvState(
            encoding=encoding,
            time=0,
            terminal=False,
        )

        obs = self._add_noise_to_obs(noise_rng, self.get_obs(state))

        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """
        Take a design step.
                action: A pos as an int from 0 to (height*width)-1
        """
        params = self.params

        collision_rng, noise_rng = jax.random.split(key)

        # Sample a random free tile in case of a collision
        dist_values = jnp.logical_and(  # True if position taken
            jnp.ones(params.n_walls + 2),
            jnp.arange(params.n_walls + 2) + 1 > state.time,
        )

        # Get zero-indexed last wall time step
        if params.fixed_n_wall_steps:
            max_n_walls = params.n_walls
            encoding_pos = state.encoding[: params.n_walls + 2]
            last_wall_step_idx = max_n_walls - 1
        else:
            max_n_walls = jnp.round(
                params.n_walls * state.encoding[0] / self.n_tiles
            ).astype(jnp.uint32)

            if self.params.first_wall_pos_sets_budget:
                encoding_pos = state.encoding[: params.n_walls + 2]
                last_wall_step_idx = jnp.maximum(max_n_walls, 1) - 1
            else:
                encoding_pos = state.encoding[1 : params.n_walls + 3]
                last_wall_step_idx = max_n_walls

        pos_dist = (
            jnp.ones(self.n_tiles).at[jnp.flip(encoding_pos)].set(jnp.flip(dist_values))
        )
        all_pos = jnp.arange(self.n_tiles, dtype=jnp.uint32)

        # Only mark collision if replace_wall_pos=False OR the agent is placed over the goal
        goal_step_idx = last_wall_step_idx + 1
        agent_step_idx = last_wall_step_idx + 2

        # Track whether it is the last time step
        next_state = state.replace(time=state.time + 1)
        done = self.is_terminal(next_state)

        # Always place agent idx in last enc position.
        is_agent_dir_step = jnp.logical_and(params.set_agent_dir, done)

        collision = jnp.logical_and(
            pos_dist[action] < 1,
            jnp.logical_or(
                not params.replace_wall_pos,
                jnp.logical_and(  # agent pos cannot override goal
                    jnp.equal(state.time, agent_step_idx),
                    jnp.equal(state.encoding[goal_step_idx], action),
                ),
            ),
        )
        collision = (collision * (1 - is_agent_dir_step)).astype(jnp.uint32)

        action = (1 - collision) * action + collision * jax.random.choice(
            collision_rng, all_pos, replace=False, p=pos_dist
        )

        enc_idx = (1 - is_agent_dir_step) * state.time + is_agent_dir_step * (-1)
        encoding = state.encoding.at[enc_idx].set(action)

        next_state = next_state.replace(encoding=encoding, terminal=done)
        reward = 0

        obs = self._add_noise_to_obs(noise_rng, self.get_obs(next_state))

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(next_state),
            reward,
            done,
            {},
        )

    def get_env_instance(self, key: chex.PRNGKey, state: EnvState) -> chex.Array:
        """
        Converts internal encoding to an instance encoding that
        can be interpreted by the `set_to_instance` method
        the paired Environment class.
        """
        params = self.params
        h = params.height
        w = params.width
        enc = state.encoding

        # === Extract agent_dir, agent_pos, and goal_pos ===
        # Num walls placed currently
        if params.fixed_n_wall_steps:
            n_walls = params.n_walls
            enc_len = self._get_encoding_dim()
            wall_pos_idx = jnp.flip(enc[: params.n_walls])
            agent_pos_idx = enc_len - 2  # Enc is full length
            goal_pos_idx = enc_len - 3
        else:
            n_walls = jnp.round(params.n_walls * enc[0] / self.n_tiles).astype(
                jnp.uint32
            )
            if params.first_wall_pos_sets_budget:
                wall_pos_idx = jnp.flip(
                    enc[: params.n_walls]
                )  # So 0-padding does not override pos=0
                enc_len = n_walls + 2  # [wall_pos] + len((goal, agent))
            else:
                wall_pos_idx = jnp.flip(enc[1 : params.n_walls + 1])
                enc_len = n_walls + 3  # [wall_pos] + len((n_walls, goal, agent))
            agent_pos_idx = (
                enc_len - 1
            )  # Positions are relative to n_walls when n_walls is variable.
            goal_pos_idx = enc_len - 2

        # Get agent + goal info (set agent/goal pos 1-step out of range if they are not yet placed)
        goal_placed = state.time > jnp.array([goal_pos_idx], dtype=jnp.uint32)
        goal_pos = goal_placed * jnp.array(
            [enc[goal_pos_idx] % w, enc[goal_pos_idx] // w], dtype=jnp.uint32
        ) + (~goal_placed) * jnp.array([w, h], dtype=jnp.uint32)

        agent_placed = state.time > jnp.array([agent_pos_idx], dtype=jnp.uint32)
        agent_pos = agent_placed * jnp.array(
            [enc[agent_pos_idx] % w, enc[agent_pos_idx] // w], dtype=jnp.uint32
        ) + (~agent_placed) * jnp.array([w, h], dtype=jnp.uint32)

        agent_dir_idx = jnp.floor((4 * enc[-1] / self.n_tiles)).astype(jnp.uint8)

        # Make wall map
        wall_start_time = jnp.logical_and(  # 1 if explicitly predict # blocks, else 0
            not params.fixed_n_wall_steps, not params.first_wall_pos_sets_budget
        ).astype(jnp.uint32)
        wall_map = jnp.zeros(h * w, dtype=jnp.bool_)
        wall_values = jnp.arange(params.n_walls) + wall_start_time < jnp.minimum(
            state.time, n_walls + wall_start_time
        )
        wall_values = jnp.flip(wall_values)
        wall_map = wall_map.at[wall_pos_idx].set(wall_values)

        # Zero out walls where agent and goal reside
        agent_mask = (
            agent_placed * (~(jnp.arange(h * w) == state.encoding[agent_pos_idx]))
            + ~agent_placed * wall_map
        )
        goal_mask = (
            goal_placed * (~(jnp.arange(h * w) == state.encoding[goal_pos_idx]))
            + ~goal_placed * wall_map
        )
        wall_map = wall_map * agent_mask * goal_mask
        wall_map = wall_map.reshape(h, w)

        return EnvInstance(
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            goal_pos=goal_pos,
            wall_map=wall_map,
        )

    def is_terminal(self, state: EnvState) -> bool:
        # if params.fixed_n_wall_steps:
        # 	max_n_walls = params.n_walls
        # 	done_steps = state.time >= self.max_episode_steps()
        # else:
        # 	max_n_walls = jnp.round(params.n_walls*state.encoding[0]/self.n_tiles)
        # 	max_episode_steps = \
        # 		self.max_episode_steps() - (params.n_walls - max_n_walls)
        # 	done_steps = state.time >= max_episode_steps
        done_steps = state.time >= self.max_episode_steps()
        return jnp.logical_or(done_steps, state.terminal)

    def _get_post_terminal_obs(self, state: EnvState):
        dtype = jnp.float32 if self.params.normalize_obs else jnp.uint8
        image = jnp.zeros(
            (self.params.height + 2, self.params.width + 2, 3), dtype=dtype
        )

        return OrderedDict(
            dict(
                image=image,
                time=state.time,
                noise=jnp.zeros(self.params.noise_dim, dtype=jnp.float32),
            )
        )

    def get_obs(self, state: EnvState):
        instance = self.get_env_instance(jax.random.PRNGKey(0), state)

        image = make_maze_map(
            self.params,
            instance.wall_map,
            instance.goal_pos,
            instance.agent_pos,
            instance.agent_dir_idx,
            pad_obs=False,
        )

        if self.params.normalize_obs:
            image = image / 10.0

        return OrderedDict(
            dict(
                image=image,
                time=state.time,
            )
        )

    @property
    def default_params(self):
        return EnvParams()

    @property
    def name(self) -> str:
        """Environment name."""
        return "UEDMaze"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        params = self.params
        return spaces.Discrete(params.height * params.width, dtype=jnp.uint32)

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        params = self.params
        max_episode_steps = self.max_episode_steps()
        spaces_dict = {
            "image": spaces.Box(0, 255, (params.height + 2, params.width + 2, 3)),
            "time": spaces.Discrete(max_episode_steps),
        }
        if self.params.noise_dim > 0:
            spaces_dict.update({"noise": spaces.Box(0, 1, (self.params.noise_dim,))})
        return spaces.Dict(spaces_dict)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        params = self.params
        encoding_dim = self._get_encoding_dim()
        max_episode_steps = self.max_episode_steps()
        h = params.height
        w = params.width
        return spaces.Dict(
            {
                "encoding": spaces.Box(0, 255, (encoding_dim,)),
                "time": spaces.Discrete(max_episode_steps),
                "terminal": spaces.Discrete(2),
            }
        )

    def _get_encoding_dim(self) -> int:
        encoding_dim = self.max_episode_steps()
        if not self.params.set_agent_dir:
            encoding_dim += 1  # max steps is 1 less than full encoding dim

        return encoding_dim

    def max_episode_steps(self) -> int:
        if self.params.fixed_n_wall_steps or self.params.first_wall_pos_sets_budget:
            max_episode_steps = self.params.n_walls + 2
        else:
            max_episode_steps = self.params.n_walls + 3

        if self.params.set_agent_dir:
            max_episode_steps += 1

        return max_episode_steps


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname

register_ued(env_id="Maze", entry_point=module_path + ":UEDMaze")
