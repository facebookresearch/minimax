"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
import chex

from minimax.envs.registration import register
from .common import (
    DIR_TO_VEC,
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    make_maze_map,
)
from .maze import Maze, EnvParams, EnvState, Actions


# ======== Singleton mazes ========
class MazeSingleton(Maze):
    def __init__(
        self,
        height=15,
        width=15,
        wall_map=None,
        goal_pos=None,
        agent_pos=None,
        agent_dir_idx=None,
        agent_view_size=5,
        see_through_walls=True,
        see_agent=False,
        normalize_obs=False,
        obs_agent_pos=False,
        max_episode_steps=None,
        singleton_seed=-1,
    ):
        super().__init__(
            height=height,
            width=width,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
            obs_agent_pos=obs_agent_pos,
            max_episode_steps=max_episode_steps,
            singleton_seed=singleton_seed,
        )

        if wall_map is None:
            self.wall_map = jnp.zeros((height, width), dtype=jnp.bool_)
        else:
            self.wall_map = jnp.array(
                [[int(x) for x in row.split()] for row in wall_map], dtype=jnp.bool_
            )
        height, width = self.wall_map.shape

        if max_episode_steps is None:
            max_episode_steps = (
                2 * (height + 2) * (width + 2)
            )  # Match original eval steps

        self.goal_pos_choices = None
        if goal_pos is None:
            self.goal_pos = jnp.array([height, width]) - jnp.ones(2, dtype=jnp.uint32)
        elif isinstance(goal_pos, (tuple, list)) and isinstance(
            goal_pos[0], (tuple, list)
        ):
            self.goal_pos_choices = jnp.array(goal_pos, dtype=jnp.uint32)
            self.goal_pos = goal_pos[0]
        else:
            self.goal_pos = jnp.array(goal_pos, dtype=jnp.uint32)

        if agent_pos is None:
            self.agent_pos = jnp.zeros(2, dtype=jnp.uint32)
        else:
            self.agent_pos = jnp.array(agent_pos, dtype=jnp.uint32)

        self.agent_dir_idx = agent_dir_idx

        if self.agent_dir_idx is None:
            self.agent_dir_idx = 0

        self.params = EnvParams(
            height=height,
            width=width,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
            obs_agent_pos=obs_agent_pos,
            max_episode_steps=max_episode_steps,
            singleton_seed=-1,
        )

        self.maze_map = make_maze_map(
            self.params,
            self.wall_map,
            self.goal_pos,
            self.agent_pos,
            self.agent_dir_idx,
            pad_obs=True,
        )

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def reset_env(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, EnvState]:

        if self.agent_dir_idx is None:
            key, subkey = jax.random.split(key)
            agent_dir_idx = jax.random.choice(subkey, 4)
        else:
            agent_dir_idx = self.agent_dir_idx

        if self.goal_pos_choices is not None:
            key, subkey = jax.random.split(key)
            goal_pos = jax.random.choice(subkey, self.goal_pos_choices)
            maze_map = make_maze_map(
                self.params,
                self.wall_map,
                goal_pos,
                self.agent_pos,
                agent_dir_idx,
                pad_obs=True,
            )
        else:
            goal_pos = self.goal_pos
            maze_map = self.maze_map

        state = EnvState(
            agent_pos=self.agent_pos,
            agent_dir=DIR_TO_VEC[agent_dir_idx],
            agent_dir_idx=agent_dir_idx,
            goal_pos=goal_pos,
            wall_map=self.wall_map,
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        return self.get_obs(state), state


# ======== Specific mazes ========
class SixteenRooms(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 1 0 0 1 0 0 1 0 0 0",
            "0 0 0 0 0 0 0 0 0 1 0 0 0",
            "0 0 0 1 0 0 1 0 0 0 0 0 0",
            "1 0 1 1 1 0 1 1 0 1 1 1 0",
            "0 0 0 1 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 1 0 0 1 0 0 0",
            "1 1 0 1 0 1 1 0 1 1 1 0 1",
            "0 0 0 1 0 0 0 0 0 1 0 0 0",
            "0 0 0 1 0 0 1 0 0 0 0 0 0",
            "0 1 1 1 1 0 1 1 0 1 0 1 1",
            "0 0 0 1 0 0 1 0 0 1 0 0 0",
            "0 0 0 0 0 0 1 0 0 0 0 0 0",
            "0 0 0 1 0 0 0 0 0 1 0 0 0",
        ]
        goal_pos = (11, 11)
        agent_pos = (1, 1)
        agent_dir_idx = 0

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class SixteenRooms2(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 1 0 0 0 0 0 1 0 0 0",
            "0 0 0 0 0 0 1 0 0 1 0 0 0",
            "0 0 0 1 0 0 1 0 0 1 0 0 0",
            "1 1 1 1 0 1 1 0 1 1 1 0 1",
            "0 0 0 1 0 0 1 0 0 0 0 0 0",
            "0 0 0 0 0 0 1 0 0 1 0 0 0",
            "1 0 1 1 1 1 1 0 1 1 1 1 1",
            "0 0 0 1 0 0 1 0 0 1 0 0 0",
            "0 0 0 1 0 0 0 0 0 0 0 0 0",
            "1 1 0 1 1 0 1 1 0 1 1 1 1",
            "0 0 0 1 0 0 1 0 0 1 0 0 0",
            "0 0 0 0 0 0 1 0 0 0 0 0 0",
            "0 0 0 1 0 0 1 0 0 1 0 0 0",
        ]
        goal_pos = (11, 11)
        agent_pos = (1, 1)
        agent_dir_idx = None

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class Labyrinth(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 1 1 1 1 1 1 1 1 1 1 1 0",
            "0 1 0 0 0 0 0 0 0 0 0 1 0",
            "0 1 0 1 1 1 1 1 1 1 0 1 0",
            "0 1 0 1 0 0 0 0 0 1 0 1 0",
            "0 1 0 1 0 1 1 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 0 0 1 0 0 0 1 0 1 0",
            "0 1 1 1 1 1 1 1 1 1 0 1 0",
            "0 0 0 0 0 1 0 0 0 0 0 1 0",
            "1 1 1 1 0 1 0 1 1 1 1 1 0",
            "0 0 0 0 0 1 0 0 0 0 0 0 0",
        ]
        goal_pos = (6, 6)
        agent_pos = (0, 12)
        agent_dir_idx = 0

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class LabyrinthFlipped(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 1 1 1 1 1 1 1 1 1 1 1 0",
            "0 1 0 0 0 0 0 0 0 0 0 1 0",
            "0 1 0 1 1 1 1 1 1 1 0 1 0",
            "0 1 0 1 0 0 0 0 0 1 0 1 0",
            "0 1 0 1 0 1 1 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 0 0 1 0 0 0 1 0",
            "0 1 0 1 1 1 1 1 1 1 1 1 0",
            "0 1 0 0 0 0 0 1 0 0 0 0 0",
            "0 1 1 1 1 1 0 1 0 1 1 1 1",
            "0 0 0 0 0 0 0 1 0 0 0 0 0",
        ]
        goal_pos = (6, 6)
        agent_pos = (12, 12)
        agent_dir_idx = 2

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class Labyrinth2(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 1 0 0 0 0 0 0 0 0 0 0 0",
            "0 1 0 1 1 1 1 1 1 1 1 1 0",
            "0 1 0 1 0 0 0 0 0 0 0 1 0",
            "0 1 0 1 0 1 1 1 1 1 0 1 0",
            "0 1 0 1 0 1 0 0 0 1 0 1 0",
            "0 0 0 1 0 1 0 1 0 1 0 1 0",
            "1 1 1 1 0 1 0 1 0 1 0 1 0",
            "0 0 0 1 0 1 1 1 0 1 0 1 0",
            "0 1 0 1 0 0 0 0 0 1 0 1 0",
            "0 1 0 1 1 1 1 1 1 1 0 1 0",
            "0 1 0 0 0 0 0 0 0 0 0 1 0",
            "0 1 1 1 1 1 1 1 1 1 1 1 0",
            "0 0 0 0 0 0 0 0 0 0 0 0 0",
        ]
        goal_pos = (6, 6)
        agent_pos = (0, 0)
        agent_dir_idx = None

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class StandardMaze(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 0 0 1 0 0 0 0 1 0 0",
            "0 1 1 1 0 1 1 1 1 0 1 1 0",
            "0 1 0 0 0 0 0 0 0 0 0 0 0",
            "0 1 1 1 1 1 1 1 1 0 1 1 1",
            "0 0 0 0 0 0 0 0 1 0 0 0 0",
            "1 1 1 1 1 1 0 1 1 1 1 1 0",
            "0 0 0 0 1 0 0 1 0 0 0 0 0",
            "0 1 1 0 0 0 1 1 0 1 1 1 1",
            "0 0 1 0 1 0 0 1 0 0 0 1 0",
            "1 0 1 0 1 1 0 1 1 1 0 1 0",
            "1 0 1 0 0 1 0 0 0 1 0 0 0",
            "1 0 1 1 0 1 1 1 0 1 1 1 0",
            "0 0 0 1 0 0 0 1 0 1 0 0 0",
        ]
        goal_pos = (6, 12)
        agent_pos = (6, 0)
        agent_dir_idx = 0

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class StandardMaze2(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 1 0 1 0 0 0 0 1 0 0",
            "0 1 0 1 0 1 1 1 1 0 0 0 1",
            "0 1 0 0 0 0 0 0 0 0 1 0 0",
            "0 1 1 1 1 1 1 1 1 0 1 1 1",
            "0 0 0 1 0 0 1 0 1 0 1 0 0",
            "1 1 0 1 0 1 1 0 1 0 1 0 0",
            "0 1 0 1 0 0 0 0 1 0 1 1 0",
            "0 1 0 1 1 0 1 1 1 0 0 1 0",
            "0 1 0 0 1 0 0 1 1 1 0 1 0",
            "0 1 1 0 1 1 0 1 0 1 0 1 0",
            "0 1 0 0 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 0 0 1 0 0 0 1 0 0 0 0 0",
        ]
        goal_pos = (12, 4)
        agent_pos = (0, 6)
        agent_dir_idx = None

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class StandardMaze3(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 0 1 0 1 0 0 0 0 0 0",
            "0 1 1 1 1 0 1 0 1 1 1 1 0",
            "0 1 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 1 1 1 1 0 1 0 1 0 1",
            "1 1 0 1 0 0 0 0 1 0 1 0 0",
            "0 0 0 1 0 1 1 0 1 0 1 1 0",
            "0 1 0 1 0 1 0 0 1 0 0 1 0",
            "0 1 0 1 0 1 0 1 1 1 0 1 1",
            "0 1 0 0 0 1 0 1 0 1 0 0 0",
            "0 1 1 1 0 1 0 1 0 1 1 1 0",
            "0 1 0 0 0 1 0 1 0 0 0 1 0",
            "0 1 0 1 1 1 0 1 0 1 0 1 0",
            "0 1 0 0 0 1 0 0 0 1 0 0 0",
        ]
        goal_pos = (12, 6)
        agent_pos = (3, 0)
        agent_dir_idx = None

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class SmallCorridor(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 1 1 1 1 1 1 1 1 1 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 0 0 0 0 0 0 0 0 0 0 0 0",
        ]
        goal_pos = [
            (2, 5),
            (4, 5),
            (6, 5),
            (8, 5),
            (10, 5),
            (2, 7),
            (4, 7),
            (6, 7),
            (8, 7),
            (10, 7),
        ]
        agent_pos = (0, 6)
        agent_dir_idx = None

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class LargeCorridor(MazeSingleton):
    def __init__(self, see_agent=False, normalize_obs=False):
        wall_map = [
            "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        ]
        goal_pos = [
            (2, 8),
            (4, 8),
            (6, 8),
            (8, 8),
            (10, 8),
            (12, 8),
            (14, 8),
            (16, 8),
            (2, 10),
            (4, 10),
            (6, 10),
            (8, 10),
            (10, 10),
            (12, 10),
            (14, 10),
            (16, 10),
        ]
        agent_pos = (0, 9)
        agent_dir_idx = None

        super().__init__(
            wall_map=wall_map,
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_dir_idx=agent_dir_idx,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
        )


class FourRooms(Maze):
    def __init__(
        self,
        height=17,
        width=17,
        agent_view_size=5,
        see_through_walls=True,
        see_agent=False,
        normalize_obs=False,
        max_episode_steps=250,
        singleton_seed=-1,
    ):

        super().__init__(
            height=height,
            width=width,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
            max_episode_steps=max_episode_steps,
            singleton_seed=singleton_seed,
        )

        assert height % 2 == 1 and width % 2 == 1, "Grid height and width must be odd"

        wall_map = jnp.zeros((height, width), dtype=jnp.bool_)
        wall_map = wall_map.at[height // 2, :].set(True)
        wall_map = wall_map.at[:, width // 2].set(True)
        self.wall_map = wall_map

        self.room_h = height // 2
        self.room_w = width // 2

        self.all_pos_idxs = jnp.arange(height * width)
        self.goal_pos_mask = (~wall_map).flatten()
        self.agent_pos_mask = self.goal_pos_mask

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # Randomize door positions
        params = self.params

        key, x_rng, y_rng = jax.random.split(key, 3)
        x_door_idxs = jax.random.randint(x_rng, (2,), 0, self.room_w) + jnp.array(
            [0, self.room_w + 1], dtype=jnp.uint32
        )

        y_door_idxs = jax.random.randint(y_rng, (2,), 0, self.room_h) + jnp.array(
            [0, self.room_h + 1], dtype=jnp.uint32
        )

        wall_map = self.wall_map.at[self.room_h, x_door_idxs].set(False)
        wall_map = wall_map.at[y_door_idxs, self.room_w].set(False)

        # Randomize goal pos
        key, subkey = jax.random.split(key)
        goal_pos_idx = jax.random.choice(
            subkey, self.all_pos_idxs, shape=(), p=self.goal_pos_mask
        )
        goal_pos = jnp.array(
            [goal_pos_idx % params.width, goal_pos_idx // params.width],
            dtype=jnp.uint32,
        )

        # Randomize agent pos
        key, subkey = jax.random.split(key)
        agent_pos_mask = self.agent_pos_mask.at[goal_pos_idx].set(False)
        agent_pos_idx = jax.random.choice(
            subkey, self.all_pos_idxs, shape=(), p=self.agent_pos_mask
        )
        agent_pos = jnp.array(
            [agent_pos_idx % params.width, agent_pos_idx // params.width],
            dtype=jnp.uint32,
        )

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, 4)

        maze_map = make_maze_map(
            self.params, wall_map, goal_pos, agent_pos, agent_dir_idx, pad_obs=True
        )

        state = EnvState(
            agent_pos=agent_pos,
            agent_dir=DIR_TO_VEC[agent_dir_idx],
            agent_dir_idx=agent_dir_idx,
            goal_pos=goal_pos,
            wall_map=wall_map,
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        return self.get_obs(state), state


class Crossing(Maze):
    def __init__(
        self,
        height=9,
        width=9,
        n_crossings=5,
        agent_view_size=5,
        see_through_walls=True,
        see_agent=False,
        normalize_obs=False,
        max_episode_steps=250,
        singleton_seed=-1,
    ):
        self.n_crossings = n_crossings
        max_episode_steps = 4 * (height + 2) * (width + 2)

        super().__init__(
            height=height,
            width=width,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
            max_episode_steps=max_episode_steps,
            singleton_seed=singleton_seed,
        )

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        params = self.params
        height, width = params.height, params.width
        goal_pos = jnp.array([width - 1, height - 1])
        agent_pos = jnp.array([0, 0], dtype=jnp.uint32)
        agent_dir_idx = 0

        # Generate walls
        wall_map = jnp.zeros((height, width), dtype=jnp.bool_)

        row_y_choices = jnp.arange(1, height - 1, 2)
        col_x_choices = jnp.arange(1, width - 1, 2)

        rng, subrng = jax.random.split(key)
        dirs = jax.random.permutation(
            subrng,
            jnp.concatenate(
                (jnp.zeros(len(row_y_choices)), jnp.ones(len(col_x_choices)))
            ),
        )[: self.n_crossings]

        n_v = sum(dirs.astype(jnp.uint32))
        n_h = len(dirs) - n_v

        rng, row_rng, col_rng = jax.random.split(rng, 3)

        row_ys_mask = jax.random.permutation(
            row_rng, (jnp.arange(len(row_y_choices)) < n_v).repeat(2)
        )
        if height % 2 == 0:
            row_ys_mask = jnp.concatenate((row_ys_mask, jnp.zeros(2)))
        else:
            row_ys_mask = jnp.concatenate((row_ys_mask, jnp.zeros(1)))

        row_ys_mask = jnp.logical_and(
            jnp.zeros(height, dtype=jnp.bool_).at[row_y_choices].set(True), row_ys_mask
        )

        col_xs_mask = jax.random.permutation(
            col_rng, (jnp.arange(len(col_x_choices)) < n_h).repeat(2)
        )
        if width % 2 == 0:
            col_xs_mask = jnp.concatenate((col_xs_mask, jnp.zeros(2)))
        else:
            col_xs_mask = jnp.concatenate((col_xs_mask, jnp.zeros(1)))

        col_xs_mask = jnp.logical_and(
            jnp.zeros(width, dtype=jnp.bool_).at[col_x_choices].set(True), col_xs_mask
        )

        wall_map = jnp.logical_or(
            wall_map, jnp.tile(jnp.expand_dims(row_ys_mask, -1), (1, width))
        )

        wall_map = jnp.logical_or(
            wall_map, jnp.tile(jnp.expand_dims(col_xs_mask, 0), (height, 1))
        )

        # Generate wall openings
        def _scan_step(carry, rng):
            wall_map, pos, passed_wall, last_dir, last_dir_idx = carry

            dir_idx = jax.random.randint(rng, (), 0, 2)

            go_dir = (~passed_wall) * DIR_TO_VEC[dir_idx] + passed_wall * last_dir
            next_pos = pos + go_dir

            # If next pos is the right border, force direction to be down
            collide = jnp.logical_or((next_pos[0] >= width), (next_pos[1] >= height))
            go_dir = collide * DIR_TO_VEC[(dir_idx + 1) % 2] + (~collide) * go_dir
            dir_idx = (dir_idx + 1) % 2 + (~collide) * dir_idx

            next_pos = collide * (pos + go_dir) + (~collide) * next_pos

            last_dir = go_dir
            last_dir_idx = dir_idx
            pos = next_pos

            passed_wall = wall_map[pos[1], pos[0]]
            wall_map = wall_map.at[pos[1], pos[0]].set(False)

            return (
                wall_map,
                pos.astype(jnp.uint32),
                passed_wall,
                last_dir,
                last_dir_idx,
            ), None

        n_steps_to_goal = width + height - 2
        rng, *subrngs = jax.random.split(rng, n_steps_to_goal + 1)

        pos = agent_pos
        passed_wall = jnp.array(False)
        last_dir = DIR_TO_VEC[0]

        (wall_map, pos, passed_wall, last_dir, last_dir_idx), _ = jax.lax.scan(
            _scan_step,
            (wall_map, pos, passed_wall, last_dir, 0),
            jnp.array(subrngs),
            length=n_steps_to_goal,
        )

        maze_map = make_maze_map(
            self.params, wall_map, goal_pos, agent_pos, agent_dir_idx, pad_obs=True
        )

        state = EnvState(
            agent_pos=agent_pos,
            agent_dir=DIR_TO_VEC[agent_dir_idx],
            agent_dir_idx=agent_dir_idx,
            goal_pos=goal_pos,
            wall_map=wall_map,
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        return self.get_obs(state), state


NEIGHBOR_WALL_OFFSETS = jnp.array(
    [
        [1, 0],  # right
        [0, 1],  # bottom
        [-1, 0],  # left
        [0, -1],  # top
        [0, 0],  # self
    ],
    dtype=jnp.int32,
)


class PerfectMaze(Maze):
    def __init__(
        self,
        height=13,
        width=13,
        agent_view_size=5,
        see_through_walls=True,
        see_agent=False,
        normalize_obs=False,
        max_episode_steps=250,
        singleton_seed=-1,
    ):

        assert height % 2 == 1 and width % 2 == 1, "Maze dimensions must be odd."

        max_episode_steps = 2 * (width + 2) * (height + 2)
        super().__init__(
            height=height,
            width=width,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
            max_episode_steps=max_episode_steps,
            singleton_seed=singleton_seed,
        )

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """
        Generate a perfect maze using an iterated search procedure.
        """
        params = self.params
        height, width = self.params.height, self.params.width
        n_tiles = height * width

        # Track maze wall map
        wall_map = jnp.ones((height, width), dtype=jnp.bool_)

        # Track visited, walkable tiles
        _h = height // 2 + 1
        _w = width // 2 + 1
        visited_map = jnp.zeros((_h, _w), dtype=jnp.bool_)
        vstack = jnp.zeros((_h * _w, 2), dtype=jnp.uint32)
        vstack_size = 0

        # Get initial start tile in walkable index
        key, subkey = jax.random.split(key)
        start_pos_x = jax.random.randint(subkey, (), 0, _w)
        start_pos_y = jax.random.randint(subkey, (), 0, _h)
        start_pos = jnp.array([start_pos_x, start_pos_y], dtype=jnp.uint32)

        # Set initial start tile as visited
        visited_map = visited_map.at[start_pos[1], start_pos[0]].set(True)
        wall_map = wall_map.at[2 * start_pos[1], 2 * start_pos[0]].set(False)
        vstack = vstack.at[vstack_size : vstack_size + 2].set(start_pos)
        vstack_size += 2

        def _scan_step(carry, key):
            # Choose last visited tile and move to a neighbor
            wall_map, visited_map, vstack, vstack_size = carry

            abs_pos = 2 * vstack[vstack_size - 1]

            neighbor_wall_offsets = NEIGHBOR_WALL_OFFSETS.at[-1].set(
                vstack[vstack_size - 2] - vstack[vstack_size - 1]
            )

            # Find a random unvisited neighbor
            neighbor_pos = jnp.minimum(
                jnp.maximum(
                    jnp.tile(abs_pos, (len(NEIGHBOR_WALL_OFFSETS), 1))
                    + 2 * neighbor_wall_offsets,
                    0,
                ),
                jnp.array([width, height], dtype=jnp.uint32),
            )

            # Check for unvisited neighbors. Set self to unvisited if all visited.
            neighbor_visited = visited_map.at[
                neighbor_pos[:, 1] // 2, neighbor_pos[:, 0] // 2
            ].get()

            n_neighbor_visited = neighbor_visited[:4].sum()
            all_visited = n_neighbor_visited == 4
            all_visited_post = n_neighbor_visited >= 3
            neighbor_visited = neighbor_visited.at[-1].set(~all_visited)

            # Choose a random unvisited neigbor and remove walls between current tile
            # and this neighbor and at this neighbor.
            rand_neighbor_idx = jax.random.choice(
                key, jnp.arange(len(NEIGHBOR_WALL_OFFSETS)), p=~neighbor_visited
            )
            rand_neighbor_pos = neighbor_pos[rand_neighbor_idx]
            rand_neighbor_wall_pos = (
                abs_pos + (~all_visited) * neighbor_wall_offsets[rand_neighbor_idx]
            )
            remove_wall_pos = jnp.concatenate(
                (
                    jnp.expand_dims(rand_neighbor_pos, 0),
                    jnp.expand_dims(rand_neighbor_wall_pos, 0),
                ),
                0,
            )
            wall_map = wall_map.at[remove_wall_pos[:, 1], remove_wall_pos[:, 0]].set(
                False
            )

            # Set selected neighbor as visited
            visited_map = visited_map.at[
                rand_neighbor_pos[1] // 2, rand_neighbor_pos[0] // 2
            ].set(True)

            # Pop current tile from stack if all neighbors have been visited
            vstack_size -= all_visited_post

            # Push selected neighbor onto stack
            vstack = vstack.at[vstack_size].set(rand_neighbor_pos // 2)
            vstack_size += ~all_visited

            return (wall_map, visited_map, vstack, vstack_size), None

        # for i in range(3*_h*_w):
        max_n_steps = 2 * _w * _h
        key, *subkeys = jax.random.split(key, max_n_steps + 1)
        (wall_map, visited_map, vstack, vstack_size), _ = jax.lax.scan(
            _scan_step,
            (wall_map, visited_map, vstack, vstack_size),
            jnp.array(subkeys),
            length=max_n_steps,
        )

        # Randomize goal position
        all_pos_idx = jnp.arange(height * width)

        key, subkey = jax.random.split(key)
        goal_mask = ~wall_map.flatten()
        goal_pos_idx = jax.random.choice(subkey, all_pos_idx, p=goal_mask)
        goal_pos = jnp.array([goal_pos_idx % width, goal_pos_idx // width])

        # Randomize agent position
        key, subkey = jax.random.split(key)
        agent_mask = goal_mask.at[goal_pos_idx].set(False)
        agent_pos_idx = jax.random.choice(subkey, all_pos_idx, p=agent_mask)
        agent_pos = jnp.array(
            [agent_pos_idx % width, agent_pos_idx // width], dtype=jnp.uint32
        )

        # Randomize agent dir
        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, 4)

        maze_map = make_maze_map(
            self.params, wall_map, goal_pos, agent_pos, agent_dir_idx, pad_obs=True
        )

        state = EnvState(
            agent_pos=agent_pos,
            agent_dir=DIR_TO_VEC[agent_dir_idx],
            agent_dir_idx=agent_dir_idx,
            goal_pos=goal_pos,
            wall_map=wall_map,
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        return self.get_obs(state), state


class PerfectMazeMedium(PerfectMaze):
    def __init__(self, *args, **kwargs):
        super().__init__(height=19, width=19, *args, **kwargs)


class PerfectMazeExtraLarge(PerfectMaze):
    def __init__(self, *args, **kwargs):
        super().__init__(height=101, width=101, *args, **kwargs)


class Memory(MazeSingleton):
    def __init__(
        self,
        height=17,
        width=17,
        agent_view_size=7,
        see_through_walls=True,
        see_agent=False,
        normalize_obs=False,
        obs_agent_pos=False,
        max_episode_steps=250,
        singleton_seed=-1,
    ):

        # Generate walls
        wall_map = [
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "1 1 1 1 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 1 1 1 1 1 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0 1 0 0 0 0",
            "0 0 0 1 1 1 1 1 1 0 1 0 0 0 0",
            "1 1 1 1 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
            "0 0 0 0 0 0 0 0 1 0 1 0 0 0 0",
        ]

        super().__init__(
            wall_map=wall_map,
            goal_pos=(9, 5),
            agent_pos=(0, 7),
            agent_dir_idx=0,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
            obs_agent_pos=obs_agent_pos,
            max_episode_steps=max_episode_steps,
        )

        self.top_pos = jnp.array([9, 5], dtype=jnp.uint32)
        self.bottom_pos = jnp.array([9, 9], dtype=jnp.uint32)

    def reset_env(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, EnvState]:
        params = self.params
        height, width = params.height, params.width

        agent_pos = jnp.array([0, 7], dtype=jnp.uint32)
        agent_dir_idx = 0

        # Randomly generate a memory location
        is_top_goal = jax.random.randint(
            key, minval=0, maxval=2, shape=(1,), dtype=jnp.uint8
        )

        clue_pos = jnp.array((0, 6), dtype=jnp.uint32)
        self.goal_pos = is_top_goal * self.top_pos + (1 - is_top_goal) * self.bottom_pos
        self.distractor_pos = (
            is_top_goal * self.bottom_pos + (1 - is_top_goal) * self.top_pos
        )

        goal_color = (
            is_top_goal * COLOR_TO_INDEX["red"]
            + (1 - is_top_goal) * COLOR_TO_INDEX["green"]
        )

        wall_map = self.wall_map
        maze_map = make_maze_map(
            self.params,
            jnp.array(wall_map, dtype=jnp.bool_),
            self.goal_pos,
            agent_pos,
            agent_dir_idx,
            pad_obs=True,
        )

        red_goal = jnp.array(
            [OBJECT_TO_INDEX["goal"], COLOR_TO_INDEX["red"], 0], dtype=jnp.uint8
        )
        green_goal = jnp.array(
            [OBJECT_TO_INDEX["goal"], COLOR_TO_INDEX["green"], 0], dtype=jnp.uint8
        )
        clue = is_top_goal * red_goal + (1 - is_top_goal) * green_goal

        padding = params.agent_view_size - 1
        wall_map = wall_map.at[clue_pos[1], clue_pos[0]].set(True)
        maze_map = maze_map.at[padding + clue_pos[1], padding + clue_pos[0]].set(clue)

        wall_map = wall_map.at[self.top_pos[1], self.top_pos[0]].set(True)
        maze_map = maze_map.at[
            padding + self.top_pos[1], padding + self.top_pos[0]
        ].set(red_goal)

        wall_map = wall_map.at[self.bottom_pos[1], self.bottom_pos[0]].set(True)
        maze_map = maze_map.at[
            padding + self.bottom_pos[1], padding + self.bottom_pos[0]
        ].set(green_goal)

        state = EnvState(
            agent_pos=agent_pos,
            agent_dir=DIR_TO_VEC[agent_dir_idx],
            agent_dir_idx=agent_dir_idx,
            goal_pos=self.goal_pos,
            wall_map=wall_map,
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        return self.get_obs(state), state

    def get_distractor_pos(self, state):
        goal_x, goal_y = state.goal_pos
        is_top_goal = jnp.logical_and(
            goal_x == self.top_pos[0], goal_y == self.top_pos[1]
        )

        return is_top_goal * self.bottom_pos + (1 - is_top_goal) * self.top_pos

    def step_agent(
        self, key: chex.PRNGKey, state: EnvState, action: int
    ) -> Tuple[EnvState, float]:
        next_state, reward = super().step_agent(key=key, state=state, action=action)

        fwd_pos = jnp.minimum(
            jnp.maximum(
                state.agent_pos + (action == Actions.forward) * state.agent_dir, 0
            ),
            jnp.array(
                (self.params.width - 1, self.params.height - 1), dtype=jnp.uint32
            ),
        )

        distractor_pos = self.get_distractor_pos(state)
        fwd_pos_has_distractor = jnp.logical_and(
            fwd_pos[0] == distractor_pos[0], fwd_pos[1] == distractor_pos[1]
        )

        next_state = next_state.replace(
            terminal=jnp.logical_or(next_state.terminal, fwd_pos_has_distractor)
        )

        return (next_state, reward)


# ======== Registration ========
if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname

register(env_id="Maze-SixteenRooms", entry_point=module_path + ":SixteenRooms")
register(env_id="Maze-SixteenRooms2", entry_point=module_path + ":SixteenRooms2")
register(env_id="Maze-Labyrinth", entry_point=module_path + ":Labyrinth")
register(env_id="Maze-Labyrinth2", entry_point=module_path + ":Labyrinth2")
register(env_id="Maze-LabyrinthFlipped", entry_point=module_path + ":LabyrinthFlipped")
register(env_id="Maze-StandardMaze", entry_point=module_path + ":StandardMaze")
register(env_id="Maze-StandardMaze2", entry_point=module_path + ":StandardMaze2")
register(env_id="Maze-StandardMaze3", entry_point=module_path + ":StandardMaze3")
register(env_id="Maze-SmallCorridor", entry_point=module_path + ":SmallCorridor")
register(env_id="Maze-LargeCorridor", entry_point=module_path + ":LargeCorridor")

register(env_id="Maze-FourRooms", entry_point=module_path + ":FourRooms")
register(env_id="Maze-Crossing", entry_point=module_path + ":Crossing")
register(env_id="Maze-PerfectMaze", entry_point=module_path + ":PerfectMaze")
register(
    env_id="Maze-PerfectMazeMedium", entry_point=module_path + ":PerfectMazeMedium"
)
register(
    env_id="Maze-PerfectMazeXL", entry_point=module_path + ":PerfectMazeExtraLarge"
)

register(env_id="Maze-Memory", entry_point=module_path + ":Memory")
