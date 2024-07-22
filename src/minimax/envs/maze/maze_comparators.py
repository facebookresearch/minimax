"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import jax
import jax.numpy as jnp

from minimax.envs.registration import register_comparator


@jax.jit
def is_equal_map(a, b):
    agent_pos_eq = jnp.equal(a.agent_pos, b.agent_pos).all()
    goal_pos_eq = jnp.equal(a.goal_pos, b.goal_pos).all()
    wall_map_eq = jnp.equal(a.wall_map, b.wall_map).all()

    _eq = jnp.logical_and(agent_pos_eq, goal_pos_eq)
    _eq = jnp.logical_and(_eq, wall_map_eq)

    return _eq


# Register the mutators
if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname

register_comparator(
    env_id="Maze", comparator_id=None, entry_point=module_path + ":is_equal_map"
)
