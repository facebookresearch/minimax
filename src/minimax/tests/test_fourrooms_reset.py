"""The player must not start on the target in FourRooms."""
import jax
import numpy as np
import pytest

from minimax.envs.maze.maze_ood import FourRooms


@pytest.mark.parametrize("seed", [0, 681113])
def test_fourrooms_excludes_goal_when_sampling_player(seed):
    env = FourRooms()
    _, state = jax.jit(env.reset)(jax.random.PRNGKey(seed))

    assert not np.array_equal(state.agent_pos, state.goal_pos)
    x, y = np.asarray(state.agent_pos)
    assert not state.wall_map[y, x]
