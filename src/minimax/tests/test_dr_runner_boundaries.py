"""A new level must not inherit recurrent memory from the previous rollout."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minimax import agents, models
from minimax.runners.dr_runner import DRRunner


@pytest.mark.parametrize('architecture', ['lstm', None])
def test_new_levels_ignore_incoming_recurrent_state(architecture):
    model = models.make(env_name='Maze', model_name='default_student_cnn',
                        recurrent_arch=architecture, recurrent_hidden_dim=8,
                        n_conv_filters=4, hidden_dim=8)
    agent = agents.PPOAgent(model=model, n_epochs=1, n_minibatches=1)
    runner = DRRunner('Maze', {}, [agent], n_students=2, n_parallel=2,
                      n_eval=2, n_rollout_steps=4)
    initial = runner.reset(jax.random.PRNGKey(7))
    # First produce nonzero memory through an actual rollout, then reuse the same
    # RNG and complete learner state for both next-rollout calls.
    after_first = runner.run(*initial)[1:]
    with_stale = list(after_first)
    with_zero = list(after_first)
    with_zero[5] = jax.tree_util.tree_map(jnp.zeros_like, after_first[5])
    if architecture:
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree_util.tree_leaves(after_first[5]))
    stale_result = runner.run(*with_stale)
    zero_result = runner.run(*with_zero)
    for a, b in zip(jax.tree_util.tree_leaves(stale_result),
                    jax.tree_util.tree_leaves(zero_result)):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-7)
