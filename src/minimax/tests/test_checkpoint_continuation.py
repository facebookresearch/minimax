"""A checkpoint must preserve the next curriculum and learner update."""
import pickle

import jax
import numpy as np
import pytest

from minimax import agents, models
from minimax.runners.dr_runner import DRRunner
from minimax.runners.paired_runner import PAIREDRunner
from minimax.runners.plr_runner import PLRRunner


def make_runner(kind):
    model = models.make(env_name='Maze', model_name='default_student_cnn',
                        recurrent_arch='lstm', recurrent_hidden_dim=8,
                        n_conv_filters=4, hidden_dim=8)
    agent = agents.PPOAgent(model=model, n_epochs=1, n_minibatches=1)
    kwargs = dict(env_name='Maze', env_kwargs=dict(height=5, width=5, n_walls=2),
                  student_agents=[agent], n_students=1 if kind == 'plr' else 2, n_parallel=2,
                  n_eval=1, n_rollout_steps=4)
    if kind == 'dr':
        return DRRunner(**kwargs)
    if kind == 'plr':
        return PLRRunner(**kwargs, buffer_size=4, min_fill_ratio=.25,
                         replay_prob=1., ued_score='l1_value_loss')
    teacher = models.make(env_name='Maze', model_name='default_teacher_cnn',
                          output_dim=25, recurrent_arch='lstm', recurrent_hidden_dim=8,
                          n_conv_filters=4, hidden_dim=8, n_scalar_embeddings=5,
                          max_scalar=4)
    teacher_agent = agents.PPOAgent(model=teacher, n_epochs=1, n_minibatches=1)
    return PAIREDRunner(**kwargs, ued_env_kwargs=dict(n_walls=2),
                        teacher_agents=[teacher_agent])


def assert_dynamic_state_equal(a, b, *, exact):
    left = jax.tree_util.tree_flatten_with_path(a)[0]
    right = jax.tree_util.tree_flatten_with_path(b)[0]
    assert [str(p) for p, _ in left] == [str(p) for p, _ in right]
    for (_, x), (_, y) in zip(left, right):
        assert np.asarray(x).dtype == np.asarray(y).dtype
        if exact:
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('kind', ['dr', 'plr', 'paired'])
def test_checkpoint_restores_complete_state_and_next_update(kind):
    runner = make_runner(kind)
    state = runner.run(*runner.reset(jax.random.PRNGKey(7)))[1:]
    checkpoint = pickle.loads(pickle.dumps(runner.get_checkpoint_state(state)))
    resumed = make_runner(kind)
    template = resumed.reset(jax.random.PRNGKey(99))
    restored = resumed.load_checkpoint_state(template, checkpoint)
    assert_dynamic_state_equal(state, restored, exact=True)
    assert_dynamic_state_equal(runner.run(*state), resumed.run(*restored), exact=False)


def test_legacy_plr_checkpoint_without_buffer_cannot_resume():
    runner = make_runner('plr')
    state = runner.reset(jax.random.PRNGKey(7))
    checkpoint = runner.get_checkpoint_state(state)
    checkpoint[1].pop('plr_buffer', None)
    with pytest.raises(ValueError, match='missing the PLR replay buffer'):
        runner.load_checkpoint_state(state, checkpoint)


def test_legacy_non_plr_checkpoint_remains_complete():
    runner = make_runner('dr')
    state = runner.reset(jax.random.PRNGKey(7))
    checkpoint = runner.get_checkpoint_state(state)
    checkpoint[1].pop('plr_buffer', None)
    restored = runner.load_checkpoint_state(runner.reset(jax.random.PRNGKey(99)), checkpoint)
    assert_dynamic_state_equal(state, restored, exact=True)
