"""A checkpoint must preserve the next curriculum and learner update."""
import pickle
import json
import os
from pathlib import Path
import subprocess
import sys

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


def _snapshot(state):
    return [(str(path), np.asarray(value))
            for path, value in jax.tree_util.tree_flatten_with_path(state)[0]]


def _assert_snapshot(state, expected, *, exact):
    actual = _snapshot(state)
    assert [path for path, _ in actual] == [path for path, _ in expected]
    for (_, value), (_, target) in zip(actual, expected):
        assert value.dtype == target.dtype
        if exact:
            np.testing.assert_array_equal(value, target)
        else:
            np.testing.assert_allclose(value, target, atol=1e-7, rtol=1e-7)


def _checkpoint_worker(mode, kind, folder):
    runner = make_runner(kind)
    payload_path = folder / 'checkpoint.pkl'
    if mode == 'save':
        state = runner.run(*runner.reset(jax.random.PRNGKey(7)))[1:]
        payload = dict(producer_pid=os.getpid(), checkpoint=runner.get_checkpoint_state(state),
                       current=_snapshot(state), next=_snapshot(runner.run(*state)))
        payload_path.write_bytes(pickle.dumps(payload))
    else:
        payload = pickle.loads(payload_path.read_bytes())
        restored = runner.load_checkpoint_state(runner.reset(jax.random.PRNGKey(99)),
                                                 payload['checkpoint'])
        _assert_snapshot(restored, payload['current'], exact=True)
        _assert_snapshot(runner.run(*restored), payload['next'], exact=False)
        (folder / 'result.json').write_text(json.dumps(dict(producer_pid=payload['producer_pid'],
            consumer_pid=os.getpid(), restored_leaves=len(payload['current']),
            next_update_leaves=len(payload['next']))))


@pytest.mark.parametrize('kind', ['dr', 'plr', 'paired'])
def test_checkpoint_continues_after_producer_process_exits(kind, tmp_path):
    # The producer exits before the consumer starts. Static functions and
    # optimizer objects must therefore come from a fresh interpreter and runner.
    for mode in ['save', 'load']:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), mode,
                                 kind, str(tmp_path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'result.json').read_text())
    assert receipt['restored_leaves'] > 0 and receipt['next_update_leaves'] > 0


if __name__ == '__main__':
    _checkpoint_worker(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
