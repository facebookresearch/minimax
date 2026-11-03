"""Exercise ExperimentRunner through native CPU shard_map and PPO updates.

Run with XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu.
"""
import jax
import numpy as np
import pytest

from minimax.runners.xp_runner import ExperimentRunner
from minimax.util.dotdict import DefaultDotDict


def make_experiment(kind, n_devices):
    runner_kwargs = DefaultDotDict(dict(
        n_students=1 if kind == 'plr' else 2, n_parallel=4, n_eval=1,
        n_rollout_steps=4, lr=1e-4, lr_final=1e-4, lr_anneal_steps=0))
    if kind == 'plr':
        runner_kwargs.update(buffer_size=4, min_fill_ratio=.25,
                             replay_prob=1., ued_score='l1_value_loss')
    model_kwargs = dict(recurrent_arch='lstm', recurrent_hidden_dim=8,
                        n_conv_filters=4, hidden_dim=8)
    return ExperimentRunner(
        train_runner=kind, env_name='Maze', agent_rl_algo='ppo',
        student_model_name='default_student_cnn',
        teacher_model_name='default_teacher_cnn',
        train_runner_kwargs=runner_kwargs,
        env_kwargs=dict(height=5, width=5, n_walls=2),
        ued_env_kwargs=dict(n_walls=2) if kind == 'paired' else {},
        student_rl_kwargs=dict(n_epochs=1, n_minibatches=1),
        teacher_rl_kwargs=dict(n_epochs=1, n_minibatches=1),
        student_model_kwargs=dict(model_kwargs),
        teacher_model_kwargs=dict(model_kwargs), n_devices=n_devices)


@pytest.mark.parametrize('kind', ['dr', 'plr', 'paired'])
@pytest.mark.parametrize('n_devices', [1, 2])
def test_experiment_initializes_and_updates(kind, n_devices):
    if jax.local_device_count() < n_devices:
        pytest.skip('Set XLA_FLAGS=--xla_force_host_platform_device_count=2')
    experiment = make_experiment(kind, n_devices)
    state = experiment.runner.reset(jax.random.PRNGKey(7))
    initial_params = jax.tree_util.tree_map(np.asarray, state[1].params)
    for _ in range(2):
        stats, eval_stats, *state = experiment.step(state)
        jax.block_until_ready(state)
        assert eval_stats == {}
    np.testing.assert_array_equal(state[1].n_updates, 2)
    before = jax.tree_util.tree_leaves(initial_params)
    after = jax.tree_util.tree_leaves(state[1].params)
    assert all(np.isfinite(np.asarray(x)).all() for x in after)
    assert any(not np.array_equal(x, y) for x, y in zip(before, after))
    if n_devices == 2:
        assert any(len(x.addressable_shards) == 2
                   for x in jax.tree_util.tree_leaves(state))
