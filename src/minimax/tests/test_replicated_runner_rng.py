"""Declared replicas must survive host copies without changing the next update.

Run with XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu.
"""
import jax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
import numpy as np
import pytest
from minimax import agents, models
from minimax.runners.dr_runner import DRRunner
from minimax.runners.paired_runner import PAIREDRunner
from minimax.runners.plr_runner import PLRRunner

def make_runner(kind, n_devices=1):
    model = models.make(env_name='Maze', model_name='default_student_cnn',
                        recurrent_arch='lstm', recurrent_hidden_dim=8,
                        n_conv_filters=4, hidden_dim=8)
    agent = agents.PPOAgent(model=model, n_epochs=1, n_minibatches=1,
                            n_devices=n_devices)
    kwargs = dict(env_name='Maze', env_kwargs=dict(height=5, width=5, n_walls=2),
                  student_agents=[agent], n_students=1 if kind == 'plr' else 2,
                  n_parallel=2*n_devices, n_devices=n_devices,
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
    teacher_agent = agents.PPOAgent(model=teacher, n_epochs=1, n_minibatches=1,
                                   n_devices=n_devices)
    return PAIREDRunner(**kwargs, ued_env_kwargs=dict(n_walls=2),
                        teacher_agents=[teacher_agent])


@pytest.mark.parametrize('kind', ['dr', 'plr', 'paired'])
@pytest.mark.parametrize('n_devices', [1, 2])
def test_replicated_rng_survives_host_roundtrip(kind, n_devices):
    if jax.local_device_count() < n_devices:
        pytest.skip('Set XLA_FLAGS=--xla_force_host_platform_device_count=2')
    runner = make_runner(kind, n_devices)
    run = runner.run
    if n_devices > 1:
        mesh = Mesh(np.asarray(jax.local_devices()[:n_devices]), ('device',))
        inputs, outputs = runner.get_shmap_spec()
        run = jax.jit(shard_map(run, mesh=mesh, in_specs=inputs,
                               out_specs=outputs, check_rep=False))
    initial = runner.reset(jax.random.PRNGKey(7))
    state = run(*initial)[1:]
    assert not np.array_equal(initial[0], state[0])
    if n_devices > 1:
        assert state[0].is_fully_replicated
        replicas = [np.asarray(shard.data) for shard in state[0].addressable_shards]
        assert len(replicas) == n_devices
        for replica in replicas[1:]:
            np.testing.assert_array_equal(replicas[0], replica)
    expected = run(*state)
    restored = jax.tree_util.tree_map(lambda x: jax.device_put(np.asarray(x)), state)
    actual = run(*restored)
    assert not np.array_equal(state[0], actual[1])
    for left, right in zip(jax.tree_util.tree_leaves(expected),
                           jax.tree_util.tree_leaves(actual)):
        np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-7)
