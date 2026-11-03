"""PPO iterations and optimizer steps are distinct per-agent counters."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from minimax.util.rl.training import VmapTrainState


def make_state():
    return VmapTrainState.create(apply_fn=lambda x: x,
        params={'w': jnp.ones((2, 3))}, tx=optax.adam(1e-3))


def gradient_step(state):
    gradients=jax.tree_util.tree_map(jnp.ones_like,state.params)
    return jax.vmap(lambda s,g:s.apply_gradients(grads=g))(state,gradients)


def test_optimizer_steps_increment_their_own_per_agent_counter():
    state=make_state().replace(n_updates=jnp.array([4,9],dtype=jnp.uint32),
                              n_grad_updates=jnp.array([20,45],dtype=jnp.uint32))
    updated=gradient_step(gradient_step(state))
    np.testing.assert_array_equal(updated.n_grad_updates,[22,47])
    np.testing.assert_array_equal(updated.n_updates,[4,9])
    np.testing.assert_array_equal(updated.n_iters,state.n_iters)


def test_multiple_optimizer_steps_per_ppo_iteration_accumulate():
    state=make_state()
    for iteration in range(3):
        for _ in range(5):
            state=gradient_step(state)
        state=state.increment_updates()
        np.testing.assert_array_equal(state.n_updates,[iteration+1]*2)
        np.testing.assert_array_equal(state.n_grad_updates,[5*(iteration+1)]*2)
        # The actual Adam clock is an independent check on the reported count.
        np.testing.assert_array_equal(state.n_grad_updates,state.opt_state[0].count)
