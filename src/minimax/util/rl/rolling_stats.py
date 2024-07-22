"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import jax
import jax.numpy as jnp


class RollingStats:
    """
    This class tracks episodic stats, such as final returns
    and env complexity metrics. Works on a per-env basis.
    """

    def __init__(self, names, step_metrics_names=[], window=None):
        self.names = names
        self.step_metric_names = step_metrics_names
        self.window = window

    @partial(jax.jit, static_argnums=(0, 1))
    def reset_stats(self, batch_shape=(1,)):
        stats = {
            "n_episodes": jnp.zeros((*(batch_shape), 1), dtype=jnp.uint32),
            "n_steps": jnp.zeros((*(batch_shape), 1), dtype=jnp.uint32),
        }
        stats.update({name: jnp.zeros((*(batch_shape), 1)) for name in self.names})

        if self.window is not None:
            # Average over window
            stats.update(
                {
                    f"{name}_buffer": jnp.zeros((*(batch_shape), self.window))
                    for name in self.names
                }
            )

        return stats

    @partial(jax.jit, static_argnums=(0,))
    def update_stats(self, stats, done, info, max_episodes=jnp.inf):
        n_eps = stats["n_episodes"]
        n_steps = stats["n_steps"]

        for name in self.names:
            # Update stat
            if name not in info:
                continue

            new_val = info[name]

            # Only record first max_episode episodes
            done = done * (n_eps < max_episodes)
            if name in self.step_metric_names:
                n_incr_prev = n_steps
                n_incr_total = n_steps + 1
                _metric_done = True
            else:
                n_incr_prev = n_eps
                n_incr_total = n_eps + done
                _metric_done = done

            if self.window is None:
                mean = stats[name]
                new_mean = self._update_stat_mean(
                    new_val, mean, n_incr_total, _metric_done
                )
            else:
                buffer_key = f"{name}_buffer"
                buffer = stats[buffer_key]
                new_mean, buffer = self._update_stat_window(
                    new_val, buffer, n_incr_total, n_incr_prev, _metric_done
                )
                stats.update({buffer_key: buffer})

            stats.update(
                {
                    name: new_mean,
                }
            )

            # Only update n_episodes based on real episodes
            if name in self.step_metric_names:
                stats.update({"n_steps": n_incr_total})
            else:
                stats.update({"n_episodes": n_incr_total})

        return stats

    @partial(jax.jit, static_argnums=(0,))
    def _update_stat_mean(self, new_val, mean, n_eps_total, done):
        z = 1 / jnp.maximum(1, n_eps_total)
        new_mean = done * (mean * (1 - z) + new_val * z) + (1 - done) * mean

        return new_mean

    @partial(jax.jit, static_argnums=(0,))
    def _update_stat_window(self, new_val, buffer, n_eps_total, n_eps_prev, done):
        cur_val = buffer[n_eps_prev % self.window]
        new_val = done * new_val + (1 - done) * cur_val
        buffer = buffer.at[n_eps_prev % self.window].set(new_val)
        new_mean = buffer.sum() / jnp.maximum(jnp.minimum(self.window, n_eps_total), 1)

        return new_mean, buffer
