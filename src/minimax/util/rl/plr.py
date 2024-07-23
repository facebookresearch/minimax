"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import struct
import chex
import numpy as np

from .ued_scores import UEDScore


class PLRBuffer(struct.PyTreeNode):
    levels: chex.Array
    scores: chex.Array
    ages: chex.Array
    max_returns: chex.Array  # for MaxMC
    filled: chex.Array
    filled_count: chex.Array
    n_mutations: chex.Array

    ued_score: int = struct.field(
        pytree_node=False, 
        default=UEDScore.L1_VALUE_LOSS.value
    )
    replay_prob: float = struct.field(pytree_node=False, default=0.5)
    buffer_size: int = struct.field(pytree_node=False, default=100)
    staleness_coef: float = struct.field(pytree_node=False, default=0.3)
    temp: float = struct.field(pytree_node=False, default=1.0)
    use_score_ranks: bool = struct.field(pytree_node=False, default=True)
    min_fill_ratio: float = struct.field(pytree_node=False, default=0.5)
    use_robust_plr: bool = struct.field(pytree_node=False, default=False)
    use_parallel_eval: bool = struct.field(pytree_node=False, default=False)


class PLRManager:
    def __init__(
        self,
        example_level,  # Example env instance
        ued_score,
        replay_prob=0.5,
        buffer_size=100,
        staleness_coef=0.3,
        temp=1.0,
        min_fill_ratio=0.5,
        use_score_ranks=True,
        use_robust_plr=False,
        use_parallel_eval=False,
        comparator_fn=None,
        n_devices=1,
    ):

        assert not (
            ued_score == UEDScore.MAX_MC and not use_score_ranks
        ), "Cannot use proportional normalization with MaxMC, which can produce negative scores."

        self.ued_score = ued_score
        self.replay_prob = replay_prob
        self.buffer_size = buffer_size
        self.staleness_coef = staleness_coef
        self.temp = temp
        self.min_fill_ratio = min_fill_ratio
        self.use_score_ranks = use_score_ranks
        self.use_robust_plr = use_robust_plr
        self.use_parallel_eval = use_parallel_eval
        self.comparator_fn = comparator_fn

        self.n_devices = n_devices

        example_level = jax.tree_map(lambda x: jnp.array(x), example_level)
        self.levels = jax.tree_map(
            lambda x: (
                jnp.tile(jnp.zeros_like(x), (buffer_size,) + (1,) * (len(x.shape) - 1))
            ).reshape(buffer_size, *x.shape),
            example_level,
        )

        self.scores = jnp.full(buffer_size, -jnp.inf)
        self.max_returns = jnp.full(buffer_size, -jnp.inf)
        self.ages = jnp.zeros(buffer_size, dtype=jnp.uint32)
        self.filled = jnp.zeros(buffer_size, dtype=jnp.bool_)
        self.filled_count = jnp.zeros((1,), dtype=jnp.int32)
        self.n_mutations = jnp.zeros(buffer_size, dtype=jnp.uint32)

    partial(jax.jit, static_argnums=(0,))

    def reset(self):
        return PLRBuffer(
            ued_score=self.ued_score.value,
            replay_prob=self.replay_prob,
            buffer_size=self.buffer_size,
            staleness_coef=self.staleness_coef,
            temp=self.temp,
            min_fill_ratio=self.min_fill_ratio,
            use_robust_plr=self.use_robust_plr,
            use_parallel_eval=self.use_parallel_eval,
            levels=self.levels,
            scores=self.scores,
            max_returns=self.max_returns,
            ages=self.ages,
            filled=self.filled,
            filled_count=self.filled_count,
            n_mutations=self.n_mutations,
        )

    partial(jax.jit, static_argnums=(0,))

    def _get_replay_dist(self, scores, ages, filled):
        # Score dist
        if self.use_score_ranks:
            sorted_idx = jnp.argsort(-scores)  # Top first
            scores = (
                jnp.zeros(self.buffer_size, dtype=jnp.int32)
                .at[sorted_idx]
                .set(1 / jnp.arange(self.buffer_size))
            )

        scores = scores * filled
        score_dist = scores / self.temp
        z = score_dist.sum()
        z = jnp.where(jnp.equal(z, 0), 1, z)
        score_dist = jax.lax.select(
            jnp.greater(z, 0),
            score_dist / z,
            filled * 1.0,  # Assign equal weight to all present levels
        )

        # Staleness dist
        staleness_scores = ages * filled
        _z = staleness_scores.sum()
        z = jnp.where(jnp.equal(_z, 0), 1, _z)
        staleness_dist = jax.lax.select(
            jnp.greater(_z, 0),
            staleness_scores / z,
            score_dist,  # If no solutions are stale, do not sample from staleness dist
        )

        # Replay dist
        replay_dist = (
            1 - self.staleness_coef
        ) * score_dist + self.staleness_coef * staleness_dist

        return replay_dist

    partial(jax.jit, static_argnums=(0,))

    def _get_next_insert_idx(self, plr_buffer:PLRBuffer):
        return jax.lax.cond(
            jnp.greater(plr_buffer.buffer_size, plr_buffer.filled_count[0]),
            lambda *_: plr_buffer.filled_count[0],
            lambda *_: jnp.argmin(
                self._get_replay_dist(
                    plr_buffer.scores, plr_buffer.ages, plr_buffer.filled
                )
            ),
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def _sample_replay_levels(self, rng, plr_buffer:PLRBuffer, n):
        def _sample_replay_level(carry, step):
            ages = carry
            subrng = step
            replay_dist = self._get_replay_dist(
                plr_buffer.scores, ages, plr_buffer.filled
            )
            replay_idx = jax.random.choice(
                subrng, np.arange(self.buffer_size), shape=(), p=replay_dist
            )
            replay_level = jax.tree_map(
                lambda x: x.take(replay_idx, axis=0), plr_buffer.levels
            )

            ages = ((ages + 1) * (plr_buffer.filled)).at[replay_idx].set(0)

            return ages, (replay_level, replay_idx)

        rng, *subrngs = jax.random.split(rng, n + 1)
        next_ages, (replay_levels, replay_idxs) = jax.lax.scan(
            _sample_replay_level, plr_buffer.ages, jnp.array(subrngs)
        )

        next_plr_buffer = plr_buffer.replace(ages=next_ages)

        return replay_levels, replay_idxs, next_plr_buffer

    def _sample_buffer_uniform(self, rng, plr_buffer:PLRBuffer, n):
        rand_idxs = jax.random.choice(
            rng, np.arange(self.buffer_size), shape=(n,), p=plr_buffer.filled
        )
        levels = jax.tree_map(lambda x: x.take(replay_idx, axis=0), plr_buffer.levels)

        return levels, rand_idxs, plr_buffer

    # Levels must be sampled sequentially, to account for staleness
    partial(jax.jit, static_argnums=(0, 4, 5))

    def sample(self, rng, plr_buffer, new_levels, n, random=False):
        rng, replay_rng, sample_rng = jax.random.split(rng, 3)

        is_replay = jnp.greater(self.replay_prob, jax.random.uniform(replay_rng))
        is_warm = jnp.greater_equal(
            plr_buffer.filled.sum() / self.buffer_size, self.min_fill_ratio
        )

        if self.n_devices > 1:  # Synchronize replay
            is_replay = jax.lax.all_gather(is_replay, axis_name="device")[0]
            is_warm = jnp.all(jax.lax.all_gather(is_warm, axis_name="device"))

        is_replay = jnp.logical_and(is_replay, is_warm)

        if random:
            sample_fn = self._sample_buffer_uniform
        else:
            sample_fn = self._sample_replay_levels

        levels, level_idxs, next_plr_buffer = jax.lax.cond(
            is_replay,
            partial(sample_fn, n=n),
            lambda *_: (new_levels, np.full(n, -1), plr_buffer),
            *(sample_rng, plr_buffer),
        )

        # Update ages when not sampling replay
        next_plr_buffer = jax.lax.cond(
            is_replay,
            lambda *_: next_plr_buffer,
            lambda *_: next_plr_buffer.replace(
                ages=(plr_buffer.ages + n) * (plr_buffer.filled)
            ),
        )

        return levels, level_idxs, is_replay, next_plr_buffer

    @partial(jax.jit, static_argnums=(0,))
    def dedupe_levels(self, plr_buffer, levels, level_idxs):
        if self.comparator_fn is not None and level_idxs.shape[-1] > 2:

            def _check_equal(carry, step):
                match_idxs, other_levels, is_self = carry
                batch_idx, level = step

                matches = jax.vmap(self.comparator_fn, in_axes=(0, None))(
                    other_levels, level
                )

                top2match, top2match_idxs = jax.lax.top_k(matches, 2)

                is_self_dupe = jnp.logical_and(
                    is_self, top2match[1]
                )  # More than 1 match
                is_dedupe_idx = jnp.logical_and(
                    is_self_dupe, jnp.greater(batch_idx, top2match_idxs[0])
                )
                self_match_idx = top2match_idxs[0] * is_dedupe_idx - (~is_dedupe_idx)

                _match_idx = jnp.where(
                    is_self,
                    self_match_idx,  # only first
                    top2match_idxs[0],  # use first matching index in buffer
                )

                match_idxs = jnp.where(
                    matches.any(), match_idxs.at[batch_idx].set(_match_idx), match_idxs
                )

                return (match_idxs, other_levels, is_self), None

            # dedupe among batch levels
            batch_dupe_idxs = jnp.full_like(level_idxs, -1)
            (batch_dupe_idxs, _, _), _ = jax.lax.scan(
                _check_equal,
                (batch_dupe_idxs, levels, True),
                (np.arange(level_idxs.shape[-1]), levels),
            )
            batch_dupe_mask = jnp.greater(batch_dupe_idxs, -1)

            # dedupe against PLR buffer levels
            (level_idxs, _, _), _ = jax.lax.scan(
                _check_equal,
                (level_idxs, plr_buffer.levels, False),
                (np.arange(level_idxs.shape[-1]), levels),
            )

            return level_idxs, batch_dupe_mask
        else:
            return level_idxs, jnp.zeros_like(level_idxs, dtype=jnp.bool_)

    partial(jax.jit, static_argnums=(0, 7))

    def update(
        self,
        plr_buffer,
        levels,
        level_idxs,
        ued_scores,
        dupe_mask=None,
        info=None,
        ignore_val=-jnp.inf,
        parent_idxs=None,
    ):
        # Note: parent_idxs are only used for mutated levels
        done_masks = ued_scores != ignore_val
        if dupe_mask is not None:
            done_masks = jnp.logical_and(
                done_masks, ~dupe_mask
            )  # Ignore duplicate levels in batch by treating them as not done

        cur_n_mutations = plr_buffer.n_mutations
        insert_mask = jnp.zeros((self.buffer_size,), dtype=jnp.bool_)

        def update_level_info(carry, step):
            plr_buffer, insert_mask = carry
            levels = plr_buffer.levels
            scores = plr_buffer.scores
            filled = plr_buffer.filled

            score, level, level_idx, done_mask, parent_idx, max_return = step

            next_insert_idx = self._get_next_insert_idx(plr_buffer)
            is_new_level = jnp.greater(0, level_idx)
            insert_idx = jnp.where(
                is_new_level,
                next_insert_idx,  # new level
                level_idx,
            )

            should_insert = jnp.greater_equal(score, scores.at[insert_idx].get())
            should_insert = jnp.logical_and(should_insert, done_mask)

            is_existing_level = jnp.logical_and(~is_new_level, done_mask)
            should_update = jnp.logical_and(
                is_existing_level, ~insert_mask.at[insert_idx].get()
            )
            should_insert = jnp.logical_and(should_insert, ~should_update)
            next_insert_mask = jnp.where(
                should_insert, insert_mask.at[insert_idx].set(True), insert_mask
            )
            should_insert_or_update = jnp.logical_or(should_insert, should_update)

            # Update max return if needed
            next_max_returns = jnp.where(
                should_insert_or_update,
                plr_buffer.max_returns.at[insert_idx].set(max_return),
                plr_buffer.max_returns,
            )

            updated_level = jax.tree_map(
                lambda x, y: jax.lax.select(should_insert, x, y),
                level,
                jax.tree_map(lambda x: x.at[insert_idx].get(), levels),
            )
            next_levels = jax.tree_map(
                lambda x, y: x.at[insert_idx].set(y), levels, updated_level
            )

            next_scores = jnp.where(
                should_insert_or_update, scores.at[insert_idx].set(score), scores
            )
            next_filled = jnp.where(
                should_insert, filled.at[insert_idx].set(True), filled
            )

            plr_replace_kwargs = dict(
                levels=next_levels,
                scores=next_scores,
                filled=next_filled,
                filled_count=jnp.array([next_filled.sum()]),
                max_returns=next_max_returns,
            )

            # Update mutation count
            n_mutations = plr_buffer.n_mutations
            should_incr_n_mutations = jnp.logical_and(
                jnp.not_equal(parent_idx, -1), should_insert
            )
            should_reset_n_mutations = jnp.logical_and(
                jnp.equal(parent_idx, -1), should_insert_or_update
            )
            reset_n_mutations = jnp.where(
                is_existing_level, cur_n_mutations.at[insert_idx].get(), 0
            )
            next_n_mutations = jnp.where(
                should_incr_n_mutations,
                n_mutations.at[insert_idx].set(
                    cur_n_mutations.at[parent_idx].get() + 1
                ),
                n_mutations,
            )
            next_n_mutations = jnp.where(
                should_reset_n_mutations,
                n_mutations.at[insert_idx].set(reset_n_mutations),
                next_n_mutations,
            )

            plr_replace_kwargs["n_mutations"] = next_n_mutations

            next_plr_buffer = plr_buffer.replace(**plr_replace_kwargs)

            return (next_plr_buffer, next_insert_mask), None

        if parent_idxs is None:
            parent_idxs = jnp.full_like(level_idxs, -1)

        if plr_buffer.ued_score == UEDScore.MAX_MC.value:
            max_returns = info["max_returns"]
        else:
            max_returns = jnp.full_like(level_idxs, -1)
        carry = (ued_scores, levels, level_idxs, done_masks, parent_idxs, max_returns)

        (next_plr_buffer, _), _ = jax.lax.scan(
            update_level_info, (plr_buffer, insert_mask), carry
        )

        return next_plr_buffer

    @partial(jax.jit, static_argnums=(0,))
    def get_metrics(self, plr_buffer):
        replay_dist = self._get_replay_dist(
            plr_buffer.scores, plr_buffer.ages, plr_buffer.filled
        )
        weighted_n_mutations = (plr_buffer.n_mutations * replay_dist).sum()
        scores = jnp.where(plr_buffer.filled, plr_buffer.scores, 0)
        weighted_ued_score = (scores * replay_dist).sum()

        weighted_age = (plr_buffer.ages * replay_dist).sum()

        return dict(
            weighted_n_mutations=weighted_n_mutations,
            weighted_ued_score=weighted_ued_score,
            weighted_age=weighted_age,
        )


class PopPLRManager(PLRManager):
    def __init__(self, *, n_agents, **kwargs):
        super().__init__(**kwargs)

        self.n_agents = n_agents

    @partial(jax.jit, static_argnums=(0, 1))
    def reset(self, n):
        sup = super()
        return jax.vmap(lambda *_: sup.reset())(np.arange(n))

    partial(jax.jit, static_argnums=(0, 4, 5))

    def sample(self, rng, plr_buffer, new_levels, n, random=False):
        sup = super()

        rng, *vrngs = jax.random.split(rng, self.n_agents + 1)

        return jax.vmap(sup.sample, in_axes=(0, 0, 0, None, None))(
            jnp.array(vrngs), plr_buffer, new_levels, n, random
        )

    @partial(jax.jit, static_argnums=(0,))
    def dedupe_levels(self, plr_buffer, levels, level_idxs):
        sup = super()
        return jax.vmap(sup.dedupe_levels)(plr_buffer, levels, level_idxs)

    partial(jax.jit, static_argnums=(0, 7))

    def update(
        self,
        plr_buffer,
        levels,
        level_idxs,
        ued_scores,
        dupe_mask=None,
        info=None,
        ignore_val=-jnp.inf,
        parent_idxs=None,
    ):
        sup = super()
        return jax.vmap(sup.update, in_axes=(0, 0, 0, 0, 0, 0, None, 0))(
            plr_buffer,
            levels,
            level_idxs,
            ued_scores,
            dupe_mask,
            info,
            ignore_val,
            parent_idxs,
        )

    partial(jax.jit, static_argnums=(0,))

    def get_metrics(self, plr_buffer):
        sup = super()
        return jax.vmap(sup.get_metrics)(plr_buffer)
