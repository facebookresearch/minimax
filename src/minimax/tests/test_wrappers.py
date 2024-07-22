"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

import tests.dummy_test_envs as dummy_test_envs
import envs
from envs.environment_ued import UEDEnvironment
from envs.wrappers import EnvWrapper
from envs.wrappers import UEDEnvWrapper


class TestEnvWrapper:
    def setup_class(self):
        # Set up environment + wrapper
        env_kwargs = dict(reward_per_step=1, max_episode_steps=3)
        env_kwargs = env_kwargs
        ued_env_kwargs = env_kwargs
        self.env_kwargs = env_kwargs

        self.env, self.env_params, self.ued_params = envs.make(
            "DummyRewardEnv",
            env_kwargs=env_kwargs,
            ued_env_kwargs=ued_env_kwargs,
            **self._get_wrappers(),
        )

        self.dummy_rng = jax.random.PRNGKey(0)

    @staticmethod
    def _get_wrappers():
        return {"wrappers": ["env_wrapper"], "ued_wrappers": ["ued_env_wrapper"]}


class TestDefaultEnvWrapper(TestEnvWrapper):
    def setup_class(self):
        # Set up environment + wrapper
        env_kwargs = dict(reward_per_step=1, max_episode_steps=2)
        env_kwargs = env_kwargs
        ued_env_kwargs = env_kwargs

        self.env, self.env_params, self.ued_params = envs.make(
            "DummyRewardEnv",
            env_kwargs=env_kwargs,
            ued_env_kwargs=ued_env_kwargs,
            **self._get_wrappers(),
        )

        self.dummy_rng = jax.random.PRNGKey(0)

    def test_base_env(self):
        assert isinstance(self.env.base_env, UEDEnvironment)

    def test_reset_extra(self):
        extra = self.env.reset_extra()
        assert len(extra) == 0

    def test_step(self):
        obs, state, extra = self.env.reset(self.dummy_rng)
        extra = self.env.step(self.dummy_rng, state, 0)[-1]
        assert len(extra) == 0

    def test_reset(self):
        obs, state, extra = self.env.reset(self.dummy_rng)
        assert len(extra) == 0

    def test_reset_env_instance(self):
        _, ued_state = self.env.ued_env.reset(self.dummy_rng)
        instance = self.env.ued_env.get_env_instance(self.dummy_rng, ued_state)
        extra = self.env.set_env_instance(instance)[-1]
        assert len(extra) == 0

    def reset_teacher(self):
        out = self.env.reset_teacher(self.dummy_rng)
        assert len(out) == 2

    def step_teacher(self):
        _, ued_state = self.env.reset_teacher(self.dummy_rng)
        out = self.env.step_teacher(self.dummy_rng, ued_state, 0)
        assert len(out) == 5

    def reset_student(self):
        _, ued_state = self.env.reset_teacher(self.dummy_rng)
        _, state, extra = self.env.reset_student(self.dummy_rng, ued_state)
        assert len(extra) == 0


class TestMonitorReturnWrapper(TestEnvWrapper):
    @staticmethod
    def _get_wrappers():
        return {"wrappers": ["monitor_return"]}

    def test_wrap_level(self):
        assert self.env._wrap_level == 1

    def test_reset_extra(self):
        extra = self.env.reset_extra()
        assert len(extra) == 1 and extra["ep_return"] == 0

    def test_get_monitored_metrics(self):
        metrics = self.env.get_monitored_metrics()
        assert len(metrics) == 1 and "return" in metrics

    def test_reset(self):
        _, _, extra = self.env.reset(self.dummy_rng)
        assert len(extra) == 1 and extra["ep_return"] == 0

    def test_step(self):
        obs, state, extra = self.env.reset(self.dummy_rng)

        n_steps = 2
        return_ = 0
        for i in range(n_steps):
            _, state, r, _, _, extra = self.env.step(
                self.dummy_rng, state, 0, extra=extra
            )
            return_ += r

        assert extra["ep_return"] == self.env_kwargs["reward_per_step"] * n_steps

        # Finish the episode
        _, state, r, _, info, extra = self.env.step(
            self.dummy_rng, state, 0, extra=extra
        )

        assert extra["ep_return"] == 0
        assert info["return"] == self.env_kwargs["reward_per_step"] * (n_steps + 1)
