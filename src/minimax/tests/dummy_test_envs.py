"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import OrderedDict
from typing import Tuple, Optional

import jax
from flax import struct
import chex

from envs import environment
from envs import spaces
from envs.registration import register, register_ued


@struct.dataclass
class EnvState:
    time: int = 0
    terminal: bool = False


@struct.dataclass
class EnvParams:
    reward_per_step: int = 1.0
    max_episode_steps: int = 250


class DummyRewardEnv(environment.Environment):
    def __init__(self, reward_per_step=1.0, max_episode_steps=250):
        self.reward_per_step = 1.0

        self.params = EnvParams(
            reward_per_step=reward_per_step, max_episode_steps=max_episode_steps
        )

    @staticmethod
    def align_kwargs(kwargs, other_kwargs):
        return kwargs

    def reset_env(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[chex.ArrayTree, EnvState]:
        state = EnvState(time=0, terminal=False)

        return self.get_obs(state), state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        next_time = state.time + 1
        done = next_time >= self.params.max_episode_steps

        state = state.replace(time=next_time, terminal=done)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            self.params.reward_per_step,
            done,
            {},
        )

    def get_obs(self, state: EnvState) -> chex.ArrayTree:
        return OrderedDict(dict(time=state.time))

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "DummyRewardEnv"

    @property
    def num_actions(self) -> int:
        return len(self.action_set)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(1)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "time": spaces.Discrete(self.params.max_episode_steps),
            }
        )

    def state_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "time": spaces.Discrete(self.params.max_episode_steps),
                "terminal": spaces.Discrete(2),
            }
        )

    def max_episode_steps(self) -> int:
        return self.params.max_episode_steps

    # UED-specific
    def get_env_instance(self, key: chex.PRNGKey, state: EnvState) -> chex.ArrayTree:
        return state

    def set_env_instance(self, encoding: chex.ArrayTree):
        state = encoding
        return self.get_obs(state), state


# Register the env as its own UED env
if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname

register(env_id="DummyRewardEnv", entry_point=module_path + ":DummyRewardEnv")
register_ued(env_id="DummyRewardEnv", entry_point=module_path + ":DummyRewardEnv")
