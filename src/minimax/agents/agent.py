"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABC


class Agent(ABC):
    """
    Generic interface for an agent.
    """

    @property
    def is_recurrent(self):
        pass

    @property
    def action_info_keys(self):
        pass

    def init_params(self, rng, obs, carry=None):
        pass

    def init_carry(self, rng, batch_dims):
        pass

    def act(self, *args, **kwargs):
        pass

    def get_action_dist(self, dist_params, dtype):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass
