"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import importlib

from .wrappers import *
from minimax.envs.environment_ued import UEDEnvironment


# Global registry
registered_envs = []

env2entry = {}

env2ued_entry = {}

env2comparator_entry = {}

env2mutator_entry = {}

name2wrapper = {
    "env_wrapper": EnvWrapper,  # for testing,
    "ued_env_wrapper": UEDEnvWrapper,
    "monitor_return": MonitorReturnWrapper,
    "monitor_ep_metrics": MonitorEpisodicMetricsWrapper,
}


def _load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def _fn_for_entry(entry):
    if callable(entry):
        return entry
    else:
        return _load(entry)


def cls_for_env_id(env_id):
    if env_id not in env2entry.keys():
        raise ValueError(f"{env_id} is not registered.")
    else:
        entry = env2entry[env_id]
        return _fn_for_entry(entry)


def _make_env(entry, **env_kwargs):
    return _fn_for_entry(entry)(**env_kwargs)


def _apply_wrappers(env, wrappers):
    base_env = env
    if wrappers is not None and len(wrappers) > 0:
        for name in wrappers:
            wrapper_cls = name2wrapper[name]
            if wrapper_cls.is_compatible(base_env):
                env = wrapper_cls(env)

    return env


def make(
    env_id: str, env_kwargs={}, ued_env_kwargs={}, wrappers=None, ued_wrappers=None
):
    """The minimax equivalent of OpenAI's env.make(env_name)"""
    if env_id not in env2entry.keys():
        raise ValueError(f"{env_id} is not registered.")
    else:
        entry = env2entry[env_id]
        env = _make_env(entry, **env_kwargs)

    if len(ued_env_kwargs) > 0:
        if env_id not in env2ued_entry.keys():
            raise ValueError(f"{env_id} has no UED counterpart registered.")

        _env_kwargs = env_kwargs
        env_kwargs = env.default_params.__dict__
        env_kwargs.update(_env_kwargs)

        ued_entry = env2ued_entry[env_id]
        ued_env_kwargs = _fn_for_entry(ued_entry).align_kwargs(
            ued_env_kwargs, env_kwargs
        )

        ued_env = _make_env(ued_entry, **ued_env_kwargs)

        env = UEDEnvironment(env=env, ued_env=ued_env)

    base_env = env

    env = _apply_wrappers(env, wrappers)

    if isinstance(base_env, UEDEnvironment):
        env = _apply_wrappers(env, ued_wrappers)
        return env, env.env.params, env.ued_env.params

    return env, env.params


def get_comparator(env_id: str, comparator_id: str = "default"):
    entry_point = env2comparator_entry[env_id].get(comparator_id, None)
    assert (
        entry_point is not None
    ), f"Undefined comparator {comparator_id} for environment {env_id}."

    return _fn_for_entry(entry_point)


def get_mutator(env_id: str, mutator_id: str = "default"):
    entry_point = env2mutator_entry[env_id].get(mutator_id, None)
    assert (
        entry_point is not None
    ), f"Undefined mutator {mutator_id} for environment {env_id}."

    return _fn_for_entry(entry_point)


def register(env_id: str, entry_point: str):
    env2entry[env_id] = entry_point


def register_ued(env_id: str, entry_point: str):
    env2ued_entry[env_id] = entry_point


def register_comparator(env_id: str, entry_point: str, comparator_id: str = None):
    if comparator_id is None:
        comparator_id = "default"

    if env_id not in env2comparator_entry:
        env2comparator_entry[env_id] = {}
    env2comparator_entry[env_id][comparator_id] = entry_point


def register_mutator(env_id: str, entry_point: str, mutator_id: str = None):
    if mutator_id is None:
        mutator_id = "default"

    if env_id not in env2mutator_entry:
        env2mutator_entry[env_id] = {}
    env2mutator_entry[env_id][mutator_id] = entry_point
