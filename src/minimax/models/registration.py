"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import importlib
import copy

# Global registry
registered_models = {}


def _load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def _get_register_id(env_group_id, model_id):
    return f"{env_group_id.lower()}-{model_id}"


def register(env_group_id, model_id, entry_point):
    register_id = _get_register_id(env_group_id, model_id)
    if register_id in registered_models:
        raise ValueError(f"A model has already been registered as {register_id}.")
    else:
        registered_models[register_id] = entry_point


def make(env_name, model_name=None, **model_kwargs):
    env_group_id = env_name.split("-")[0].lstrip("UED")
    model_id = model_name

    register_id = _get_register_id(env_group_id, model_id)
    if register_id not in registered_models:
        raise ValueError(f"No model for {register_id} found.")
    else:
        entry = registered_models[register_id]

        if callable(entry):
            model = entry(**model_kwargs)
        else:
            model = _load(entry)(**model_kwargs)

        return model
