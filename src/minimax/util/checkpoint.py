"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import shutil
import pickle
import json
from pathlib import Path

from .dotdict import DefaultDotDict


def save_pkl_object(obj, path):
    """Helper to store pickle objects."""
    output_file = Path(path)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Stored checkpoint at {path}.")


def load_pkl_object(path: str):
    """Helper to reload pickle objects."""
    with open(path, "rb") as input:
        obj = pickle.load(input)
    print(f"Loaded checkpoint from {path}.")
    return obj


def safe_checkpoint(state, dir_path, name, index=None, archive_interval=None):
    savename = f"{name}.pkl"
    tmp_savepath = f"{name}_tmp.pkl"

    save_path = os.path.join(dir_path, savename)
    tmp_savepath = os.path.join(dir_path, tmp_savepath)

    save_pkl_object(state, tmp_savepath)

    # Rename
    os.replace(tmp_savepath, save_path)

    # Archive if needed
    if index is not None and archive_interval is not None and archive_interval > 0:
        if index % archive_interval == 0:
            archive_path = os.path.join(dir_path, f"{name}_{index}.pkl")
            shutil.copy(save_path, archive_path)


def load_config(path: str):
    with open(path) as meta_file:
        _config = json.load(meta_file)["config"]

    config = {}
    for k, v in _config.items():
        if isinstance(v, dict):
            v = DefaultDotDict(v)
        config[k] = v

    return DefaultDotDict(config)
