"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import os
import pathlib

import numpy as np

from minimax.util.dotdict import DefaultDotDict
import minimax.config.xpid_maker as xpid_maker

from pprint import pprint

def get_wandb_config():
    wandb_config_path = os.path.join(
        os.path.abspath(os.getcwd()), "config", "wandb.json"
    )
    #print(wandb_config_path)
    if os.path.exists(wandb_config_path):
        with open(wandb_config_path, "r") as config_file:
            config = json.load(config_file)
            #print(config)
            if len(config) == 2:
                return {
                    "wandb_base_url": config["base_url"],
                    "wandb_api_key": config["api_key"],
                }

    return {}


def generate_train_cmds(
    cmd,
    params,
    num_trials=1,
    start_index=0,
    newlines=False,
    xpid_generator=None,
    xpid_prefix="",
    include_wandb_group=False,
    count_set=None,
):
    separator = " \\\n" if newlines else " "

    cmds = []

    if xpid_generator:
        params["xpid"] = xpid_generator(cmd, params, xpid_prefix)
        if include_wandb_group:
            params["wandb_group"] = params["xpid"]

    start_seed = params["seed"]

    for t in range(num_trials):
        params["seed"] = start_seed + t + start_index

        _cmd = [f"python -m {cmd}"]

        trial_idx = t + start_index
        for k, v in params.items():
            if v is None:
                continue

            if k == "xpid":
                v = f"{v}_{trial_idx}"

                assert len(v) < 256, f"{v} exceeds 256 characters!"

                if count_set is not None:
                    count_set.add(v)

            if v == "*":
                v = f'"*"'

            _cmd.append(f"--{k}={v}")

        _cmd = separator.join(_cmd)

        cmds.append(_cmd)

    return cmds


def generate_all_params_for_grid(grid, defaults={}):
    def update_params_with_choices(prev_params, param, choices):
        updated_params = []
        for v in choices:
            for p in prev_params:
                updated = p.copy()
                updated[param] = v
                updated_params.append(updated)

        return updated_params

    all_params = [{}]
    for param, choices in grid.items():
        all_params = update_params_with_choices(all_params, param, choices)

    full_params = []
    for p in all_params:
        d = defaults.copy()
        d.update(p)
        full_params.append(d)

    return full_params


def parse_args():
    parser = argparse.ArgumentParser(description="Make commands")

    parser.add_argument(
        "--dir",
        type=str,
        default="config/configs/",
        help="Path to directory with .json configs",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Name of .json config for hyperparameter search-grid",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="Name of .json config for hyperparameter search-grid",
    )

    parser.add_argument(
        "--start_index", default=0, type=int, help="Starting trial index of xpid runs"
    )

    parser.add_argument(
        "--count",
        action="store_true",
        help="Print number of generated commands at the end of output.",
    )

    parser.add_argument(
        "--checkpoint", action="store_true", help="Whether to start from checkpoint"
    )

    parser.add_argument(
        "--wandb_base_url", type=str, default=None, help="wandb base url"
    )
    parser.add_argument("--wandb_api_key", type=str, default=None, help="wandb api key")
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="wandb project name"
    )

    parser.add_argument(
        "--include_wandb_group",
        action="store_true",
        help="Whether to include wandb group in cmds.",
    )

    return parser.parse_args()


def xpid_from_params(cmd, p, prefix=""):
    p = DefaultDotDict(p)

    env_info = xpid_maker.get_env_info(p)
    runner_info = xpid_maker.get_runner_info(p)
    a_algo_info = xpid_maker.get_algo_info(p, role="student")

    a_info = a_algo_info
    if cmd != "finetune":
        a_model_info = xpid_maker.get_model_info(p, role="student")
        a_info = f"{a_info}_{a_model_info}"
        pt_info = ""
    else:
        pt_agent_info = "tch" if p.get("ft_teacher") else "st"
        pt_info = f"-{p.get('checkpoint_name', 'checkpoint')}_{pt_agent_info}"

    tch_info = ""
    train_runner = p.get("train_runner", "dr")
    if train_runner == "paired":
        tch_algo_info = xpid_maker.get_algo_info(p, role="teacher")
        tch_model_info = xpid_maker.get_model_info(p, role="teacher")
        tch_info = f"_tch_{tch_algo_info}_{tch_model_info}"

    xpid = f"{train_runner}-{env_info}-{runner_info}-{a_info}{tch_info}{pt_info}"

    return xpid


def setup_config_dir():
    config_dir = "config/configs"
    if not os.path.exists(os.path.join(config_dir, "maze")):
        os.makedirs(config_dir, exist_ok=True)

        import shutil

        this_path = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(this_path, "configs")

        for item in os.listdir(src_path):
            src_item = os.path.join(src_path, item)
            dst_item = os.path.join(config_dir, item)

            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, symlinks=True)
            else:
                shutil.copy(src_item, dst_item)


if __name__ == "__main__":
    args = parse_args()

    # Default parameters
    params = {
        # Not needed.
    }

    setup_config_dir()

    json_filename = args.config
    #print(json_filename)
    if not json_filename.endswith(".json"):
        json_filename += ".json"

    grid_path = os.path.join(
        os.path.expandvars(os.path.expanduser(args.dir)), json_filename
    )
    config = json.load(open(grid_path))
    cmd = config.get("cmd", "train")
    grid = config["args"]
    #pprint(grid)
    xpid_prefix = "" if "xpid_prefix" not in config else config["xpid_prefix"]

    if args.checkpoint:
        params["checkpoint"] = True
    #print(grid)
    if "wandb_project" in grid:
        params["wandb_project"] = args.wandb_project

        if args.wandb_base_url:
            params["wandb_base_url"] = args.wandb_base_url
        if args.wandb_api_key:
            params["wandb_api_key"] = args.wandb_api_key
        #pprint(params)
        params.update(get_wandb_config())

    #pprint(params)
    # Generate all parameter combinations within grid, using defaults for fixed params
    all_params = generate_all_params_for_grid(grid, defaults=params)

    #pprint(all_params)
    unique_xpids = None
    if args.count:
        unique_xpids = set()

    # Print all commands
    if cmd == "eval":
        xpid_generator = None
    else:
        xpid_generator = xpid_from_params
    count = 0
    for p in all_params:
        cmds = generate_train_cmds(
            cmd,
            p,
            num_trials=args.n_trials,
            start_index=args.start_index,
            newlines=True,
            xpid_generator=xpid_generator,
            xpid_prefix=xpid_prefix,
            include_wandb_group=args.include_wandb_group,
            count_set=unique_xpids,
        )

        for c in cmds:
            print(c + "\n")
            count += 1

    if args.count:
        print(f"Generated {len(unique_xpids)} unique commands.")
        print("Sweep over")
        grid_sizes = []
        for k, v in grid.items():
            if len(v) > 1:
                grid_sizes.append(len(v))
                print(f"{k}: {len(v)}")

        print(f"Total num settings: {np.prod(grid_sizes)}")
