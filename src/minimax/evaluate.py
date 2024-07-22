"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import json
import re
import fnmatch
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as spstats
import jax
import jax.numpy as jnp
from tqdm import tqdm  # type: ignore

from minimax.util.parsnip import Parsnip
from minimax.util.args import str2bool
from minimax.util.checkpoint import load_pkl_object, load_config
from minimax.util.dotdict import DefaultDotDict
from minimax.util.loggers import HumanOutputFormat
from minimax.util.rl import AgentPop
from minimax.runners import EvalRunner
import minimax.models as models
import minimax.agents as agents


parser = Parsnip()

# ==== Define top-level arguments
parser.add_argument("--seed", type=int, default=1, help="Random seed.")
parser.add_argument(
    "--log_dir",
    type=str,
    default="~/logs/minimax",
    help="Log directory containing experiment dirs.",
)
parser.add_argument(
    "--xpid", type=str, default="latest", help="Experiment ID dir name for model."
)
parser.add_argument(
    "--xpid_prefix", type=str, default=None, help="Experiment ID dir name for model."
)
parser.add_argument(
    "--checkpoint_name", type=str, default="checkpoint", help="Name of checkpoint .pkl."
)
parser.add_argument("--env_names", type=str, help="csv of evaluation environments.")
parser.add_argument(
    "--n_episodes", type=int, default=1, help="Number of evaluation episodes."
)
parser.add_argument(
    "--agent_idxs",
    type=str,
    default="*",
    help="Indices of agents to evaluate. '*' indicates all.",
)
parser.add_argument(
    "--render_mode",
    type=str,
    nargs="?",
    const=True,
    default=None,
    help="Visualize episodes.",
)
parser.add_argument("--results_path", type=str, default="results/", help="Results dir.")
parser.add_argument(
    "--results_fname", type=str, default=None, help="Results filename (without .csv)."
)


if __name__ == "__main__":
    """
	Usage: 
		python -m eval \
		--xpid= \
		--env_names="Maze-SixteenRooms" \
		--n_episodes=100 \
		--agent_idxs=0
	"""
    args = parser.parse_args()

    log_dir_path = os.path.expandvars(os.path.expanduser(args.log_dir))

    xpids = []
    if args.xpid_prefix is not None:
        # Get all matching xpid directories
        all_xpids = fnmatch.filter(os.listdir(log_dir_path), f"{args.xpid_prefix}*")
        filter_re = re.compile(".*_[0-9]*$")
        xpids = [x for x in all_xpids if filter_re.match(x)]
    else:
        xpids = [args.xpid]

    pbar = tqdm(total=len(xpids))

    all_eval_stats = defaultdict(list)
    for xpid in xpids:
        xpid_dir_path = os.path.join(log_dir_path, xpid)
        checkpoint_path = os.path.join(xpid_dir_path, f"{args.checkpoint_name}.pkl")
        meta_path = os.path.join(xpid_dir_path, f"meta.json")

        # Load checkpoint info
        if not os.path.exists(meta_path):
            print(f"Configuration at {meta_path} does not exist. Skipping...")
            continue

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint path {checkpoint_path} does not exist. Skipping...")
            continue

        xp_args = load_config(meta_path)

        agent_idxs = args.agent_idxs
        if agent_idxs == "*":
            agent_idxs = np.arange(xp_args.train_runner_args.n_students)
        else:
            agent_idxs = np.array([int(x) for x in agent_idxs.split(",")])
            assert (
                np.max(agent_idxs) <= xp_args.train_runner_args.n_students
            ), "Agent index is out of bounds."

        runner_state = load_pkl_object(checkpoint_path)
        params = runner_state[1]["params"]
        params = jax.tree_util.tree_map(
            lambda x: jnp.take(x, indices=agent_idxs, axis=0), params
        )

        with jax.disable_jit(args.render_mode is not None):
            student_model = models.make(
                env_name=xp_args.env_name,
                model_name=xp_args.student_model_name,
                **xp_args.student_model_args,
            )

            pop = AgentPop(
                agent=agents.PPOAgent(model=student_model), n_agents=len(agent_idxs)
            )

            # We force EvalRunner to select all params, since we've already
            # extracted the relevant agent indices.
            runner = EvalRunner(
                pop=pop,
                env_names=args.env_names,
                env_kwargs=xp_args.eval_env_args,
                n_episodes=args.n_episodes,
                render_mode=args.render_mode,
                agent_idxs="*",
            )

            rng = jax.random.PRNGKey(args.seed)
            _eval_stats = runner.run(rng, params)

            eval_stats = {}
            for k, v in _eval_stats.items():
                prefix_match = re.match(r"^eval/(a[0-9]+):.*", k)
                if prefix_match is not None:
                    prefix = prefix_match.groups()[0]
                    _idx = int(prefix.lstrip("a").rstrip(":"))
                    idx = agent_idxs[_idx]
                    new_prefix = f"a{idx}"
                    new_k = k.replace(prefix, new_prefix)
                    eval_stats[new_k] = v
                else:
                    eval_stats[k] = v

            for k, v in eval_stats.items():
                all_eval_stats[k].append(float(v))

            pbar.update(1)

    pbar.close()

    aggregate_eval_stats = {}
    for k, v in all_eval_stats.items():
        mean = np.mean(all_eval_stats[k])
        if len(all_eval_stats[k]) > 1:
            sem = spstats.sem(all_eval_stats[k])
        else:
            sem = 0.0
        aggregate_eval_stats[k] = f"{mean: 0.2}+/-{sem: 0.2}"

        _min = np.min(all_eval_stats[k])
        aggregate_eval_stats[f"min:{k}"] = f"{_min: 0.2}"

    logger = HumanOutputFormat(sys.stdout)
    logger.writekvs(aggregate_eval_stats)

    if args.results_fname is not None:
        if args.results_fname.strip('"') == "*":
            results_fname = args.xpid_prefix or args.xpid
        else:
            results_fname = args.results_fname

        df = pd.DataFrame.from_dict(all_eval_stats)
        results_path = args.results_path
        if not os.path.isabs(results_path):
            results_path = os.path.join(os.path.abspath(__file__), results_path)
        results_path = os.path.join(results_path, f"{results_fname}.csv")
        df.to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")
