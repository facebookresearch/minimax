"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import time
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import minimax.envs as envs
from minimax.envs.maze.maze import Actions
from minimax.envs.viz.grid_viz import GridVisualizer


def redraw(state, obs, extras):
    if extras["is_ued_maze"]:
        env_instance = extras["env"].get_env_instance(None, state)
        maze_map = extras["render_env"].set_env_instance(env_instance)[1].maze_map
        extras["viz"].render(
            extras["params"], state, highlight=False, maze_map=maze_map
        )
    else:
        extras["viz"].render(extras["params"], state)
        if extras["obs_viz"] is not None:
            extras["obs_viz"].render_grid(
                np.asarray(obs["image"]), k_rot90=3, agent_dir_idx=3
            )


def reset(key, env, extras):
    key, subkey = jax.random.split(extras["rng"])
    obs, state = extras["jit_reset"](subkey)

    extras["rng"] = key
    extras["obs"] = obs
    extras["state"] = state
    extras["n"] += 1

    if not extras["is_ued_maze"]:
        metrics = env.get_env_metrics(state)
        print(metrics)
        extras["n_walls_total"] += metrics["n_walls"]

    if not extras["is_ued_maze"]:
        print(f"mean walls: {extras['n_walls_total']/extras['n']}", flush=True)

    redraw(state, obs, extras)


def step(env, action, extras):
    key, subkey = jax.random.split(extras["rng"])
    obs, state, reward, done, info = env.step_env(subkey, extras["state"], action)
    extras["obs"] = obs
    extras["state"] = state
    # print(f"reward={reward}, agent_dir={obs['agent_dir']}")
    print(f"reward={reward}")

    if done or action == Actions.done:
        key, subkey = jax.random.split(subkey)
        reset(subkey, env, extras)
    else:
        redraw(state, obs, extras)

    extras["rng"] = key


def key_handler(env, extras, event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        extras["jit_reset"]((env, extras))
        return

    if event.key == "left":
        step(env, Actions.left, extras)
        return
    if event.key == "right":
        step(env, Actions.right, extras)
        return
    if event.key == "up":
        step(env, Actions.forward, extras)
        return

    # Spacebar
    if event.key == " ":
        step(env, Actions.toggle, extras)
        return
    if event.key == "a":
        step(env, Actions.pickup, extras)
        return
    if event.key == "d":
        step(env, Actions.drop, extras)
        return

    if event.key == "enter":
        step(env, Actions.done, extras)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="Environment name", default="Maze")
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=90,
    )
    parser.add_argument(
        "--render_agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )
    parser.add_argument(
        "--height",
        default=13,
        type=int,
        help="height",
    )
    parser.add_argument(
        "--width",
        default=13,
        type=int,
        help="width",
    )
    parser.add_argument(
        "--n_walls",
        default=10,
        type=int,
        help="Number of walls",
    )
    parser.add_argument(
        "--agent_view_size",
        default=5,
        type=int,
        help="Number of walls",
    )
    parser.add_argument(
        "--screenshot_path",
        type=str,
        default=None,
        help="maze.png",
    )
    args = parser.parse_args()

    kwargs = dict(
        height=args.height,
        width=args.width,
        n_walls=args.n_walls,
        agent_view_size=args.agent_view_size,
        see_through_walls=True,
        see_agent=True,
        normalize_obs=False,
        sample_n_walls=False,
        replace_wall_pos=False,
        max_episode_steps=250,
    )
    kwargs = {}
    env, params = envs.make(args.env, kwargs)
    params = env.params

    is_ued_maze = False
    render_env = None
    if args.env.startswith("UEDMaze"):
        is_ued_maze = True
        render_env, _ = envs.make("Maze", kwargs)

    viz = GridVisualizer()
    obs_viz = None
    if args.render_agent_view:
        obs_viz = GridVisualizer()

    with jax.disable_jit(False):
        jit_reset = jax.jit(env.reset_env, static_argnums=(1,))
        key = jax.random.PRNGKey(args.seed)
        key, subkey = jax.random.split(key)
        o0, s0 = jit_reset(subkey)
        if is_ued_maze:
            maze_map = render_env.set_env_instance(env.get_env_instance(None, s0))[
                1
            ].maze_map
            viz.render(params, s0, highlight=False, maze_map=maze_map)
        else:
            viz.render(params, s0)
            if obs_viz is not None:
                obs_viz.render_grid(np.asarray(o0["image"]), k_rot90=3, agent_dir_idx=3)

        key, subkey = jax.random.split(key)
        extras = {
            "rng": subkey,
            "state": s0,
            "obs": o0,
            "params": params,
            "viz": viz,
            "obs_viz": obs_viz,
            "jit_reset": jit_reset,
            "n_walls_total": 0,
            "n": 0,
            "env": env,
            "render_env": render_env,
            "is_ued_maze": is_ued_maze,
        }

        if args.screenshot_path is not None:
            print("saving")
            viz.screenshot(args.screenshot_path)

        viz.window.reg_key_handler(partial(key_handler, env, extras))
        viz.show(block=True)
