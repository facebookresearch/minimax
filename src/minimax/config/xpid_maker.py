"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial


def _get_base_role(role):
    return role.removesuffix("_tch").removesuffix("_st")


def _get_runner_info(p):
    n_students = p.get("n_students", 1)
    n_eval = p.get("n_eval", 1)

    n_devices = p.get("n_devices", 1)
    device_info = ""
    if n_devices > 1:
        device_info = f"_d{n_devices}"

    return f"r{n_students}s_{p.n_parallel}p_{n_eval}e_{p.n_rollout_steps}t_ae{p.adam_eps}{device_info}"


def _get_runner_info_dr(p):
    ac_info = _get_runner_info(p)

    reset_info = ""
    if p.ac_reset_env_on_rollout:
        reset_info = f"r"
    if len(reset_info) > 0:
        reset_info = f"_{reset_info}"

    return f"{ac_info}{reset_info}"


def _get_ued_runner_info(p):
    info = _get_runner_info(p)

    if p.ued_score == "relative_regret":
        ued_score = "r"
    elif p.ued_score == "mean_relative_regret":
        ued_score = "mr"
    elif p.ued_score == "population_regret":
        ued_score = "p"
    elif p.ued_score == "neg_return":
        ued_score = "nr"
    elif p.ued_score == "l1_value_loss":
        ued_score = "l1v"
    elif p.ued_score == "positive_value_loss":
        ued_score = "pvl"
    elif p.ued_score == "max_mc":
        ued_score = "mm"
    elif p.ued_score == "value_disagreement":
        ued_score = "vd"
    else:
        raise ValueError(f"Unsupported ued_score {ued_score}")

    info = f"{info}_s{ued_score}"

    return info


def _get_plr_runner_info(p):
    info = _get_ued_runner_info(p)

    plr_info = f"p{p.plr_replay_prob}b{p.plr_buffer_size}t{p.plr_temp}s{p.plr_staleness_coef}m{p.plr_min_fill_ratio}"
    if p.plr_use_score_ranks:
        plr_info = f"{plr_info}r"

    if p.plr_mutation_fn:
        plr_info = f"{plr_info}_m{p.plr_mutation_fn[:3]}{p.plr_n_mutations}{p.plr_mutation_criterion[:3]}"
        if p.plr_mutation_criterion != "batch":
            plr_info = f"{plr_info}{p.plr_mutation_subsample_size}"

    plr_prefix = ""
    if p.plr_use_robust_plr:
        plr_prefix += "r"
    if p.plr_use_parallel_eval:
        plr_prefix += "p"
    if p.plr_force_unique:
        plr_prefix += "f"
    if len(plr_prefix) > 0:
        plr_prefix += "_"

    return f"{plr_prefix}{plr_info}_{info}"


def _get_runner_info_paired(p):
    return _get_ued_runner_info(p)


def _get_env_info_default(p):
    return p.env_name.lower().replace("-", "_")


def _get_env_info_maze(p):
    see_agent = "na" if not p.maze_see_agent else ""

    placement_info = ""
    if p.maze_replace_wall_pos:
        placement_info = f"f"
    if p.maze_sample_n_walls:
        placement_info = f"{placement_info}s"
    if len(placement_info) > 0:
        placement_info = f"_{placement_info}"

    return f"{p.env_name}{p.maze_height}x{p.maze_width}w{p.maze_n_walls}{see_agent}{placement_info}"


def _get_env_info_maze_ued(p):
    see_agent = "na" if not p.maze_see_agent else ""

    info = f"_{see_agent}_ld{p.maze_ued_noise_dim}"

    placement_info = ""
    if p.maze_ued_fixed_n_wall_steps:
        placement_info = f"f"
    if p.maze_ued_replace_wall_pos:
        placement_info = f"{placement_info}r"
    if p.maze_ued_set_agent_dir:
        placement_info = f"{placement_info}d"
    if p.maze_ued_first_wall_pos_sets_budget:
        placement_info = f"{placement_info}b"
    if len(placement_info) > 0:
        placement_info = f"_{placement_info}"
    info = f"{info}{placement_info}"

    return f"{p.env_name}{p.maze_height}x{p.maze_width}w{p.maze_n_walls}{info}"


def _get_model_info_maze_default(p, role):
    model_info = ""
    if f"{role}_recurrent_arch" in p and p[f"{role}_recurrent_arch"] is not None:
        model_info = (
            f"{p[f'{role}_recurrent_arch']}_h{p[f'{role}_recurrent_hidden_dim']}"
        )

        if p[f"{role}_recurrent_arch"] == "s5":
            model_info = f"{model_info}nb{p.get(f'{role}_s5_n_blocks', 1)}nl{p.get(f'{role}_s5_n_layers',4)}"

            activation = p.get(f"{role}_s5_activation")
            if activation == "half_glu1":
                activation = "hg1"
            elif activation == "half_glu2":
                activation = "hg2"
            elif activation == "full_glu":
                activation = "fg"
            elif activation == "gelu":
                activation = "g"
            else:
                activation = "hg1"
            model_info = f"a{activation}_{model_info}"

            ln_key = f"{role}_s5_layernorm_pos"
            ln_info = None
            if ln_key in p:
                ln = p[ln_key]
                if ln == "pre":
                    ln_info = "pr"
                elif ln == "post":
                    ln_info = "po"

            if ln_info is not None:
                model_info = f"l{ln_info}_{model_info}"

    model_info = f"_{model_info}" if len(model_info) > 0 else ""

    value_info = ""
    value_ensemble_key = f"{role}_value_ensemble_size"
    value_ensemble_size = p.get(value_ensemble_key)
    if value_ensemble_size and value_ensemble_size > 1:
        value_info = f"ve{value_ensemble_size}"

    base_activation = p.get(f"{role}_base_activation", "relu")[:2]

    model_info = f"h{p[f'{role}_hidden_dim']}cf{p[f'{role}_n_conv_filters']}fc{p[f'{role}_n_hidden_layers']}se{p[f'{role}_scalar_embed_dim']}ba_{base_activation}{model_info}{value_info}"

    return model_info


def _get_algo_info_ppo(p, role):
    if role == "student":
        lr = str(p.lr)
        if "lr_final" in p:
            lr_final = (
                "" if p.lr_final is None or p.lr_final == p.lr else str(p.lr_final)
            )
            if len(lr_final) > 0:
                lr = f"{lr}_{lr_final}"

        return f"ppo_lr{lr}g{p.discount}cv{p.student_value_loss_coef}ce{p.student_entropy_coef}e{p.student_ppo_n_epochs}mb{p.student_ppo_n_minibatches}l{p.student_gae_lambda}_pc{p.student_ppo_clip_eps}"
    else:
        if "teacher_lr" in p:
            teacher_lr = str(p.lr) if p.teacher_lr is None else str(p.teacher_lr)
        else:
            teacher_lr = str(p.lr)

        if "teacher_lr_final" in p:
            teacher_lr_final = (
                str(p.lr_final)
                if p.teacher_lr_final is None
                else str(p.teacher_lr_final)
            )
        else:
            teacher_lr_final = str(p.lr_final) if "lr_final" in p else ""

        if teacher_lr_final == teacher_lr:
            teacher_lr_final = ""

        if len(teacher_lr_final) > 0:
            teacher_lr = f"{teacher_lr}_{teacher_lr_final}"

        return f"ppo_lr{teacher_lr}g{p.teacher_discount}cv{p.teacher_value_loss_coef}ce{p.teacher_entropy_coef}e{p.teacher_ppo_n_epochs}mb{p.teacher_ppo_n_minibatches}l{p.teacher_gae_lambda}pc{p.teacher_ppo_clip_eps}"


# ============================================================

RUNNER_INFO_HANDLERS = {
    "dr": _get_runner_info_dr,
    "plr": _get_plr_runner_info,
    "paired": _get_runner_info_paired,
}

ENV_INFO_HANDLERS = {
    "maze": _get_env_info_maze,
    "maze_ued": _get_env_info_maze_ued,
}

MODEL_INFO_HANDLERS = {
    "maze": {
        "default_student_cnn": partial(_get_model_info_maze_default, role="student"),
        "default_teacher_cnn": partial(_get_model_info_maze_default, role="teacher"),
    },
}

ALGO_INFO_HANDLERS = {"ppo": _get_algo_info_ppo}


def get_runner_info(p):
    return RUNNER_INFO_HANDLERS[p.get("train_runner", "dr")](p)


def get_env_info(p):
    p.env_name = p.env_name.lower()
    env_name = p.env_name
    if p.train_runner in [
        "paired",
    ]:
        env_name = f"{env_name}_ued"

    return ENV_INFO_HANDLERS.get(env_name, _get_env_info_default)(p)


def get_model_info(p, role="student"):
    model_name = p.get(f"{role}_model_name")
    if model_name is None:
        model_name = p["student_model_name"]
    env_name = p.env_name.lower().split("-")[0]

    return MODEL_INFO_HANDLERS[env_name][model_name](p)


def get_algo_info(p, role="student"):
    return ALGO_INFO_HANDLERS[p.agent_rl_algo](p, role)
