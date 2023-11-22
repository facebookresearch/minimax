# Command-line usage guide for `minimax.train`

Parsing command-line arguments is handled by [`Parsnip`](parsnip.md). 

You can quickly generate batches of training commands from a JSON configuration file using [`minimax.config.make_cmd`](make_cmd.md).

## General arguments

| Argument                | Description                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `seed`                  | Random seed, should be unique per experimental run                                                   |
| `agent_rl_algo`         | Base RL algorithm used for training (e.g. PPO)                                                       |
| `n_total_updates`       | Total number of updates for the training run                                                         |
| `train_runner`          | Which training runner to use, e.g. `dr`, `plr`, or `paired`                                          |
| `n_devices`             | Number of devices over which to shard the environment batch dimension                                |
| `n_students`            | Number of students in the autocurriculum                                                             |
| `n_parallel`            | Number of parallel environments                                                                      |
| `n_eval`                | Number of parallel trials per environment (environment batch dimension is then `n_parallel*n_eval`) |
| `n_rollout_steps`       | Number of steps per rollout (used for each update cycle)                                             |
| `lr`                    | Learning rate                                                                                        |
| `lr_final`              | Final learning rate, based on linear schedule. Defaults to `None`, corresponding to no schedule.     |
| `lr_anneal_steps`       | Number of steps over which to linearly anneal from `lr` to `lr_final`                                |
| `student_value_coef`    | Value loss coefficient                                                                               |
| `student_entropy_coef`  | Entropy bonus coefficient                                                                            |
| `student_unroll_update` | Unroll multi-gradient updates this many times (can lead to speed ups)                                |
| `max_grad_norm`         | Clip gradients beyond this magnitude                                                                 |
| `adam_eps`              | Value of $`\epsilon`$ numerical stability constant for Adam                                            |
| `discount`              | Discount factor $`\gamma`$ for the student's RL optimization                                           |
| `n_unroll_rollout`      | Unroll rollout scans this many times (can lead to speed ups)                                         |

## Logging arguments

| Argument            | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `verbose`           | Random seed, should be unique per experimental run       |
| `track_env_metrics` | Track per rollout batch environment metrics if `True`    |
| `log_dir`           | Path to directory storing all experiment folders         |
| `xpid`              | Unique name for experiment folder, stored in `--log_dir` |
| `log_interval`      | Log training statistics every this many rollout cycles   |
| `wandb_base_url`    | Base API URL if logging with `wandb`                     |
| `wandb_api_key`     | API key for `wandb`                                      |
| `wandb_entity`      | `wandb` entity associated with the experiment run        |
| `wandb_project`     | `wandb` project for the experiment run                   |
| `wandb_group`       | `wandb` group for the experiment run                     |

## Checkpointing arguments

| Argument               | Description                                                                   |
| ---------------------- | ----------------------------------------------------------------------------- |
| `checkpoint_interval`  | Random seed, should be unique per experimental run                            |
| `from_last_checkpoint` | Begin training from latest `checkpoint.pkl`, if any, in the experiment folder |
| `archive_interval`     | Save an additional checkpoint for models trained per this many rollout cycles |

## Evaluation arguments

| Argument          | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| `test_env_names`  | Random seed, should be unique per experimental run                   |
| `test_n_episodes` | Average test results over this many episodes per test environment    |
| `test_agent_idxs` | Test agents at these indices (csv of indices or `*` for all indices) |

## PPO arguments

These arguments activate when `--agent_rl_algo=ppo`.

| Argument                      | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| `student_ppo_n_epochs`        | Random seed, should be unique per experimental run          |
| `student_ppo_n_epochs`        | Number of PPO epochs per update cycle                       |
| `student_ppo_n_minibatches`   | Number of minibatches per PPO epoch                         |
| `student_ppo_clip_eps`        | Clip coefficient for PPO                                    |
| `student_ppo_clip_value_loss` | Perform value clipping if `True`                            |
| `gae_lambda`                  | Lambda discount factor for Generalized Advantage Estimation |

## PAIRED arguments

The arguments in this section activate when `--train_runner=paired`.

| Argument                  | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| `teacher_lr`              | Learning rate                                                         |
| `teacher_lr_final`        | Anneal learning rate to this value (defaults to `teacher_lr`)         |
| `teacher_lr_anneal_steps` | Number of steps over which to linearly anneal from `lr` to `lr_final` |
| `teacher_discount`        | Discount factor, $`\gamma`$                                             |
| `teacher_value_loss_coef` | Value loss coefficient                                                |
| `teacher_entropy_coef`    | Entropy bonus coefficient                                             |
| `teacher_n_unroll_update` | Unroll multi-gradient updates this many times (can lead to speed ups) |
| `ued_score`               | Name of UED objective, e.g. `relative_regret`                         |

These PPO-specific arguments for teacher optimization further activate when `--agent_rl_algo=ppo`.

| Argument                      | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| `teacher_ppo_n_epochs`        | Number of PPO epochs per update cycle                       |
| `teacher_ppo_n_minibatches`   | Number of minibatches per PPO epoch                         |
| `teacher_ppo_clip_eps`        | Clip coefficient for PPO                                    |
| `teacher_ppo_clip_value_loss` | Perform value clipping if `True`                            |
| `teacher_gae_lambda`          | Lambda discount factor for Generalized Advantage Estimation |

## PLR arguments

The arguments in this section activate when `--train_runner=paired`.

| Argument                      | Description                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `ued_score`                   | Name of UED objective (aka PLR scoring function)                                                              |
| `plr_replay_prob`             | Replay probability                                                                                            |
| `plr_buffer_size`             | Size of level replay buffer                                                                                   |
| `plr_staleness_coef`          | Staleness coefficient                                                                                         |
| `plr_temp`                    | Score distribution temperature                                                                                |
| `plr_use_score_ranks`         | Use rank-based prioritization (rather than proportional)                                                      |
| `plr_min_fill_ratio`          | Only replay once level replay buffer is filled above this ratio                                               |
| `plr_use_robust_plr`          | Use robust PLR (i.e. only update policy on replay levels)                                                     |
| `plr_force_unique`            | Force level replay buffer members to be unique                                                                |
| `plr_use_parallel_eval`       | Use Parallel PLR or Parallel ACCEL (if `plr_mutation_fn` is set)                                              |
| `plr_mutation_fn`             | If set, PLR becomes ACCEL. Use `'default'` for default mutation operator per environment.                     |
| `plr_n_mutations`             | Number of applications of `plr_mutation_fn` per mutation cycle.                                               |
| `plr_mutation_criterion`      | How replay levels are selected for mutation (e.g. `batch`, `easy`, `hard`).                                   |
| `plr_mutation_subsample_size` | Number of replay levels selected for mutation according to the criterion (ignored if using `batch` criterion) |

## Environment-specific arguments

### Maze

See the [`AMaze`](envs/maze.md) docs for details on how to specify [training](envs/maze.md#student-environment), [evaluation](envs/maze.md#student-environment), and [teacher-specific](envs/maze.md#teacher-environment) environment parameters via command line
