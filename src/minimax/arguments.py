"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse

from minimax.util.parsnip import Parsnip
from minimax.util.args import str2bool


parser = Parsnip()


# ==== Define top-level arguments
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='Training seed.')
parser.add_argument(
    '--agent_rl_algo',
    type=str,
    default='ppo',
    choices=['ppo'],
    help='Base RL algorithm to use.')
parser.add_argument(
    '--n_total_updates',
    type=int,
    default=30000,
    help='Total number of student gradient updates.')
parser.add_argument(
    '--train_runner',
    type=str,
    default='dr',
    choices=['dr', 'plr', 'paired'],
    help='Algorithm runner.')
parser.add_argument(
    '--n_devices',
    type=int,
    default=1,
    help='Number of devices.')


# ==== RL runner arguments.
train_runner_subparser = parser.add_subparser(
    name='train_runner')
train_runner_subparser.add_argument(
    '--n_students', 
    type=int, 
    default=1, 
    help='Number of students in population.')
train_runner_subparser.add_argument(
    '--n_parallel', 
    type=int, 
    default=1, 
    help='Number of parallel environments per rollout.')
train_runner_subparser.add_argument(
    '--n_eval', 
    type=int, 
    default=1, 
    help='Number of student evaluations per environment.')
train_runner_subparser.add_argument(
    '--n_rollout_steps', 
    type=int, 
    default=250, 
    help='Number of rollout steps.')
train_runner_subparser.add_argument(
    '--lr', 
    type=float, 
    default=1e-4, 
    help='Initial learning rate.')
train_runner_subparser.add_argument(
    '--lr_final', 
    type=float, 
    default=None,
    nargs="?", 
    help='Final learning rate.')
train_runner_subparser.add_argument(
    '--lr_anneal_steps', 
    type=int, 
    default=0,
    nargs="?", 
    help='Number of learning rate annealing steps.')
train_runner_subparser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max norm of gradients.')
train_runner_subparser.add_argument(
    '--adam_eps',
    type=float,
    default=1e-5,
    help='Adam eps.')
train_runner_subparser.add_argument(
    '--track_env_metrics', 
    type=str2bool,
    default=False,
    help='Track env metrics during training. Can reduce SPS.')
train_runner_subparser.add_argument(
    '--discount',
    type=float,
    default=0.995,
    help='Student discount factor for rewards')
train_runner_subparser.add_argument(
    '--n_unroll_rollout',
    type=int,
    default=1,
    help='Number of times to unroll rollout scan.')
train_runner_subparser.add_argument(
    '--render',
    type=str2bool,
    default=False,
    help='Whether to render.')

# ------ AC-specific arguments -----
dr_subparser = parser.add_subparser(
    name='dr',
    prefix='dr',
    dependency={'train_runner': 'dr'},
    dest='train_runner')

# -------- General UED arguments --------
parser.add_dependent_argument(
    '--ued_score',
    type=str,
    default='relative_regret',
    dependency={'train_runner': ['plr', 'paired']},
    dest='train_runner',
    choices=[
        'relative_regret',
        'mean_relative_regret',
        'population_regret', 
        'neg_return', # aka minimax adversarial
        'l1_value_loss', 
        'positive_value_loss',
        'max_mc',
        'value_disagreement'
    ],
    help='UED score of agent.')

# -------- PAIRED arguments --------
plr_subparser = parser.add_subparser(
    name='plr',
    prefix='plr',
    dependency={'train_runner': 'plr'},
    dest='train_runner')
plr_subparser.add_argument(
    '--replay_prob',
    type=float,
    default=0.5,
    help='PLR replay probability.'
)
plr_subparser.add_argument(
    '--buffer_size',
    type=int,
    default=128,
    help='PLR level buffer size.'
)
plr_subparser.add_argument(
    '--staleness_coef',
    type=float,
    default=0.3,
    help='Staleness coefficient.'
)
plr_subparser.add_argument(
    '--temp',
    type=float,
    default=1.0,
    help='Score distribution temperature.'
)
plr_subparser.add_argument(
    '--use_score_ranks',
    type=str2bool,
    default=True,
    help='Use rank-based prioritiziation.'
)
plr_subparser.add_argument(
    '--min_fill_ratio',
    type=float,
    default=0.5,
    help='Minimum fill ratio before level replay begins.'
)
plr_subparser.add_argument(
    '--use_robust_plr',
    type=str2bool,
    default=True,
    help='Use robust PLR.'
)
plr_subparser.add_argument(
    '--use_parallel_eval',
    type=str2bool,
    default=False,
    help='Use rank-based prioritiziation.'
)
plr_subparser.add_argument(
    '--force_unique',
    type=str2bool,
    default=False,
    help='Force level buffer members to be unique.'
)
plr_subparser.add_argument(
    '--mutation_fn',
    type=str,
    default=None,
    help='Name of mutation function for ACCEL.'
)
plr_subparser.add_argument(
    '--n_mutations',
    type=int,
    default=0,
    help='Number of mutations per iteration of ACCEL.'
)
plr_subparser.add_argument(
    '--mutation_criterion',
    type=str,
    default='batch',
    help='Criterion for choosing PLR buffer members to mutate.'
)
plr_subparser.add_argument(
    '--mutation_subsample_size',
    type=int,
    default=0,
    help='Number of PLR buffer members to mutate into a full batch.'
)


# -------- PAIRED arguments --------
paired_subparser = parser.add_subparser(
    name='paired',
    prefix='paired',
    dependency={'train_runner': 'paired'},
    dest='train_runner')


# ==== Student RL arguments.
student_rl_subparser = parser.add_subparser(
    name='student_rl',
    prefix='student')
student_rl_subparser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.0,
    help='entropy term coefficient')
student_rl_subparser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
student_rl_subparser.add_argument(
    '--n_unroll_update',
    type=int,
    default=1,
    help='Number of times to unroll minibatch scan.')

# -------- Student PPO arguments. --------
student_ppo_subparser = parser.add_subparser(
    name='student_ppo',
    prefix='student_ppo',
    dest='student_rl',
    dependency={'agent_rl_algo': 'ppo'})
student_ppo_subparser.add_argument(
    '--n_epochs',
    type=int,
    default=5,
    help='Number of PPO epochs.')
student_ppo_subparser.add_argument(
    '--n_minibatches',
    type=int,
    default=1,
    help='Number of minibatches per PPO epoch.')
student_ppo_subparser.add_argument(
    '--clip_eps',
    type=float,
    default=0.2,
    help='PPO clip parameter')
student_ppo_subparser.add_argument(
    '--clip_value_loss',
    type=str2bool,
    default=True,
    help='ppo clip value loss')
parser.add_dependent_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    prefix='student',
    dependency={'agent_rl_algo': 'ppo'},
    dest='train_runner',
    help='GAE lambda parameter for student.')


# ==== Teacher RL arguments.
teacher_rl_subparser = parser.add_subparser(
    name='teacher_rl',
    prefix='teacher',
    dependency={'train_runner':['paired']})
teacher_rl_subparser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.0,
    help='entropy term coefficient')
teacher_rl_subparser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
teacher_rl_subparser.add_argument(
    '--n_unroll_update',
    type=int,
    default=1,
    help='Number of times to unroll minibatch scan.')
parser.add_dependent_argument(
    '--teacher_discount',
    type=float,
    default=0.995,
    dependency={'train_runner': 'paired'},
    dest='train_runner',
    help='discount factor for rewards')
parser.add_dependent_argument(
    '--teacher_lr',
    type=float, 
    default=None,
    nargs="?",
    dependency={'agent_rl_algo': 'ppo', 'train_runner': 'paired'},
    dest='train_runner',
    help='Initial learning rate of teacher.')
parser.add_dependent_argument(
    '--teacher_lr_final',
    type=float,
    default=None,
    nargs="?",
    dependency={'agent_rl_algo': 'ppo', 'train_runner': 'paired'},
    dest='train_runner',
    help='Initial learning rate of teacher.')
parser.add_dependent_argument(
    '--teacher_lr_anneal_steps',
    type=int,
    default=0,
    nargs="?",
    dependency={'agent_rl_algo': 'ppo', 'train_runner': 'paired'},
    dest='train_runner',
    help='Initial learning rate of teacher.')


# -------- Teacher PPO arguments. --------
teacher_ppo_subparser = parser.add_subparser(
    name='teacher_ppo',
    prefix='teacher_ppo',
    dest='teacher_rl',
    dependency={'agent_rl_algo': 'ppo', 'train_runner': 'paired'})
teacher_ppo_subparser.add_argument(
    '--n_epochs',
    type=int,
    default=5,
    help='Number of PPO epochs.')
teacher_ppo_subparser.add_argument(
    '--n_minibatches',
    type=int,
    default=1,
    help='Number of minibatches per PPO epoch.')
teacher_ppo_subparser.add_argument(
    '--clip_eps',
    type=float,
    default=0.2,
    help='PPO clip parameter')
teacher_ppo_subparser.add_argument(
    '--clip_value_loss',
    type=str2bool,
    default=True,
    help='ppo clip value loss')
parser.add_dependent_argument(
    '--teacher_gae_lambda',
    type=float,
    default=0.95,
    dependency={'agent_rl_algo': 'ppo', 'train_runner': 'paired'},
    dest='train_runner',
    help='GAE lambda parameter for teacher.')


# ==== Student model arguments.
parser.add_argument(
    '--student_model_name',
    type=str,
    default='default_student_cnn',
    help='Name of student model architecture.')

# Placeholder group for student model args
student_model_parser = parser.add_subparser(
    name='student_model',
    prefix='student')

# ---- Maze args for student model ----
student_maze_model_parser = parser.add_subparser(
    name='student_maze_model',
    prefix='student',
    dest="student_model",
    dependency={'env_name': 'Maze*'})
student_maze_model_parser.add_argument(
    '--recurrent_arch',
    type=str,
    default=None,
    nargs='?',
    choices=['gru', 'lstm', 's5'],
    help='Student RNN architecture.')
student_maze_model_parser.add_argument(
    '--recurrent_hidden_dim',
    type=int,
    default=0,
    help='Student recurrent hidden state size.')
student_maze_model_parser.add_argument(
    '--hidden_dim',
    type=int,
    default=32,
    help='Student hidden dimension.')
student_maze_model_parser.add_argument(
    '--n_hidden_layers',
    type=int,
    default=1,
    help='Student number of hidden layers in policy/value heads.')
student_maze_model_parser.add_argument(
    '--n_conv_filters',
    type=int,
    default=16,
    help='Number of CNN filters for student.')
student_maze_model_parser.add_argument(
    '--n_scalar_embeddings',
    type=int,
    default=4,
    help='Defaults to 4 directional embeddings.')
student_maze_model_parser.add_argument(
    '--scalar_embed_dim',
    type=int,
    default=5,
    help='Dimensionality of scalar direction embeddings.')
student_maze_model_parser.add_argument(
    '--base_activation',
    type=str,
    default='relu',
    choices=['relu', 'gelu', 'crelu', 'leaky_relu'],
    help='Nonlinearity for intermediate layers.')
student_maze_model_parser.add_argument(
    '--value_ensemble_size',
    type=int,
    default=1,
    help='Size of value ensemble. Defaults to 1 (no ensemble).')
student_maze_model_parser.add_argument(
    '--s5_n_blocks',
    type=int,
    default=1,
    help='Number of S5 blocks.')
student_maze_model_parser.add_argument(
    '--s5_n_layers',
    type=int,
    default=4,
    help='Number of S5 encoder layers.')
student_maze_model_parser.add_argument(
    '--s5_layernorm_pos',
    type=str,
    default=None,
    help='Layernorm pos in S5.')
student_maze_model_parser.add_argument(
    '--s5_activation',
    type=str,
    default="half_glu1",
    choices=["half_glu1", "half_glu2", "full_glu", "gelu"],
    help='Number of S5 encoder layers.')


# ==== Teacher model arguments.
parser.add_dependent_argument(
    '--teacher_model_name',
    dependency={'train_runner': ['paired']},
    type=str,
    help='Name of teacher model architecture.'
)

# Placeholder group for teacher model args
teacher_model_parser = parser.add_subparser(
    name='teacher_model',
    prefix='teacher',
    dependency={'train_runner': ['paired']})

# ---- Maze args for PAIRED teacher model ----
teacher_maze_model_parser = parser.add_subparser(
    name='teacher_maze_model',
    prefix='teacher',
    dest="teacher_model",
    dependency={'train_runner': 'paired', 'env_name': 'Maze*'})
teacher_maze_model_parser.add_argument(
    '--recurrent_arch',
    type=str,
    default=None,
    nargs='?',
    choices=['gru', 'lstm', 's5'],
    help='Teacher RNN architecture.')
teacher_maze_model_parser.add_argument(
    '--recurrent_hidden_dim',
    type=int,
    default=0,
    help='Teacher recurrent hidden state size.')
teacher_maze_model_parser.add_argument(
    '--hidden_dim',
    type=int,
    default=32,
    help='Teacher hidden dimension.')
teacher_maze_model_parser.add_argument(
    '--n_hidden_layers',
    type=int,
    default=1,
    help='Teacher number of hidden layers in policy/value heads.')
teacher_maze_model_parser.add_argument(
    '--n_conv_filters',
    type=int,
    default=128,
    help='Number of CNN filters for teacher.')
teacher_maze_model_parser.add_argument(
    '--scalar_embed_dim',
    type=int,
    default=10,
    help='Dimensionality of time-step embeddings.')
teacher_maze_model_parser.add_argument(
    '--base_activation',
    type=str,
    default='relu',
    choices=['relu', 'gelu', 'crelu', 'leaky_relu'],
    help='Nonlinearity for intermediate layers.')
teacher_maze_model_parser.add_argument(
    '--s5_n_blocks',
    type=int,
    default=1,
    help='Number of S5 blocks.')
teacher_maze_model_parser.add_argument(
    '--s5_n_layers',
    type=int,
    default=4,
    help='Number of S5 encoder layers.')
teacher_maze_model_parser.add_argument(
    '--s5_layernorm_pos',
    type=str,
    default=None,
    help='Layernorm pos in S5.')
teacher_maze_model_parser.add_argument(
    '--s5_activation',
    type=str,
    default="half_glu1",
    choices=["half_glu1", "half_glu2", "full_glu", "gelu"],
    help='Number of S5 encoder layers.')


# ==== Environment arguments.
parser.add_argument(
    '--env_name',
    type=str,
    default='Maze',
    help='Environment to train on')
env_parser = parser.add_subparser(
    name='env')

# -------- UED environment arguments. --------
ued_env_parser = parser.add_subparser(
    name='ued_env')

# ======== Envoronment-specific subparsers ======== 
# -------- Maze --------
env_maze_parser = parser.add_subparser(
    name='maze',
    prefix='maze',
    dependency={'env_name': ['Maze', 'Maze-MemoryMaze']},
    dest='env')
env_maze_parser.add_argument(
    '--height',
    type=int,
    default=13,
    help='Height of training mazes.')
env_maze_parser.add_argument(
    '--width',
    type=int,
    default=13,
    help='Width of training mazes.')
env_maze_parser.add_argument(
    '--n_walls',
    type=int,
    default=25,
    help='Maximum number of walls in training mazes.')
env_maze_parser.add_argument(
    '--replace_wall_pos',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help='Sample wall positions with replacement.')
env_maze_parser.add_argument(
    '--sample_n_walls',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help='Uniformly sample n_walls between 0 and n_walls.')
# -------- Maze* environments --------
env_maze_all_parser = parser.add_subparser(
    name='maze_all',
    prefix='maze',
    dependency={'env_name': 'Maze*'},
    dest='env')
env_maze_all_parser.add_argument(
    '--see_agent',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=True,
    help='Whether the agent sees itself in observations.')
env_maze_all_parser.add_argument(
    '--normalize_obs',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=True,
    help='Ensure observations are between 0 and 1.')
env_maze_all_parser.add_argument(
    '--obs_agent_pos',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help='Include agent xy pos in observations.')
env_maze_all_parser.add_argument(
    '--max_episode_steps',
    type=int,
    default=250,
    help='Maximum number of steps in training episodes.')

# -------- Maze UED --------
maze_ued_parser = parser.add_subparser(
    name='maze_ued',
    prefix='maze_ued',
    dependency={'env_name': ['Maze', 'Maze-MemoryMaze'], 'train_runner': 'paired'},
    dest='ued_env')
maze_ued_parser.add_argument(
    '--replace_wall_pos',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help='Teacher can sample same wall pos multiple times (resulting in variable n_walls).')
maze_ued_parser.add_argument(
    '--fixed_n_wall_steps',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help='Teacher samples exactly n_walls wall positions for each level.')
maze_ued_parser.add_argument(
    '--first_wall_pos_sets_budget',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help='The first wall positional index determines the wall budget.')
maze_ued_parser.add_argument(
    '--noise_dim',
    type=int,  
    default=50,
    help="Dimension of episodic noise vector injected into the teacher's observation.")
maze_ued_parser.add_argument(
    '--n_walls',
    type=int,  
    default=25,
    help="Number walls the adversary can place.")
maze_ued_parser.add_argument(
    '--set_agent_dir',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help='Teacher chooses the agent direction on last time step.')
maze_ued_parser.add_argument(
    '--normalize_obs',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=True,
    help='Normalize teacher observations.')


# Logging arguments (All top-level arguments.).
parser.add_argument(
    "--verbose", 
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help="Print progress to stdout.")
parser.add_argument(
    '--xpid',
    default='latest',
    help='name for the run - prefix to log files')
parser.add_argument(
    '--log_dir',
    default='~/logs/minimax/',
    help='directory to save agent logs')
parser.add_argument(
    '--log_interval',
    type=int,
    default=1,
    help='log interval, one log per n updates')
parser.add_argument(
    "--from_last_checkpoint", 
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help="Begin training from latest checkpoint if available.")
parser.add_argument(
    "--checkpoint_interval", 
    type=int, 
    default=0,
    help="Save model every this many updates.")
parser.add_argument(
    "--archive_interval", 
    type=int, 
    default=0,
    help="Save an archived model every this many updates.")
parser.add_argument(
    "--archive_init_checkpoint", 
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=False,
    help="Archive the initial checkpoint.")
parser.add_argument(
    '--test_interval',
    type=int,
    default=10,
    help='Evaluate on test envs every this many updates.')


# Evaluation args.
eval_parser = parser.add_subparser(
    name='eval',
    prefix='test')
eval_parser.add_argument(
    '--n_episodes',
    type=int,
    default=10,
    help='Number of test episodes per environment')
eval_parser.add_argument(
    '--env_names',
    type=str,
    default=None,
    help='Test environments to evaluate on.')
eval_parser.add_argument(
    '--agent_idxs',
    type=str,
    default='*',
    help="csv of agents to evaluate. '*' indicates all.")
eval_env_parser = parser.add_subparser(
    name='eval_env',
    prefix='test_env',
)

# -------- Maze eval arguments. --------
maze_eval_parser = parser.add_subparser(
    name='maze_eval',
    prefix='maze_test',
    dependency={'env_name': 'Maze*'},
    dest='eval_env'
)
maze_eval_parser.add_argument(
    "--see_agent", 
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=True,
    help="Maze observations include the agent.")
maze_eval_parser.add_argument(
    '--normalize_obs',
    type=str2bool, 
    nargs='?', 
    const=True, 
    default=True,
    help='Ensures observations are between 0 and 1.')


# -------- wandb arguments. --------
wandb_parser = parser.add_subparser(
    name='wandb',
    prefix='wandb')
wandb_parser.add_argument(
    "--base_url",
    type=str,
    default="https://api.wandb.ai",
    help='wandb base url'
)
wandb_parser.add_argument(
    "--api_key",
    type=str,
    default=None,
    help='wandb api key'    
)
wandb_parser.add_argument(
    "--entity",
    type=str,
    default='minqi',
    help='Team name'
)
wandb_parser.add_argument(
    "--project",
    type=str,
    default='paired',
    help='wandb project name for logging'
)
wandb_parser.add_argument(
    "--group",
    type=str,
    default=None,
    help='wandb group name for logging'
)
