import time, sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

import minimax.envs as envs
import minimax.models as models
from minimax.agents import PPOAgent
from minimax.util.loggers import HumanOutputFormat
from minimax.runners import PLRRunner, EvalRunner


# Config
SEED = 1
N_ROLLOUT_STEPS = 250
N_PARALLEL = 32
N_UPDATES = 30

LEARNING_RATE = 5e-5
DISCOUNT = 0.995
GAE_LAMBDA = 0.98
ENTROPY_COEF = 0.0

UED_SCORE_FN = 'max_mc'
PLR_REPLAY_PROB = .5
PLR_BUFFER_SIZE = 100
PLR_STALENESS_COEF = 0.3
PLR_TEMP = 0.1

ACCEL_MUTATION_FN = None
ACCEL_N_MUTATIONS = 0

MAZE_HEIGHT = 13
MAZE_WIDTH = 13
MAZE_N_WALLS = 60
MAZE_NORMALIZE_OBS = True
MAZE_REPLACE_WALL_POS = True

# === Uncomment below to activate ACCEL ===
# MAZE_N_WALLS = 0
# LEARNING_RATE = 1e-4
# PLR_REPLAY_PROB = 0.8
# ACCEL_MUTATION_FN = 'default'
# ACCEL_N_MUTATIONS = 20
# ACCEL_MUTATION_SUBSAMPLE_SIZE = 4

USE_PARALLEL_EVAL = False # Set to True to switch to Parallel PLR/ACCEL

N_EVAL_EPISODES = 10
EVAL_INTERVAL = 10
LOG_INTERVAL = 10
PLOT_INTERVAL = 10

EVAL_RENDER_MODE = 'ipython' # Set to true to make final eval rendered


# Set up logger
logger = HumanOutputFormat(sys.stdout)

# Set up plots
ax_names = [
	'Train return',
	'UED score',
	'Shortest path length', 
	'# walls',
	'Solved rate: SixteenRooms', 
	'Solved rate: Labyrinth', 
	'Solved rate: Maze', 
	'SPS']
col_names = [
	'return',
	'plr_weighted_ued_score',
	'env/shortest_path_length',
	'env/n_walls',
	'eval/a0:test_solved_rate:Maze-SixteenRooms', 
	'eval/a0:test_solved_rate:Maze-Labyrinth', 
	'eval/a0:test_solved_rate:Maze-StandardMaze',
	'sps']
col2ax = {k:v for k,v in zip(col_names, ax_names)}
fig, axes = plt.subplots(len(ax_names)//4, 4, figsize=(15, 5))
plt.subplots_adjust(hspace=0.5)

# Make student
model = models.make(
	env_name='Maze', model_name='default_student_cnn', recurrent_arch='lstm')
student = PPOAgent(model=model, n_epochs=5, n_minibatches=1, entropy_coef=ENTROPY_COEF)

# Make runner
env_kwargs = dict(
    height=MAZE_HEIGHT, 
    width=MAZE_WIDTH, 
    n_walls=MAZE_N_WALLS, 
    replace_wall_pos=MAZE_REPLACE_WALL_POS, 
    normalize_obs=MAZE_NORMALIZE_OBS,
    max_episode_steps=N_ROLLOUT_STEPS)
runner = PLRRunner(
	env_name='Maze',
	env_kwargs=env_kwargs,
	student_agents=[student,],
	n_students=1,
	n_parallel=N_PARALLEL,
	n_rollout_steps=N_ROLLOUT_STEPS,
    lr=LEARNING_RATE,
    discount=DISCOUNT,
    gae_lambda=GAE_LAMBDA,
    use_parallel_eval=USE_PARALLEL_EVAL,
	replay_prob=PLR_REPLAY_PROB,
	buffer_size=PLR_BUFFER_SIZE,
	staleness_coef=PLR_STALENESS_COEF,
    temp=PLR_TEMP,
	use_robust_plr=True,
	ued_score=UED_SCORE_FN,
	mutation_fn=ACCEL_MUTATION_FN,
	n_mutations=ACCEL_N_MUTATIONS,
    track_env_metrics=True,)

# Reset runner state
rng = jax.random.PRNGKey(SEED)
rng, subrng = jax.random.split(rng)
runner_state = runner.reset(subrng)

# Make evaluation runner
eval_env_kwargs = dict(normalize_obs=True)
eval_runner_kwargs = dict(
	pop=runner.student_pop,
	env_names="Maze-SixteenRooms,Maze-Labyrinth,Maze-StandardMaze",
	env_kwargs=eval_env_kwargs,
	n_episodes=N_EVAL_EPISODES,
	agent_idxs='*'
)
eval_runner = EvalRunner(**eval_runner_kwargs)

# Train
df = pd.DataFrame(columns=col_names)
train_steps = 0
for i in range(N_UPDATES):
	start = time.time()
	stats, *runner_state = runner.run(*runner_state)
	end = time.time()

	sps = 1/(end-start)*runner.step_batch_size*runner.n_rollout_steps
	stats.update({'steps': train_steps, 'sps': sps})

	if i % EVAL_INTERVAL == 0:
		params = runner_state[1].params
		eval_stats = eval_runner.run(rng, params)
		stats.update(eval_stats)

	df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)

	if i % LOG_INTERVAL == 0:
		stats = {k:v if v != -np.inf else None for k,v in stats.items()}
		logger.writekvs(stats)

	if i % PLOT_INTERVAL == 0:
		for j, col in enumerate(col2ax):
			_df = df[df[col].notna()][['n_updates', col]]
			xs = _df['n_updates']
			ys = _df[col]
			if len(ys) == 0 or xs.max() == 0:
				continue

			ax = axes[j//4, j%4]
			ax.clear()
			ax.set_title(col2ax[col])
			ax.plot(xs, ys)
			ax.set_xlim(0, max(xs.max(), 1))

			if 'solved_rate' in col or 'return' in col:
				ax.set_ylim(0, 1)

		display.display(plt.gcf())
		display.clear_output(wait=True)

logger.writekvs(stats) # Rewrite last stats to stdout, since refreshing plot cleared it

# Final eval
with jax.disable_jit(EVAL_RENDER_MODE is not None):
	eval_runner = EvalRunner(render_mode=EVAL_RENDER_MODE, **eval_runner_kwargs)
	rng = jax.random.PRNGKey(SEED)
	params = runner_state[1].params
	eval_stats = eval_runner.run(rng, params)
	logger.writekvs(eval_stats)