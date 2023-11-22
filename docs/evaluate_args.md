# Command-line usage guide for `minimax.evaluate`

You can evaluate student agent checkpoints using `minimax.evaluate` as follows:

```bash
python -m minimax.evaluate \
--seed 1 \
--log_dir <absolute path log directory> \
--xpid_prefix <select checkpoints with xpids matching this prefix> \
--env_names <csv string of test environment names> \
--n_episodes <number of trials per test environment> \
--results_path <path to results folder> \
--results_fname <filename of output results csv>
```

Some behaviors of `minimax.evaluate` to be aware of:
- This command will search `log_dir` for all experiment directories with names matching `xpid_prefix` and evaluate the checkpoint named `<checkpoint_name>.pkl`. 
- `minimax.evaluate` assumes xpid values end with a unique index, so that they match the regex `.*_[0-9]+$`.
- The results will be averaged over all such checkpoints (at most one checkpoint per matching experiment folder). Using the `--xpid_prefix` argument can be useful for evaluating corresponding to the same experimental configuration with different training seeds (and thus share an xpid prefix, e.g. <xpid_prefix_0>, <xpid_prefix_1>, <xpid_prefix_2>).

If you would like to evaluate a checkpoint for only a single experiment, specify the full experiment directory name using `--xpid` instead of using `--xpid_prefix`.


## All command-line arguments
| Argument          | Description                                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `seed`            | Random seed for evaluation                                                                                                       |
| `log_dir`         | Directory containing experiment folders                                                                                          |
| `xpid`            | Name of experiment folder, i.e. the experiment ID                                                                                |
| `xpid_prefix`     | Evaluate and average results over checkpoints for experiments with experiment IDs matching this prefix (ignores `--xpid` if set) |
| `checkpoint_name` | Name of checkpoint to evaluate (in each matching experiment folder)                                                              |
| `env_names`       | Number of devices over which to shard the environment batch dimension                                                            |
| `n_episodes`      | Number of students in the autocurriculum                                                                                         |
| `agent_idxs`      | Indices of student agents to evaluate (csv of indices or `*` for all indices)                                                    |
| `results_path`    | Number of parallel environments                                                                                                  |
| `results_fname`   | Number of parallel trials per environment (environment)                                                                         |
| `render_mode`     | If set, renders the evaluation episode. Requires disabling JIT. Use `'ipython'` if rendering inside an IPython notebook.         |
