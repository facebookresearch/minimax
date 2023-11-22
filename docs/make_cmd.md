# Generating commands

The `minimax.config.make_cmd` module enables generating batches of commands from a JSON configuration file, e.g. for running array jobs with Slurm. The JSON should adhere to the following format:
- Each key is a valid command-line argument for `minimax.train`.
- Each value is a list of values for the corresponding command-line argument. Commands are generated for each combination of command-line argument values.
- Boolean values should be specified as 'True' or 'False'.
- If a value is specified as `null`, the associated command-line argument is not included in the generated command (and thus would take on the default value specified when defining the argument parser).

You can try it out by running the following command in your project root directory:

```
python -m minimax.config.make_cmd --config maze/plr
```

The above command will create a directory called `config` in the calling directory with a subdirectory `config/maze` containing configuration files for several autocurriculum methods. 

By default, `minimax.config.make_cmd` searches for configuration files inside `config`. You can create your own JSON configuration files within `config`. If your JSON configuration is located at `config/path/to/my/json`, then you can generate commands with it by calling `minimax.config.make_cmd --config path/to/my/json`.

## Configuring `wandb`

If your configuration includes the argument `wandb_project`, then `minimax.config.make_cmd` will look for a JSON dictionary with your credentials at `config/wandb.json`. The expected format of this JSON file is

```json
{
	"base_url": <URL for wandb API endpoint, e.g. https://api.wandb.ai>,
	"api_key": <Your wandb API key>
}
```