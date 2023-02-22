# PPO-v3
Adding Dreamer-v3's implementation tricks to CleanRL's PPO

## Get started

```bash
poetry install
poetry run pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu117
```


## Running experiments

For quick and easy experiments, feel free to just run them with the `--track` flag and use wandb's report to visualize them. For slightly more serious experiments, please use the benchmark utility.

The following commands will generate the dry run commands.
```bash
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 Pong-v5 \
    --command "poetry run python ppo-v3/ppo_atari_envpool --track" \
    --num-seeds 3
```
```
autotag feature is enabled
identified git tag: v0.0.1-4-gb17d3b5
local variable 'pr_number' referenced before assignment
======= commands to run:
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Breakout-v5 --seed 1
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Pong-v5 --seed 1
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Breakout-v5 --seed 2
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Pong-v5 --seed 2
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Breakout-v5 --seed 3
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Pong-v5 --seed 3
not running the experiments because --workers is set to 0; just printing the commands to run
```

Once you are comfortable with the commands, you can run them with `--workers 1` to run them in a single machine

```
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 Pong-v5 \
    --command "poetry run python ppo-v3/ppo_atari_envpool --track" \
    --num-seeds 3
    --workers 1
```

>>**Warning** While it is possible to run it with `--workers 6`, it is not recommended. Envpool will likely compete for resources with the other workers and will slow down the experiments.

It is also possible to use slurm. For example, the following command will generate a slurm script and submit it to the slurm queue.

```
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 Pong-v5 \
    --command "poetry run python ppo-v3/ppo_atari_envpool --track" \
    --num-seeds 3
    --workers 1
    --slurm-gpus-per-task 1 \
    --slurm-template-path ppov3.slurm_template
```
```
autotag feature is enabled
identified git tag: v0.0.1-4-gb17d3b5
local variable 'pr_number' referenced before assignment
======= commands to run:
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Breakout-v5 --seed 1
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Pong-v5 --seed 1
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Breakout-v5 --seed 2
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Pong-v5 --seed 2
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Breakout-v5 --seed 3
poetry run python ppo-v3/ppo_atari_envpool --track --env-id Pong-v5 --seed 3
not running the experiments because --workers is set to 0; just printing the commands to run
======= slurm commands to run:
saving command in slurm/5edda234-e8b9-4014-a164-9964e2475847.slurm
running sbatch slurm/5edda234-e8b9-4014-a164-9964e2475847.slurm
sbatch: [info] Determined priority for your job: idle
Submitted batch job 19846
```

The logs will be available in the `slurm/logs` folder.


## `poetry` tips

Poetry locks the dependencies and makes sure that we are using the same versions of the dependencies. It also helps us manage our virtual environments. It is a great tool to use for any python project. For a good usage tutorial, check out [CleanRL's usage guide](https://docs.cleanrl.dev/get-started/basic-usage/)

Additionally, when adding dependencies, it's best to use the `poetry add mypackage` command. When updating dependencies, it's best to use the `poetry update mypackage` command. As a last resort, you can modify `pyproject.toml` directly, and then run `poetry lock --no-update`.

>>**Warning** Do not attempt to run `poetry lock`. It will update all dependencies and will likely take a long time to complete.

