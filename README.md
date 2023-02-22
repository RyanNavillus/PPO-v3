# PPO-v3
Adding Dreamer-v3's implementation tricks to CleanRL's PPO

## Get started

```bash
poetry install
poetry run pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu117
```


## Running experiments and compare results

For quick and easy experiments, feel free to just run them with the `--track` flag and use wandb's report to visualize them. For slightly more serious experiments, please use the benchmark utility.

```bash

```

## `poetry` tips

Poetry locks the dependencies and makes sure that we are using the same versions of the dependencies. It also helps us manage our virtual environments. It is a great tool to use for any python project. For a good usage tutorial, check out [CleanRL's usage guide](https://docs.cleanrl.dev/get-started/basic-usage/)

Additionally, when adding dependencies, it's best to use the `poetry add mypackage` command. When updating dependencies, it's best to use the `poetry update mypackage` command. As a last resort, you can modify `pyproject.toml` directly, and then run `poetry lock --no-update`.

>>**Warning** Do not attempt to run `poetry lock`. It will update all dependencies and will likely take a long time to complete.

