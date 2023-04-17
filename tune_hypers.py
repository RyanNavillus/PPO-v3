import optuna

from cleanrl_utils.tuner import Tuner

tuner = Tuner(
    script="ppo_v3/ppo_envpool_tricks.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="median",
    target_scores={
        "Breakout-v5": [2, 30],
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_loguniform("learning-rate", 0.00025, 0.01),
        "vf-coef": trial.suggest_uniform("vf-coef", 0, 2),
        "ent-coef": trial.suggest_uniform("ent-coef", 0, 0.1),
        "max-grad-norm": trial.suggest_uniform("max-grad-norm", 0, 2),
        "total-timesteps": 2000000,
        "num-envs": 128,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
tuner.tune(
    num_trials=100,
    num_seeds=3,
)