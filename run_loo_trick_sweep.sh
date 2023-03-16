timesteps=10000000
env_id="Breakout-v5 MsPacman-v5 Assault-v5 Zaxxon-v5 Tennis-v5"
seeds=2
workers=1
gittag=$(git describe --tags)

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all_nosymlog --two-hot True --percentile-scale True --critic-ema True --unimix 0.01 --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all_notwohot --symlog True --percentile-scale True --critic-ema True --unimix 0.01 --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all_nopercentile --symlog True --two-hot True --critic-ema True --unimix 0.01 --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all_nocriticema --symlog True --two-hot True --percentile-scale True --unimix 0.01 --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all_nounimix --symlog True --two-hot True --percentile-scale True --critic-ema True --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all_nozeroinit --symlog True --two-hot True --percentile-scale True --critic-ema True --unimix 0.01 --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers


poetry run python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' 'baselines-ppo2-cnn' \
    --filters '?we=ryan-colab&wpn=PPO-v3&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
    "ppo_envpool_tricks_nosymlog?tag=${gittag}" \
    "ppo_envpool_tricks_notwohot?tag=${gittag}" \
    "ppo_envpool_tricks_nopercentile?tag=${gittag}" \
    "ppo_envpool_tricks_nocriticema?tag=${gittag}" \
    "ppo_envpool_tricks_nounimix?tag=${gittag}" \
    "ppo_envpool_tricks_nozeroinit?tag=${gittag}" \
    --env-ids $env_id \
    --ncols 2 \
    --ncols-legend 2 \
    --output-filename static/cleanrl_vs_baselines \
    --scan-history \
    --report
