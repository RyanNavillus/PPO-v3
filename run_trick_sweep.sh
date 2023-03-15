timesteps=10000
env_id="MsPacman-v5 Pong-v5"
seeds=2
workers=2
git-tag="v0.0.1-22-g4f9ce4b"

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_none --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all --symlog True --two-hot True --percentile-scale True --critic-ema --unimix 0.01 --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_symlog --symlog True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_twohot --two-hot True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_percentile --percentile-scale True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_criticema --critic-ema True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_unimix --unimix 0.01 --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_zeroinit --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds $seeds \
    --workers $workers

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' 'baselines-ppo2-cnn' \
    --filters '?we=ryan-colab&wpn=PPO-v3&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
    'ppo_envpool_tricks_none?tag=${git-tag}' \
    'ppo_envpool_tricks_symlog?tag=${git-tag}' \
    'ppo_envpool_tricks_twohot?tag=${git-tag}' \
    'ppo_envpool_tricks_percentile?tag=${git-tag}' \
    'ppo_envpool_tricks_criticema?tag=${git-tag}' \
    'ppo_envpool_tricks_unimix?tag=${git-tag}' \
    'ppo_envpool_tricks_zeroinit?tag=${git-tag}' \
    'ppo_envpool_tricks_all?tag=${git-tag}' \
    --env-ids $env_id \
    --check-empty-runs False \
    --ncols 2 \
    --ncols-legend 2 \
    --output-filename static/cleanrl_vs_baselines \
    --scan-history \
    --report
