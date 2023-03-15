timesteps=10000000
env_id="MsPacman-v5"
workers=2

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_none --total-timesteps $timesteps --track" \
    --num-seeds 3 \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_symlog --symlog True --total-timesteps $timesteps --track" \
    --num-seeds 3 \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_twohot --two-hot True --total-timesteps $timesteps --track" \
    --num-seeds 3 \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_percentile --percentile-scale True --total-timesteps $timesteps --track" \
    --num-seeds 3 \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_unimix --unimix 0.01 --total-timesteps $timesteps --track" \
    --num-seeds 3 \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_zero --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds 3 \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all --symlog True --two-hot True --percentile-scale True --unimix 0.01 --critic-zero-init True --total-timesteps $timesteps --track" \
    --num-seeds 3 \
    --workers $workers

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' 'baselines-ppo2-cnn' \
    --filters '?we=dream-team-v3&wpn=PPO-v3&ceik=env_id&cen=exp_name&metric=charts/episodic_return' 'ppo_envpool_tricks_none?tag=v0.0.1-22-g4f9ce4b' 'ppo_envpool_tricks_symlog?tag=v0.0.1-22-g4f9ce4b' 'ppo_envpool_tricks_twohot?tag=v0.0.1-22-g4f9ce4b' 'ppo_envpool_tricks_percentile?tag=v0.0.1-22-g4f9ce4b' 'ppo_envpool_tricks_unimix?tag=v0.0.1-22-g4f9ce4b' 'ppo_envpool_tricks_zero?tag=v0.0.1-22-g4f9ce4b' 'ppo_envpool_tricks_all?tag=v0.0.1-22-g4f9ce4b' \
    --env-ids $env_id \
    --check-empty-runs False \
    --ncols 2 \
    --ncols-legend 2 \
    --output-filename static/cleanrl_vs_baselines \
    --scan-history \
    --report
