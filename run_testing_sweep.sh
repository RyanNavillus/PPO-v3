timesteps=10000000
env_id="Breakout-v5"
seeds=1
workers=1
gittag=$(git describe --tags)


# Seed 1
poetry run python -m cleanrl_utils.benchmark \
   --env-ids $env_id \
   --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.98_coef0.1 --critic-ema True --two-hot True  --critic-ema-coef 0.1 --critic-ema-rate 0.98 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
   --env-ids $env_id \
   --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.99_coef1.0 --critic-ema True --two-hot True  --critic-ema-coef 1.0 --critic-ema-rate 0.99 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.98_coef0.5 --critic-ema True --two-hot True  --critic-ema-coef 0.5 --critic-ema-rate 0.98 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.9_coef1.0 --critic-ema True --two-hot True  --critic-ema-coef 1.0 --critic-ema-rate 0.9 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers

# Seed 2
poetry run python -m cleanrl_utils.benchmark \
   --env-ids $env_id \
   --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.98_coef0.1 --critic-ema True --two-hot True  --critic-ema-coef 0.1 --critic-ema-rate 0.98 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers \
    --start-seed 2

poetry run python -m cleanrl_utils.benchmark \
   --env-ids $env_id \
   --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.99_coef1.0 --critic-ema True --two-hot True  --critic-ema-coef 1.0 --critic-ema-rate 0.99 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers \
    --start-seed 2

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.98_coef0.5 --critic-ema True --two-hot True  --critic-ema-coef 0.5 --critic-ema-rate 0.98 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers \
    --start-seed 2

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.9_coef1.0 --critic-ema True --two-hot True  --critic-ema-coef 1.0 --critic-ema-rate 0.9 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers \
    --start-seed 2

# Seed 3
poetry run python -m cleanrl_utils.benchmark \
   --env-ids $env_id \
   --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.98_coef0.1 --critic-ema True --two-hot True  --critic-ema-coef 0.1 --critic-ema-rate 0.98 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers \
    --start-seed 3

poetry run python -m cleanrl_utils.benchmark \
   --env-ids $env_id \
   --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.99_coef1.0 --critic-ema True --two-hot True  --critic-ema-coef 1.0 --critic-ema-rate 0.99 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers \
    --start-seed 3

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.98_coef0.5 --critic-ema True --two-hot True  --critic-ema-coef 0.5 --critic-ema-rate 0.98 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers \
    --start-seed 3

poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_envpool_tricks_criticdebug.py --exp-name ppo_envpool_tricks_critic_ngs_rate0.9_coef1.0 --critic-ema True --two-hot True  --critic-ema-coef 1.0 --critic-ema-rate 0.9 --total-timesteps 10000000 --track" \
    --num-seeds $seeds \
    --workers $workers \
    --start-seed 3