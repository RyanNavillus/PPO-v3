timesteps=10000000
env_id="Breakout-v5 MsPacman-v5 Assault-v5 Zaxxon-v5 Tennis-v5"
seeds=3
workers=1
gittag=$(git describe --tags)1

for (( startseed=1; startseed<=$seeds; startseed++ ))
do
poetry run python -m cleanrl_utils.benchmark \
    --env-ids $env_id \
    --command "poetry run python ppo_v3/ppo_atari_envpool.py --exp-name ppo_atari_envpool_machado --total-timesteps $timesteps --track" \
    --num-seeds 1 \
    --start-seed $startseed \
    --workers $workers
done


#poetry run python -m cleanrl_utils.benchmark \
#    --env-ids $env_id \
#    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_none --total-timesteps $timesteps --track" \
#    --num-seeds $seeds \
#    --workers $workers
#
#poetry run python -m cleanrl_utils.benchmark \
#    --env-ids $env_id \
#    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_all --symlog True --two-hot True --percentile-scale True --critic-ema --unimix 0.01 --critic-zero-init True --total-timesteps $timesteps --track" \
#    --num-seeds $seeds \
#    --workers $workers
#
#poetry run python -m cleanrl_utils.benchmark \
#    --env-ids $env_id \
#    --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_symlog --symlog True --total-timesteps $timesteps --track" \
#    --num-seeds $seeds \
#    --workers $workers

# poetry run python -m cleanrl_utils.benchmark \
#     --env-ids $env_id \
#     --command "poetry run python ppo_v3/ppo_envpool_tricks.py --exp-name ppo_envpool_tricks_twohot --two-hot True --total-timesteps $timesteps --track" \
