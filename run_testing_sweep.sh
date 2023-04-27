timesteps=10000000
env_id="Breakout-v5"
seeds=1
workers=1
gittag=$(git describe --tags)


poetry run python -m cleanrl_utils.benchmark \
   --env-ids "YarsRevenge-v5"\
   --command "poetry run python ppo_v3/ppo_atari_envpool_resnet_200M.py --exp-name ppo_atari_envpool_resnet_200M --num-envs 128 --device cuda:0 --total-timesteps 50000000 --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
   --env-ids "Battlezone-v5"\
   --command "poetry run python ppo_v3/ppo_atari_envpool_resnet_200M.py --exp-name ppo_atari_envpool_resnet_200M --num-envs 128 --device cuda:1 --total-timesteps 50000000 --track" \
    --num-seeds $seeds \
    --workers $workers

poetry run python -m cleanrl_utils.benchmark \
   --env-ids "KungFuMaster-v5"\
   --command "poetry run python ppo_v3/ppo_atari_envpool_resnet_200M.py --exp-name ppo_atari_envpool_resnet_200M --num-envs 128 --device cuda:3 --total-timesteps 50000000 --track" \
    --num-seeds $seeds \
    --workers $workers

