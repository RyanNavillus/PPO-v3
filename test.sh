python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python ppo-v3/ppo_atari_envpool --track" \
    --num-seeds 1 \
    --workers 1 \
    --slurm-gpus-per-task 1 \
    --slurm-template-path ppov3.slurm_template
