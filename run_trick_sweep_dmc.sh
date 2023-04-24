timesteps=100000000
env_id="AcrobotSwingup-v1 AcrobotSwingupSparse-v1 BallInCupCatch-v1 CartpoleBalance-v1 CartpoleBalanceSparse-v1 CartpoleSwingup-v1 CartpoleSwingupSparse-v1 CartpoleThreePoles-v1 CartpoleTwoPoles-v1 CheetahRun-v1 FingerSpin-v1 FingerTurnEasy-v1 FingerTurnHard-v1 FishSwim-v1 FishUpright-v1 HopperHop-v1 HopperStand-v1 HumanoidRun-v1 HumanoidRunPureState-v1 HumanoidStand-v1 HumanoidWalk-v1 HumanoidCMURun-v1 HumanoidCMUStand-v1 ManipulatorBringBall-v1 ManipulatorBringPeg-v1 ManipulatorInsertBall-v1 ManipulatorInsertPeg-v1 PendulumSwingup-v1 PointMassEasy-v1 PointMassHard-v1 ReacherEasy-v1 ReacherHard-v1 SwimmerSwimmer6-v1 SwimmerSwimmer15-v1 WalkerRun-v1 WalkerStand-v1 WalkerWalk-v1"
#env_id="AcrobotSwingup-v1 CartpoleBalance-v1 FingerSpin-v1 HopperHop-v1 ManipulatorInsertBall-v1 WalkerWalk-v1"
seeds=3
workers=1000
gittag=$(git describe --tags)

for (( startseed=3; startseed<=$seeds; startseed++ ))
do
    poetry run python -m cleanrl_utils.benchmark \
        --env-ids $env_id \
        --command "singularity exec --nv --bind /fs/nexus-scratch/rsulli /fs/nexus-scratch/rsulli/ppov3.simg python ppo_v3/ppo_envpool_tricks_dmc.py --exp-name ppo_envpool_tricks_dmc_all --num-envs 128  --symlog True --two-hot True --percentile-scale True --critic-ema True --unimix 0.01 --critic-zero-init True --total-timesteps $timesteps --track" \
        --start-seed $startseed \
        --num-seeds 1 \
        --workers $workers \
        --slurm-gpus-per-task 1 \
        --slurm-template-path ppov3.slurm_template

    poetry run python -m cleanrl_utils.benchmark \
       --env-ids $env_id \
       --command "singularity exec --nv --bind /fs/nexus-scratch/rsulli /fs/nexus-scratch/rsulli/ppov3.simg python ppo_v3/ppo_envpool_tricks_dmc.py --exp-name ppo_envpool_tricks_dmc_symlog --num-envs 128  --symlog True --total-timesteps $timesteps --track" \
       	--start-seed $startseed \
       	--num-seeds 1 \
       	--workers $workers \
        --slurm-gpus-per-task 1 \
        --slurm-template-path ppov3.slurm_template

    poetry run python -m cleanrl_utils.benchmark \
        --env-ids $env_id \
        --command "singularity exec --nv --bind /fs/nexus-scratch/rsulli /fs/nexus-scratch/rsulli/ppov3.simg python ppo_v3/ppo_envpool_tricks_dmc.py --exp-name ppo_envpool_tricks_dmc_twohot --num-envs 128  --two-hot True --total-timesteps $timesteps --track" \
        --start-seed $startseed \
        --num-seeds 1 \
        --workers $workers \
        --slurm-gpus-per-task 1 \
        --slurm-template-path ppov3.slurm_template

    poetry run python -m cleanrl_utils.benchmark \
        --env-ids $env_id \
        --command "singularity exec --nv --bind /fs/nexus-scratch/rsulli /fs/nexus-scratch/rsulli/ppov3.simg python ppo_v3/ppo_envpool_tricks_dmc.py --exp-name ppo_envpool_tricks_dmc_percentile --num-envs 128  --percentile-scale True --total-timesteps $timesteps --track" \
        --start-seed $startseed \
        --num-seeds 1 \
        --workers $workers \
        --slurm-gpus-per-task 1 \
        --slurm-template-path ppov3.slurm_template

    poetry run python -m cleanrl_utils.benchmark \
        --env-ids $env_id \
        --command "singularity exec --nv --bind /fs/nexus-scratch/rsulli /fs/nexus-scratch/rsulli/ppov3.simg python ppo_v3/ppo_envpool_tricks_dmc.py --exp-name ppo_envpool_tricks_dmc_criticema --num-envs 128  --critic-ema True --total-timesteps $timesteps --track" \
        --start-seed $startseed \
        --num-seeds 1 \
        --workers $workers \
        --slurm-gpus-per-task 1 \
        --slurm-template-path ppov3.slurm_template

    poetry run python -m cleanrl_utils.benchmark \
        --env-ids $env_id \
        --command "singularity exec --nv --bind /fs/nexus-scratch/rsulli /fs/nexus-scratch/rsulli/ppov3.simg python ppo_v3/ppo_envpool_tricks_dmc.py --exp-name ppo_envpool_tricks_dmc_unimix --num-envs 128  --unimix 0.01 --total-timesteps $timesteps --track" \
        --start-seed $startseed \
        --num-seeds 1 \
        --workers $workers \
        --slurm-gpus-per-task 1 \
        --slurm-template-path ppov3.slurm_template

    poetry run python -m cleanrl_utils.benchmark \
        --env-ids $env_id \
        --command "singularity exec --nv --bind /fs/nexus-scratch/rsulli /fs/nexus-scratch/rsulli/ppov3.simg python ppo_v3/ppo_envpool_tricks_dmc.py --exp-name ppo_envpool_tricks_dmc_zeroinit --num-envs 128  --critic-zero-init True --total-timesteps $timesteps --track" \
        --start-seed $startseed \
        --num-seeds 1 \
        --workers $workers \
        --slurm-gpus-per-task 1 \
        --slurm-template-path ppov3.slurm_template

    poetry run python -m cleanrl_utils.benchmark \
        --env-ids $env_id \
        --command "singularity exec --nv --bind /fs/nexus-scratch/rsulli /fs/nexus-scratch/rsulli/ppov3.simg python ppo_v3/ppo_envpool_tricks_dmc.py --exp-name ppo_envpool_tricks_dmc_none --num-envs 128 --total-timesteps $timesteps --track" \
        --start-seed $startseed \
        --num-seeds 1 \
        --workers $workers \
        --slurm-gpus-per-task 1 \
        --slurm-template-path ppov3.slurm_template
done

