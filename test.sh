python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python ppo_v3/ppo_atari_envpool.py --track" \
    --num-seeds 1 \
    --workers 1 \
    --slurm-gpus-per-task 1 \
    --slurm-template-path ppov3.slurm_template


## baseline experiments (only if you have the compute resources)
python -m cleanrl_utils.benchmark \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --command "poetry run python ppo_v3/ppo_atari_envpool.py --track" \
    --num-seeds 3 \
    --workers 1 \
    --slurm-gpus-per-task 1 \
    --slurm-template-path ppov3.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids  AcrobotSwingup-v1 CartpoleBalance-v1 CartpoleBalanceSparse-v1 CarpoletSwingup-v1 CartpoleSwingupSparse-v1 CheetahRun-v1 BallInCupCatch-v1 FingerSpin-v1 FingerTurnEasy-v1 FingerTurnHard-v1 HopperHop-v1 HopperStand-v1 PendulumSwingup-v1 ReacherEasy-v1 ReacherHard-v1 WalkerRun-v1 WalkerStand-v1 WalkerWalk-v1 \
    --command "poetry run python ppo_v3/ppo_dmc_envpool.py --track" \
    --num-seeds 3 \
    --workers 1 \
    --slurm-gpus-per-task 1 \
    --slurm-template-path ppov3.slurm_template