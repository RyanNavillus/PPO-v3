import sys
import os
import runpy

atari_envs = set(["Alien-v5", "Amidar-v5", "Assault-v5", "Asterix-v5", "Asteroids-v5", "Atlantis-v5", "BankHeist-v5", "BattleZone-v5", "BeamRider-v5", "Berzerk-v5", "Bowling-v5", "Boxing-v5", "Breakout-v5", "Centipede-v5", "ChopperCommand-v5", "CrazyClimber-v5", "Defender-v5", "DemonAttack-v5", "DoubleDunk-v5", "Enduro-v5", "FishingDerby-v5", "Freeway-v5", "Frostbite-v5", "Gopher-v5", "Gravitar-v5", "Hero-v5", "IceHockey-v5", "Jamesbond-v5", "Kangaroo-v5", "Krull-v5", "KungFuMaster-v5", "MontezumaRevenge-v5", "MsPacman-v5", "NameThisGame-v5", "Phoenix-v5", "Pitfall-v5", "Pong-v5", "PrivateEye-v5", "Qbert-v5", "Riverraid-v5", "RoadRunner-v5", "Robotank-v5", "Seaquest-v5", "Skiing-v5", "Solaris-v5", "SpaceInvaders-v5", "StarGunner-v5", "Surround-v5", "Tennis-v5", "TimePilot-v5", "Tutankham-v5", "UpNDown-v5", "Venture-v5", "VideoPinball-v5", "WizardOfWor-v5", "YarsRevenge-v5", "Zaxxon-v5"])
dmc_envs = set(["AcrobotSwingup-v1", "AcrobotSwingupSparse-v1", "BallInCupCatch-v1", "CartpoleBalance-v1", "CartpoleBalanceSparse-v1", "CartpoleSwingup-v1", "CartpoleSwingupSparse-v1", "CartpoleThreePoles-v1", "CartpoleTwoPoles-v1", "CheetahRun-v1", "FingerSpin-v1", "FingerTurnEasy-v1", "FingerTurnHard-v1", "FishSwim-v1", "FishUpright-v1", "HopperHop-v1", "HopperStand-v1", "HumanoidRun-v1", "HumanoidRunPureState-v1", "HumanoidStand-v1", "HumanoidWalk-v1", "HumanoidCMURun-v1", "HumanoidCMUStand-v1", "ManipulatorBringBall-v1", "ManipulatorBringPeg-v1", "ManipulatorInsertBall-v1", "ManipulatorInsertPeg-v1", "PendulumSwingup-v1", "PointMassEasy-v1", "PointMassHard-v1", "ReacherEasy-v1", "ReacherHard-v1", "SwimmerSwimmer6-v1", "SwimmerSwimmer15-v1", "WalkerRun-v1", "WalkerStand-v1", "WalkerWalk-v1"])

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == "__main__":
    # Call the correct script for the given environment
    if len(sys.argv) > 1:
        env_idx = next(i for i, string in enumerate(sys.argv) if "--env-id" in string)
        env_id = sys.argv[env_idx].split("=")[1]
        args = " ".join(sys.argv[1:])

        if env_id in atari_envs:
            #os.system(f"python ppo_v3/ppo_atari_envpool.py {args}")
            experiment = runpy.run_path(path_name="ppo_v3/ppo_envpool_tricks.py", run_name="__main__")
            run_name = experiment["run_name"]
        elif env_id in dmc_envs:
            #os.system(f"python ppo_v3/ppo_dmc_envpool.py {args}")
            experiment = runpy.run_path(path_name="ppo_v3/ppo_envpool_tricks_dmc.py", run_name="__main__")
            run_name = experiment["run_name"]
        else:
            print("Invalid env name")
    else:
        print("No script name provided")
