import optuna
import argparse

from cleanrl_utils.tuner import Tuner


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, default="trick_tune",
        help="the name of this optuna study")
    parser.add_argument("--device", type=str, default="cuda",
        help="Choose the device used for optimization. Choose 'cuda', 'cpu', or specify a gpu with 'cuda:0'")
    args = parser.parse_args()

    # fmt: on
    return args


if __name__ == "__main__":

    args = parse_args()

    tuner = Tuner(
        study_name=args.study_name,
        script="switch.py",
        metric="charts/episodic_return",
        metric_last_n_average_window=50,
        direction="maximize",
        aggregation_type="median",
        target_scores={
            "Alien-v5": [227.8, 7127.7],
            # "Amidar-v5": [5.8, 1719.5],
            # "Assault-v5": [222.4, 742],
            # "Asterix-v5": [210, 8503.3],
            # "Asteroids-v5": [719.1, 47388.7],
            # "Atlantis-v5": [12850, 29028.1],
            # "BankHeist-v5": [14.2, 753.1],
            "BattleZone-v5": [2360, 37187.5],
            # "BeamRider-v5": [363.9, 16926.5],
            # "Berzerk-v5": [123.7, 2630.4],
            # "Bowling-v5": [23.1, 160.7],
            # "Boxing-v5": [0.1, 12.1],
            "Breakout-v5": [1.7, 30.5],
            # "Centipede-v5": [2090.9, 12017],
            # "ChopperCommand-v5": [811, 7387.8],
            # "CrazyClimber-v5": [10780.5, 35829.4],
            # "Defender-v5": [2874.5, 18688.9],
            # "DemonAttack-v5": [152.1, 1971],
            # "DoubleDunk-v5": [-18.6, -16.4],
            "Enduro-v5": [0, 860.5],
            # "FishingDerby-v5": [-91.7, -38.7],
            # "Freeway-v5": [0, 29.6],
            # "Frostbite-v5": [65.2, 4334.7],
            # "Gopher-v5": [257.6, 2412.5],
            "Gravitar-v5": [173, 3351.4],
            # "Hero-v5": [1027, 30826.4],
            # "IceHockey-v5": [-11.2, 0.9],
            # "Jamesbond-v5": [29, 302.8],
            # "Kangaroo-v5": [52, 3035],
            # "Krull-v5": [1598, 2665.5],
            "KungFuMaster-v5": [258.5, 22736.3],
            # "MontezumaRevenge-v5": [0, 4753.3],
            # "MsPacman-v5": [307.3, 6951.6],
            # "NameThisGame-v5": [2292.3, 8049],
            # "Phoenix-v5": [761.4, 7242.6],
            # "Pitfall-v5": [-229.4, 6463.7],
            "Pong-v5": [-20.7, 14.6],
            # "PrivateEye-v5": [24.9, 69571.3],
            # "Qbert-v5": [163.9, 13455],
            # "Riverraid-v5": [1338.5, 17118],
            # "RoadRunner-v5": [11.5, 7845],
            # "Robotank-v5": [2.2, 11.9],
            "Seaquest-v5": [68.4, 42054.7],
            # "Skiing-v5": [-17098.1, -4336.9],
            # "Solaris-v5": [1236.3, 12326.7],
            # "SpaceInvaders-v5": [148, 1668.7],
            # "StarGunner-v5": [664, 10250],
            # "Surround-v5": [-10, 6.5],
            "Tennis-v5": [-23.8, -8.3],
            # "TimePilot-v5": [3568, 5229.2],
            # "Tutankham-v5": [11.4, 167.6],
            # "UpNDown-v5": [533.4, 11693.2],
            # "Venture-v5": [0, 1187.5],
            # "VideoPinball-v5": [0, 17667.9],
            # "WizardOfWor-v5": [563.5, 4756.5],
            "YarsRevenge-v5": [3092.9, 54576.9],
            # "Zaxxon-v5": [32.5, 9173.3],
            "AcrobotSwingup-v1": [0, 1000],
            # "AcrobotSwingupSparse-v1": [0, 1000],
            # "BallInCupCatch-v1": [0, 1000],
            # "CartpoleBalance-v1": [0, 1000],
            # "CartpoleBalanceSparse-v1": [0, 1000],
            # "CartpoleSwingup-v1": [0, 1000],
            "CartpoleSwingupSparse-v1": [0, 1000],
            # "CartpoleThreePoles-v1": [0, 1000],
            # "CartpoleTwoPoles-v1": [0, 1000],
            "CheetahRun-v1": [0, 1000],
            # "FingerSpin-v1": [0, 1000],
            # "FingerTurnEasy-v1": [0, 1000],
            # "FingerTurnHard-v1": [0, 1000],
            # "FishSwim-v1": [0, 1000],
            # "FishUpright-v1": [0, 1000],
            "HopperHop-v1": [0, 1000],
            # "HopperStand-v1": [0, 1000],
            # "HumanoidRun-v1": [0, 1000],
            # "HumanoidRunPureState-v1": [0, 1000],
            "HumanoidStand-v1": [0, 1000],
            # "HumanoidWalk-v1": [0, 1000],
            # "HumanoidCMURun-v1": [0, 1000],
            # "HumanoidCMUStand-v1": [0, 1000],
            # "ManipulatorBringBall-v1": [0, 1000],
            # "ManipulatorBringPeg-v1": [0, 1000],
            # "ManipulatorInsertBall-v1": [0, 1000],
            # "ManipulatorInsertPeg-v1": [0, 1000],
            "PendulumSwingup-v1": [0, 1000],
            # "PointMassEasy-v1": [0, 1000],
            # "PointMassHard-v1": [0, 1000],
            "ReacherEasy-v1": [0, 1000],
            "ReacherHard-v1": [0, 1000],
            # "SwimmerSwimmer6-v1": [0, 1000],
            # "SwimmerSwimmer15-v1": [0, 1000],
            "WalkerRun-v1": [0, 1000],
            "WalkerStand-v1": [0, 1000],
            # "WalkerWalk-v1": [0, 1000]
        },
        params_fn=lambda trial: {
            "learning-rate": trial.suggest_float("learning-rate", 0.00025, 0.01, log=True),
            "vf-coef": trial.suggest_float("vf-coef", 0, 2),
            "ent-coef": trial.suggest_float("ent-coef", 0, 0.1),
            "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 2),
            "total-timesteps": 10000000,
            "num-envs": 128,
        },
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(),
        device=args.device,
        wandb_kwargs={
            "project": "ppo-v3",
            "entity": "ryan-colab",
            #"sync_tensorboard": True,
        }
    )
    tuner.tune(
        num_trials=100,
        num_seeds=3,
    )
