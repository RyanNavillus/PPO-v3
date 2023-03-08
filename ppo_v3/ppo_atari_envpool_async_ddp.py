import argparse
import itertools
import os
import random
import time
import uuid
from distutils.util import strtobool

import envpool
import gym
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-v3",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="dream-team-v3",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Breakout-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--local-num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--async-batch-size", type=int, default=4,
        help="the envpool's batch size in the async mode")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    parser.add_argument("--compile", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Whether to use `torch.compile` (only available in PyTorch 2.0+)")
    parser.add_argument("--backend", type=str, default="nccl", choices=["gloo", "nccl", "mpi"],
        help="the backend to use for distributed training")
    args = parser.parse_args()
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    args.async_update = int(args.local_num_envs / args.async_batch_size)
    # fmt: on
    return args


ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping


def make_env(env_id, seed, num_envs, async_batch_size=None):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            batch_size=async_batch_size,
            episodic_life=False,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 6
            repeat_action_probability=0.25,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12
            noop_max=1,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12 (no-op is deprecated in favor of sticky action, right?)
            full_action_space=True,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) Tab. 5
            max_episode_steps=ATARI_MAX_FRAMES,  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.network(x)
    
class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
    
    def forward(self, x):
        return self.actor(x)
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = layer_init(nn.Linear(512, 1), std=1)
    
    def forward(self, x):
        return self.critic(x)

def get_value(network, critic, x):
    return critic(network(x / 255.0))

def get_action_and_value(network, actor, critic, x):
    hidden = network(x / 255.0)
    logits = actor(hidden)
    probs = Categorical(logits=logits)
    action = probs.sample()
    return action, probs.log_prob(action), critic(hidden)

def get_action_and_value2(network, actor, critic, x, action):
    hidden = network(x / 255.0)
    logits = actor(hidden)
    probs = Categorical(logits=logits)
    return action, probs.log_prob(action), probs.entropy(), critic(hidden)


if __name__ == "__main__":
    args = parse_args()
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["RANK"])
    args.num_envs = args.local_num_envs * args.world_size
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    dist.init_process_group(args.backend)
    torch.cuda.set_device(args.local_rank)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"
    if args.local_rank == 0:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(args.env_id, args.seed, args.local_num_envs, args.async_batch_size)()
    envs.async_reset()
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    network = Network().to(device)
    actor = Actor(envs).to(device)
    critic = Critic().to(device)
    network = DDP(network)
    actor = DDP(actor)
    critic = DDP(critic)

    if args.compile:
        network = torch.compile(network)
        actor = torch.compile(actor)
        critic = torch.compile(critic)
        get_action_and_value = torch.compile(get_action_and_value)
        get_action_and_value2 = torch.compile(get_action_and_value2)
        get_value = torch.compile(get_value)

    agent_params = list(network.parameters()) + \
        list(actor.parameters()) + \
        list(critic.parameters())
    for p in agent_params:
        dist.broadcast(p.data, src=0)
    optimizer = optim.Adam(agent_params, lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps * args.async_update, args.async_batch_size) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps * args.async_update, args.async_batch_size) + envs.single_action_space.shape).to(device)
    env_ids = np.zeros((args.num_steps * args.async_update, args.async_batch_size), dtype=np.int32)
    logprobs = torch.zeros((args.num_steps * args.async_update, args.async_batch_size)).to(device)
    rewards = torch.zeros((args.num_steps * args.async_update, args.async_batch_size)).to(device)
    dones = torch.zeros((args.num_steps * args.async_update, args.async_batch_size)).to(device)
    values = torch.zeros((args.num_steps * args.async_update, args.async_batch_size)).to(device)

    lastobs = torch.zeros((args.local_num_envs,) + envs.single_observation_space.shape).to(device)
    lastdones = torch.zeros((args.local_num_envs,)).to(device)
    lastrewards = torch.zeros((args.local_num_envs,)).to(device)
    lastenvids = np.zeros((args.local_num_envs,), dtype=np.int32)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.local_batch_size
    next_obs, next_reward, next_done, info = envs.recv()
    next_obs = torch.Tensor(next_obs).to(device)
    next_reward = torch.Tensor(next_reward).to(device).view(-1)
    next_done = torch.Tensor(next_done).to(device)
    lastobs[info["env_id"]] = next_obs
    lastdones[info["env_id"]] = next_done
    lastrewards[info["env_id"]] = next_reward

    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)

    for update in range(1, num_updates + 1):
        update_time_start = time.time()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps * args.async_update):
            # next_obs, next_reward, next_done, info = envs.recv()
            # next_obs = torch.Tensor(next_obs).to(device)
            global_step += args.async_batch_size * args.world_size
            obs[step] = next_obs
            dones[step] = next_done
            rewards[step] = next_reward
            env_ids[step] = info["env_id"]

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, value = get_action_and_value(network, actor, critic, next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

            envs.send(np.array(action.cpu()), info["env_id"])

            truncated = (
                info["elapsed_step"] >= envs.spec.config.max_episode_steps
            )  # https://github.com/sail-sg/envpool/issues/239
            terminated = info["terminated"]
            episode_returns[info["env_id"]] += info["reward"]
            returned_episode_returns[info["env_id"]] = np.where(
                info["terminated"] + truncated, episode_returns[info["env_id"]], returned_episode_returns[info["env_id"]]
            )
            episode_returns[info["env_id"]] *= (1 - info["terminated"]) * (1 - truncated)
            episode_lengths[info["env_id"]] += 1
            returned_episode_lengths[info["env_id"]] = np.where(
                info["terminated"] + truncated, episode_lengths[info["env_id"]], returned_episode_lengths[info["env_id"]]
            )
            episode_lengths[info["env_id"]] *= (1 - info["terminated"]) * (1 - truncated)
            done = truncated | terminated

            next_obs, next_reward, next_done, info = envs.recv()
            next_obs = torch.Tensor(next_obs).to(device)
            next_reward = torch.Tensor(next_reward).to(device).view(-1)
            next_done = torch.Tensor(next_done).to(device)
            lastobs[info["env_id"]] = next_obs
            lastdones[info["env_id"]] = next_done
            lastrewards[info["env_id"]] = next_reward
            # for idx, d in enumerate(done):
            #     if d:
            #         print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
            #         writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
            #         writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
        if args.local_rank == 0:
            avg_episodic_return = np.mean(returned_episode_returns)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
            print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")


        with torch.no_grad():
            # prepare data
            T, B = env_ids.shape
            index_ranges = torch.arange(T * B, dtype=torch.int32)
            next_index_ranges = torch.zeros_like(index_ranges, dtype=torch.long)
            last_env_ids = torch.zeros(args.local_num_envs, dtype=torch.int32) - 1
            carry = (last_env_ids, next_index_ranges)
            for x in zip(env_ids.reshape(-1), index_ranges):
                last_env_ids, next_index_ranges = carry
                env_id, index_range = x
                next_index_ranges[last_env_ids[env_id]] = torch.where(last_env_ids[env_id] != -1, index_range, next_index_ranges[last_env_ids[env_id]])
                last_env_ids[env_id] = index_range
            rewards = rewards.reshape(-1)[next_index_ranges].reshape((args.num_steps) * args.async_update, args.async_batch_size)

            # calculate GAE
            final_env_id_checked = torch.zeros(args.local_num_envs, dtype=torch.int32).to(device) - 1
            final_env_ids = torch.zeros_like(dones, dtype=torch.int32).to(device)
            advantages = torch.zeros((T, B)).to(device)
            lastgaelam = torch.zeros(args.local_num_envs).to(device)
            lastdones = torch.zeros(args.local_num_envs).to(device) + 1
            lastvalues = torch.zeros(args.local_num_envs).to(device) # TODO: we can leverage lastobs to get the lastvalues
            for t in reversed(range(T)):
                eid = env_ids[t]
                nextnonterminal = 1.0 - lastdones[eid]
                nextvalues = lastvalues[eid]
                delta = torch.where(
                    final_env_id_checked[eid] == -1, 0, rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam[eid]
                final_env_ids[t] = torch.where(final_env_id_checked[eid] == 1, 1, 0)
                final_env_id_checked[eid] = torch.where(final_env_id_checked[eid] == -1, 1, final_env_id_checked[eid])
                # the last_ variables keeps track of the actual `num_steps`
                lastgaelam[eid] = advantages[t]
                lastdones[eid] = dones[t]
                lastvalues[eid] = values[t]
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.local_batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.local_batch_size, args.local_minibatch_size):
                end = start + args.local_minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = get_action_and_value2(network, actor, critic, b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent_params, args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if args.local_rank == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar(
                "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - update_time_start)), global_step
            )

    envs.close()
    if args.local_rank == 0:
        writer.close()
        if args.track:
            wandb.finish()
