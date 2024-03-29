import argparse
import os
import random
import time
import uuid
from distutils.util import strtobool

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--wandb-entity", type=str, default="ryan-colab",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="AcrobotSwingup-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
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

    # Dreamer Tricks
    parser.add_argument("--symlog", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--symobs", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--two-hot", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--unimix", type=float, default=0.0)
    parser.add_argument("--percentile-scale", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--percentile-ema-rate", type=float, default=0.99)
    parser.add_argument("--critic-zero-init", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--critic-ema", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--critic-ema-rate", type=float, default=0.98)
    parser.add_argument("--critic-ema-coef", type=float, default=1.0)
    parser.add_argument("--return-lambda", type=float, default=0.95)

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            seed=seed,
        )
        envs = FlattenObservation(envs)
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        envs = RecordEpisodeStatistics(envs)
        return envs

    return thunk


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def calc_twohot(x, B):
    """
    x shape:(n_vals, ) is tensor of values
    B shape:(n_bins, ) is tensor of bin values
    returns a twohot tensor of shape (n_vals, n_bins)

    can verify this method is correct with:
     - calc_twohot(x, B)@B == x # expected value reconstructs x
     - (calc_twohot(x, B)>0).sum(dim=-1) == 2 # only two bins are hot
    """
    twohot = torch.zeros((x.shape+B.shape), dtype=x.dtype, device=x.device)
    k1 = (B[None, :] <= x[:, None]).sum(dim=-1)-1
    k2 = k1+1
    k1 = torch.clip(k1, 0, len(B) - 1)
    k2 = torch.clip(k2, 0, len(B) - 1)

    # Handle k1 == k2 case
    equal = (k1 == k2)
    dist_to_below = torch.where(equal, 1, torch.abs(B[k1] - x))
    dist_to_above = torch.where(equal, 0, torch.abs(B[k2] - x))

    # Assign values to two-hot tensor
    total = dist_to_above + dist_to_below
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    x_range = np.arange(len(x))
    twohot[x_range, k1] = weight_below   # assign left
    twohot[x_range, k2] = weight_above   # assign right
    return twohot


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= (1 - dones)
        self.episode_lengths *= (1 - dones)
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        # here the reshape ensures items like `'dist_to_target' --> array([0., 0., 0., 0., 0., 0., 0., 0.])`
        # will have shapes like `(8, 1)` instead of `(8,)`
        return np.concatenate([v.reshape(self.num_envs, -1) for v in observation.values()], 1)


def layer_init(layer, zero=False, std=np.sqrt(2), bias_const=0.0):
    if zero:
        torch.nn.init.zeros_(layer.weight)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)

    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.args = args

        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        if args.two_hot:
            self.B = torch.nn.Parameter(torch.linspace(-20, 20, 256))   # (256, )
            self.B.requires_grad = False
            self.critic = layer_init(nn.Linear(512, len(self.B)), zero=args.critic_zero_init, std=1)
        else:
            self.critic = layer_init(nn.Linear(512, 1), zero=args.critic_zero_init, std=1)

    def critic_val(self, net_out):  # (b, 256)
        if self.args.two_hot:
            logits_critic = self.critic(net_out)
            val = logits_critic.softmax(dim=-1) @ self.B[:, None]   # (b, 256) @ (256, 1) = (b, 1)
        else:
            val = self.critic(net_out)
            logits_critic = None
        val = symexp(val) if args.symlog else val
        return val, logits_critic

    def get_value(self, x):
        if args.symobs:
            x = symlog(x)
        val, _ = self.critic_val(self.network(x))
        return val

    def get_action_and_value(self, x, action=None):
        if args.symobs:
            x = symlog(x)
        hidden = self.network(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        val, logits_critic = self.critic_val(hidden)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), val, logits_critic


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"
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
    envs = make_env(args.env_id, args.seed, args.num_envs)()
    assert isinstance(envs.action_space, gym.spaces.Box), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    if args.compile:
        agent = torch.compile(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Create EMA of critic parameters
    if args.critic_ema:
        critic_ema = Agent(envs, args).to(device)
        critic_ema.network = agent.network
        # TODO: Test if this is correct
        critic_ema.critic.load_state_dict(agent.critic.state_dict())

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    if args.two_hot:
        logits_critics = torch.zeros((args.num_steps, args.num_envs, len(agent.B))).to(device)
    if args.percentile_scale:
        low_ema = torch.zeros(()).to(device)
        high_ema = torch.zeros(()).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        update_time_start = time.time()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, logits_critic = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                if args.two_hot:
                    logits_critics[step] = logits_critic

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, cpu_next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(cpu_next_done).to(device)

            truncated = (
                info["elapsed_step"] >= envs.spec.config.max_episode_steps
            )  # https://github.com/sail-sg/envpool/issues/239
            terminated = cpu_next_done
            done = truncated | terminated
            for idx, d in enumerate(done):
                if d:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        if args.percentile_scale:
            # calculate lambda returns like in Dreamer-V3
            with torch.no_grad():
                ret = torch.zeros_like(rewards)
                ret[-1] = values[-1]
                for t in reversed(range(len(rewards))[:-1]):
                    lam = args.return_lambda
                    ret[t] = (rewards[t] + args.gamma * (~(dones[t+1] > 0)) *
                              ((1-lam) * values[t+1] + lam * ret[t+1]))
                low, high = ret.quantile(0.05), ret.quantile(0.95)
                decay = args.percentile_ema_rate
                low_ema = low if low_ema is None else decay * low_ema + (1 - decay) * low
                high_ema = high if high_ema is None else decay * high_ema + (1 - decay) * high
                S = high_ema - low_ema
                writer.add_scalar("charts/5th_percentile", low_ema, global_step)
                writer.add_scalar("charts/95th_percentile", high_ema, global_step)
                writer.add_scalar("charts/truncated_return", torch.mean(ret).item(), global_step)
                writer.add_scalar("charts/advantage_scale", S.item(), global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        if args.symlog:
            returns = symlog(returns)
            values = symlog(values)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1) / max(1., S.item()) if args.percentile_scale else advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        if args.two_hot:
            b_logits_critics = logits_critics.reshape(-1, len(agent.B))
        
        writer.add_scalar("charts/mean_abs_advantages", torch.mean(torch.abs(b_advantages)), global_step)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, newlogitscritic = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                if args.symlog:
                    newvalue = symlog(newvalue)
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

                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]
                newvalue = newvalue.view(-1)

                # Value loss
                if args.two_hot:
                    with torch.no_grad():
                        twohot_target = calc_twohot(mb_returns, agent.B)
                    v_loss_unclipped = nn.functional.cross_entropy(newlogitscritic, twohot_target, reduction='mean')
                else:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2

                # Critic EMA
                if args.critic_ema:
                    with torch.no_grad():
                        _, _, _, val_ema, logits_ema = critic_ema.get_action_and_value(b_obs[mb_inds])
                    # regularize output distribution to match that of the EMA critic
                    if args.two_hot:
                        v_loss_reg = nn.functional.cross_entropy(newlogitscritic, logits_ema.softmax(dim=-1), reduction='mean')
                    else:
                        val_ema = symlog(val_ema.view(-1)) if args.symlog else val_ema.view(-1)
                        v_loss_reg = 0.5 * ((newvalue - val_ema) ** 2).mean()
                    v_loss_unclipped = v_loss_unclipped + args.critic_ema_coef * v_loss_reg.mean()

                # Value clipping
                if args.clip_vloss:
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = v_loss_unclipped if args.two_hot else 0.5 * v_loss_unclipped.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                
                # Update Critic EMA weights
                if args.critic_ema:
                    with torch.no_grad():
                        decay = args.critic_ema_rate
                        for fast, slow in zip(agent.critic.parameters(), critic_ema.critic.parameters()):
                            slow.copy_(decay * slow + (1 - decay) * fast)
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
    writer.close()
