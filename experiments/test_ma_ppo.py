import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from PyFlyt.pz_envs import MAQuadXHoverEnvV2


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    total_timesteps: int = 1_000_000
    num_envs: int = 8
    num_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95

    learning_rate: float = 3e-4
    anneal_lr: bool = True
    num_minibatches: int = 16
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02

    hidden_size: int = 256
    eval_episodes: int = 8
    eval_every_updates: int = 10
    render_eval: bool = False

    num_agents: int = 4
    angle_representation: str = "quaternion"
    train_flight_dome_min: float = 6.0
    train_flight_dome_max: float = 10.0
    train_max_duration_min: float = 12.0
    train_max_duration_max: float = 20.0
    train_spawn_radius_min: float = 0.8
    train_spawn_radius_max: float = 2.2
    train_spawn_jitter: float = 0.35
    train_altitude_min: float = 0.9
    train_altitude_max: float = 1.4

    team_reward_weight: float = 0.75
    include_agent_id: bool = True
    include_team_context: bool = True

    checkpoint_path: str = "pyflyt_ma_ppo.pt"


cfg = Config()
steps_per_update = cfg.num_envs * cfg.num_steps * cfg.num_agents
minibatch_size = steps_per_update // cfg.num_minibatches
num_updates = max(cfg.total_timesteps // steps_per_update, 1)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def sample_start_positions(random_state: np.random.Generator, cfg: Config) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, cfg.num_agents, endpoint=False)
    radius = random_state.uniform(cfg.train_spawn_radius_min, cfg.train_spawn_radius_max)
    altitudes = random_state.uniform(
        cfg.train_altitude_min,
        cfg.train_altitude_max,
        size=cfg.num_agents,
    )
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius
    xy += random_state.uniform(
        low=-cfg.train_spawn_jitter,
        high=cfg.train_spawn_jitter,
        size=(cfg.num_agents, 2),
    )
    return np.concatenate([xy, altitudes[:, None]], axis=-1)


def sample_env_kwargs(random_state: np.random.Generator, cfg: Config) -> dict:
    return {
        "start_pos": sample_start_positions(random_state, cfg),
        "start_orn": np.zeros((cfg.num_agents, 3), dtype=np.float64),
        "sparse_reward": False,
        "angle_representation": cfg.angle_representation,
        "flight_dome_size": float(
            random_state.uniform(cfg.train_flight_dome_min, cfg.train_flight_dome_max)
        ),
        "max_duration_seconds": float(
            random_state.uniform(cfg.train_max_duration_min, cfg.train_max_duration_max)
        ),
    }


def build_eval_suites(cfg: Config) -> list[tuple[str, dict]]:
    base_start = np.array(
        [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    return [
        (
            "train_like",
            {
                "start_pos": base_start.copy(),
                "start_orn": np.zeros((cfg.num_agents, 3), dtype=np.float64),
                "sparse_reward": False,
                "angle_representation": cfg.angle_representation,
                "flight_dome_size": 8.0,
                "max_duration_seconds": 16.0,
            },
        ),
        (
            "heldout_wide_formation",
            {
                "start_pos": base_start * np.array([1.8, 1.8, 1.0]),
                "start_orn": np.zeros((cfg.num_agents, 3), dtype=np.float64),
                "sparse_reward": False,
                "angle_representation": cfg.angle_representation,
                "flight_dome_size": 10.0,
                "max_duration_seconds": 18.0,
            },
        ),
        (
            "heldout_tight_dome",
            {
                "start_pos": base_start * np.array([0.8, 0.8, 1.0]),
                "start_orn": np.zeros((cfg.num_agents, 3), dtype=np.float64),
                "sparse_reward": False,
                "angle_representation": cfg.angle_representation,
                "flight_dome_size": 5.5,
                "max_duration_seconds": 14.0,
            },
        ),
    ]


class MultiAgentObservationAdapter:
    def __init__(self, possible_agents: list[str], obs_dim: int, cfg: Config):
        self.possible_agents = list(possible_agents)
        self.agent_to_idx = {agent: idx for idx, agent in enumerate(self.possible_agents)}
        self.base_obs_dim = obs_dim
        self.include_agent_id = cfg.include_agent_id
        self.include_team_context = cfg.include_team_context
        self.num_agents = len(self.possible_agents)
        self.agent_id_dim = self.num_agents if self.include_agent_id else 0
        self.team_context_dim = 3 if self.include_team_context else 0
        self.obs_dim = self.base_obs_dim + self.agent_id_dim + self.team_context_dim

    def transform(self, obs: np.ndarray, agent: str, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        parts = [np.asarray(obs, dtype=np.float32)]

        if self.include_agent_id:
            one_hot = np.zeros(self.num_agents, dtype=np.float32)
            one_hot[self.agent_to_idx[agent]] = 1.0
            parts.append(one_hot)

        if self.include_team_context:
            stacked = np.stack(
                [np.asarray(obs_dict[other], dtype=np.float32) for other in self.possible_agents],
                axis=0,
            )
            lin_pos = stacked[:, -3:]
            centroid = lin_pos.mean(axis=0)
            spread = np.linalg.norm(lin_pos - centroid, axis=-1).mean()
            team_context = np.array(
                [
                    len(obs_dict) / self.num_agents,
                    centroid[2],
                    spread,
                ],
                dtype=np.float32,
            )
            parts.append(team_context)

        return np.concatenate(parts).astype(np.float32, copy=False)


class ParallelEnvGroup:
    def __init__(self, cfg: Config, num_envs: int, seed: int):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = seed
        self.random_states = [
            np.random.default_rng(seed + idx) for idx in range(num_envs)
        ]
        self.envs = []

        for idx in range(num_envs):
            env = MAQuadXHoverEnvV2(**sample_env_kwargs(self.random_states[idx], cfg))
            self.envs.append(env)

        sample_env = self.envs[0]
        self.possible_agents = list(sample_env.possible_agents)
        self.base_obs_dim = int(np.prod(sample_env.observation_space(self.possible_agents[0]).shape))
        self.act_dim = int(np.prod(sample_env.action_space(self.possible_agents[0]).shape))
        self.adapter = MultiAgentObservationAdapter(
            possible_agents=self.possible_agents,
            obs_dim=self.base_obs_dim,
            cfg=cfg,
        )
        self.num_agents = len(self.possible_agents)
        self.current_obs = [dict() for _ in range(self.num_envs)]
        self.current_task_cfgs = [dict() for _ in range(self.num_envs)]

    def reset(self):
        transformed = []
        alive_masks = []
        for env_idx, env in enumerate(self.envs):
            env.close()
            kwargs = sample_env_kwargs(self.random_states[env_idx], self.cfg)
            self.envs[env_idx] = MAQuadXHoverEnvV2(**kwargs)
            self.current_task_cfgs[env_idx] = kwargs
            obs_dict, _ = self.envs[env_idx].reset(seed=self.seed + env_idx)
            self.current_obs[env_idx] = obs_dict
            transformed.append(self._transform_obs_dict(obs_dict))
            alive_masks.append(self._alive_mask(self.envs[env_idx].agents))
        return np.stack(transformed), np.stack(alive_masks)

    def step(self, actions: np.ndarray):
        next_obs_batch = []
        reward_batch = []
        done_batch = []
        alive_batch = []
        info_batch = []

        for env_idx, env in enumerate(self.envs):
            action_dict = {
                agent: actions[env_idx, agent_idx]
                for agent_idx, agent in enumerate(self.possible_agents)
                if agent in env.agents
            }

            obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)
            self.current_obs[env_idx] = obs_dict

            reward_vec = np.zeros(self.num_agents, dtype=np.float32)
            done_vec = np.ones(self.num_agents, dtype=bool)
            for agent_idx, agent in enumerate(self.possible_agents):
                reward_vec[agent_idx] = float(rewards.get(agent, 0.0))
                done_vec[agent_idx] = bool(
                    terminations.get(agent, True) or truncations.get(agent, True)
                )

            team_reward = reward_vec.mean()
            reward_vec = (
                (1.0 - self.cfg.team_reward_weight) * reward_vec
                + self.cfg.team_reward_weight * team_reward
            ).astype(np.float32)

            next_obs_batch.append(self._transform_obs_dict(obs_dict))
            reward_batch.append(reward_vec)
            done_batch.append(done_vec)
            alive_batch.append(self._alive_mask(env.agents))
            info_batch.append(infos)

        return (
            np.stack(next_obs_batch),
            np.stack(reward_batch),
            np.stack(done_batch),
            np.stack(alive_batch),
            info_batch,
        )

    def _transform_obs_dict(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        transformed = np.zeros(
            (self.num_agents, self.adapter.obs_dim),
            dtype=np.float32,
        )
        if not obs_dict:
            return transformed

        fallback_obs = next(iter(obs_dict.values()))
        full_obs_dict = {
            agent: obs_dict.get(agent, fallback_obs) for agent in self.possible_agents
        }
        for agent_idx, agent in enumerate(self.possible_agents):
            transformed[agent_idx] = self.adapter.transform(
                full_obs_dict[agent], agent, full_obs_dict
            )
        return transformed

    def _alive_mask(self, alive_agents: list[str]) -> np.ndarray:
        return np.array(
            [agent in alive_agents for agent in self.possible_agents],
            dtype=np.float32,
        )

    def close(self):
        for env in self.envs:
            env.close()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic: bool = False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = action_mean if deterministic else probs.sample()

        squashed_action = torch.tanh(action)
        log_prob = probs.log_prob(action).sum(-1)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6).sum(-1)
        entropy = probs.entropy().sum(-1)
        value = self.critic(x).squeeze(-1)
        return squashed_action, log_prob, entropy, value

    def evaluate_given_squashed_action(self, x, squashed_action):
        squashed_action = torch.clamp(squashed_action, -0.999999, 0.999999)
        raw_action = 0.5 * torch.log((1 + squashed_action) / (1 - squashed_action))
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        log_prob = probs.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6).sum(-1)
        entropy = probs.entropy().sum(-1)
        value = self.critic(x).squeeze(-1)
        return log_prob, entropy, value


@torch.no_grad()
def evaluate(agent: ActorCritic, cfg: Config, device: str, render: bool = False):
    results = []
    for suite_idx, (suite_name, env_kwargs) in enumerate(build_eval_suites(cfg)):
        episode_team_returns = []
        episode_survival = []

        for episode in range(cfg.eval_episodes):
            env = MAQuadXHoverEnvV2(
                **env_kwargs,
                render_mode="human" if render and suite_idx == 0 and episode == 0 else None,
            )
            adapter = MultiAgentObservationAdapter(
                possible_agents=list(env.possible_agents),
                obs_dim=int(np.prod(env.observation_space(env.possible_agents[0]).shape)),
                cfg=cfg,
            )
            obs_dict, _ = env.reset(seed=cfg.seed + 10_000 + episode)
            done = False
            team_return = 0.0

            while not done:
                full_obs = {
                    agent: obs_dict.get(agent, next(iter(obs_dict.values())))
                    for agent in env.possible_agents
                }
                obs_batch = np.stack(
                    [
                        adapter.transform(full_obs[agent], agent, full_obs)
                        for agent in env.possible_agents
                    ]
                )
                obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                action_batch, _, _, _ = agent.get_action_and_value(
                    obs_t, deterministic=True
                )
                action_np = action_batch.cpu().numpy()
                action_dict = {
                    env.possible_agents[idx]: action_np[idx]
                    for idx in range(len(env.possible_agents))
                    if env.possible_agents[idx] in env.agents
                }

                obs_dict, rewards, terminations, truncations, _ = env.step(action_dict)
                if rewards:
                    team_return += float(np.mean(list(rewards.values())))
                done = not env.agents or all(
                    terminations.get(agent_name, True) or truncations.get(agent_name, True)
                    for agent_name in env.possible_agents
                )

            episode_team_returns.append(team_return)
            episode_survival.append(len(env.agents) / len(env.possible_agents))
            env.close()

        results.append(
            (
                suite_name,
                {
                    "mean_team_return": float(np.mean(episode_team_returns)),
                    "std_team_return": float(np.std(episode_team_returns)),
                    "mean_final_survival": float(np.mean(episode_survival)),
                },
            )
        )
    return results


def main():
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    env_group = ParallelEnvGroup(cfg=cfg, num_envs=cfg.num_envs, seed=cfg.seed)

    obs_dim = env_group.adapter.obs_dim
    act_dim = env_group.act_dim
    agent = ActorCritic(obs_dim, act_dim, cfg.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    obs_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, cfg.num_agents, obs_dim),
        dtype=torch.float32,
        device=device,
    )
    actions_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, cfg.num_agents, act_dim),
        dtype=torch.float32,
        device=device,
    )
    logprobs_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, cfg.num_agents),
        dtype=torch.float32,
        device=device,
    )
    rewards_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, cfg.num_agents),
        dtype=torch.float32,
        device=device,
    )
    dones_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, cfg.num_agents),
        dtype=torch.float32,
        device=device,
    )
    alive_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, cfg.num_agents),
        dtype=torch.float32,
        device=device,
    )
    values_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, cfg.num_agents),
        dtype=torch.float32,
        device=device,
    )

    global_step = 0
    start_time = time.time()
    next_obs_np, next_alive_np = env_group.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_alive = torch.tensor(next_alive_np, dtype=torch.float32, device=device)
    next_done = 1.0 - next_alive

    episodic_team_returns: list[float] = []
    running_team_returns = np.zeros(cfg.num_envs, dtype=np.float32)

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for step in range(cfg.num_steps):
            global_step += cfg.num_envs * cfg.num_agents
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            alive_buf[step] = next_alive

            flat_next_obs = next_obs.reshape(cfg.num_envs * cfg.num_agents, obs_dim)
            flat_alive = next_alive.reshape(cfg.num_envs * cfg.num_agents)

            with torch.no_grad():
                flat_actions, flat_logprob, _, flat_values = agent.get_action_and_value(
                    flat_next_obs
                )

            flat_actions = flat_actions.reshape(cfg.num_envs, cfg.num_agents, act_dim)
            flat_logprob = flat_logprob.reshape(cfg.num_envs, cfg.num_agents)
            flat_values = flat_values.reshape(cfg.num_envs, cfg.num_agents)

            actions_buf[step] = flat_actions
            logprobs_buf[step] = flat_logprob * next_alive
            values_buf[step] = flat_values * next_alive

            next_obs_np, rewards_np, dones_np, next_alive_np, _ = env_group.step(
                flat_actions.cpu().numpy()
            )
            rewards_np = rewards_np * next_alive_np
            running_team_returns += rewards_np.mean(axis=1)

            for env_idx in range(cfg.num_envs):
                if next_alive_np[env_idx].sum() == 0:
                    episodic_team_returns.append(float(running_team_returns[env_idx]))
                    running_team_returns[env_idx] = 0.0

            rewards_buf[step] = torch.tensor(
                rewards_np, dtype=torch.float32, device=device
            )
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_alive = torch.tensor(next_alive_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(dones_np, dtype=torch.float32, device=device)

            dead_envs = np.where(next_alive_np.sum(axis=1) == 0)[0]
            for env_idx in dead_envs:
                env_group.envs[env_idx].close()
                kwargs = sample_env_kwargs(env_group.random_states[env_idx], cfg)
                env_group.envs[env_idx] = MAQuadXHoverEnvV2(**kwargs)
                obs_dict, _ = env_group.envs[env_idx].reset(seed=cfg.seed + update + env_idx)
                env_group.current_obs[env_idx] = obs_dict
                next_obs[env_idx] = torch.tensor(
                    env_group._transform_obs_dict(obs_dict),
                    dtype=torch.float32,
                    device=device,
                )
                next_alive[env_idx] = torch.tensor(
                    env_group._alive_mask(env_group.envs[env_idx].agents),
                    dtype=torch.float32,
                    device=device,
                )
                next_done[env_idx] = 1.0 - next_alive[env_idx]

        with torch.no_grad():
            flat_next_obs = next_obs.reshape(cfg.num_envs * cfg.num_agents, obs_dim)
            next_value = agent.get_value(flat_next_obs).reshape(cfg.num_envs, cfg.num_agents)
            next_value = next_value * next_alive
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = torch.zeros((cfg.num_envs, cfg.num_agents), device=device)

            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]

                next_nonterminal = next_nonterminal * alive_buf[t]
                delta = (
                    rewards_buf[t]
                    + cfg.gamma * next_values * next_nonterminal
                    - values_buf[t]
                )
                lastgaelam = (
                    delta
                    + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgaelam
                )
                advantages[t] = lastgaelam * alive_buf[t]

            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions_buf.reshape((-1, act_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_alive = alive_buf.reshape(-1)

        valid_mask = b_alive > 0.5
        b_obs = b_obs[valid_mask]
        b_actions = b_actions[valid_mask]
        b_logprobs = b_logprobs[valid_mask]
        b_advantages = b_advantages[valid_mask]
        b_returns = b_returns[valid_mask]
        b_values = b_values[valid_mask]

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_inds = np.arange(len(b_obs))
        clipfracs = []
        approx_kl = torch.tensor(0.0, device=device)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(batch_inds)

            for start in range(0, len(batch_inds), minibatch_size):
                end = start + minibatch_size
                mb_inds = batch_inds[start:end]
                if len(mb_inds) == 0:
                    continue

                newlogprob, entropy, newvalue = agent.evaluate_given_squashed_action(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -cfg.clip_coef,
                    cfg.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        sps = int(global_step / (time.time() - start_time))
        avg_team_return = (
            float(np.mean(episodic_team_returns[-20:]))
            if episodic_team_returns
            else float("nan")
        )
        print(
            f"update={update:4d}/{num_updates} | "
            f"step={global_step:8d} | "
            f"avg_team_return(20)={avg_team_return:8.3f} | "
            f"approx_kl={approx_kl.item():8.5f} | "
            f"clipfrac={np.mean(clipfracs) if clipfracs else 0.0:8.3f} | "
            f"sps={sps}"
        )

        if update % cfg.eval_every_updates == 0:
            for suite_name, metrics in evaluate(agent, cfg, cfg.device, cfg.render_eval):
                print(
                    f"[eval:{suite_name}] "
                    f"mean_team_return={metrics['mean_team_return']:.3f} +/- "
                    f"{metrics['std_team_return']:.3f} | "
                    f"mean_final_survival={metrics['mean_final_survival']:.2%}"
                )

    env_group.close()
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "config": asdict(cfg),
        },
        cfg.checkpoint_path,
    )
    print(f"Saved model to {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
