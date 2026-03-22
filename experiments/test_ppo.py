import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import PyFlyt.gym_envs  # noqa: F401


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    env_id: str = "PyFlyt/QuadX-Waypoints-v4"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    total_timesteps: int = 1_000_000
    num_envs: int = 8
    num_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95

    learning_rate: float = 3e-4
    anneal_lr: bool = True

    num_minibatches: int = 16
    update_epochs: int = 10

    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02

    hidden_size: int = 256
    eval_episodes: int = 8
    eval_every_updates: int = 10
    render_eval: bool = False

    max_observed_targets: int = 6
    target_feature_dim: int = 4
    include_task_context: bool = True

    train_num_targets_min: int = 2
    train_num_targets_max: int = 4
    train_flight_dome_min: float = 4.0
    train_flight_dome_max: float = 6.0
    train_goal_reach_distance_min: float = 0.18
    train_goal_reach_distance_max: float = 0.28
    train_max_duration_seconds_min: float = 10.0
    train_max_duration_seconds_max: float = 16.0
    train_yaw_target_prob: float = 0.25
    sparse_reward: bool = False

    checkpoint_path: str = "pyflyt_ppo.pt"


cfg = Config()
batch_size = cfg.num_envs * cfg.num_steps
minibatch_size = batch_size // cfg.num_minibatches
num_updates = cfg.total_timesteps // batch_size


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def sample_train_task(random_state: np.random.Generator, cfg: Config) -> dict:
    return {
        "sparse_reward": cfg.sparse_reward,
        "num_targets": int(
            random_state.integers(
                cfg.train_num_targets_min,
                cfg.train_num_targets_max + 1,
            )
        ),
        "use_yaw_targets": bool(
            random_state.random() < cfg.train_yaw_target_prob
        ),
        "goal_reach_distance": float(
            random_state.uniform(
                cfg.train_goal_reach_distance_min,
                cfg.train_goal_reach_distance_max,
            )
        ),
        "flight_dome_size": float(
            random_state.uniform(
                cfg.train_flight_dome_min,
                cfg.train_flight_dome_max,
            )
        ),
        "max_duration_seconds": float(
            random_state.uniform(
                cfg.train_max_duration_seconds_min,
                cfg.train_max_duration_seconds_max,
            )
        ),
    }


def build_eval_suites(cfg: Config) -> list[tuple[str, dict]]:
    train_center = {
        "sparse_reward": cfg.sparse_reward,
        "num_targets": 3,
        "use_yaw_targets": False,
        "goal_reach_distance": 0.22,
        "flight_dome_size": 5.0,
        "max_duration_seconds": 12.0,
    }
    return [
        ("train_like", train_center),
        (
            "heldout_more_targets",
            {**train_center, "num_targets": cfg.max_observed_targets},
        ),
        (
            "heldout_bigger_dome",
            {**train_center, "flight_dome_size": 8.0},
        ),
        (
            "heldout_yaw_targets",
            {**train_center, "use_yaw_targets": True, "goal_reach_distance": 0.2},
        ),
        (
            "heldout_tighter_goal",
            {**train_center, "goal_reach_distance": 0.14},
        ),
    ]


def default_watch_env_kwargs(cfg: Config) -> dict:
    return build_eval_suites(cfg)[0][1]


class WaypointObservationWrapper(gym.ObservationWrapper):
    """Pads variable-length waypoint observations to a fixed-size flat vector."""

    def __init__(
        self,
        env: gym.Env,
        max_observed_targets: int,
        target_feature_dim: int,
        include_task_context: bool,
    ):
        super().__init__(env)
        self.max_observed_targets = max_observed_targets
        self.target_feature_dim = target_feature_dim
        self.include_task_context = include_task_context

        attitude_space = env.observation_space["attitude"]
        self.attitude_dim = attitude_space.shape[0]
        extra_context_dim = 4 if include_task_context else 0
        flat_dim = (
            self.attitude_dim
            + max_observed_targets * target_feature_dim
            + max_observed_targets
            + extra_context_dim
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, observation):
        attitude = np.asarray(observation["attitude"], dtype=np.float32)
        target_deltas = np.asarray(observation["target_deltas"], dtype=np.float32)

        padded_targets = np.zeros(
            (self.max_observed_targets, self.target_feature_dim),
            dtype=np.float32,
        )
        target_mask = np.zeros(self.max_observed_targets, dtype=np.float32)

        visible_targets = min(len(target_deltas), self.max_observed_targets)
        if visible_targets > 0:
            feature_dim = min(target_deltas.shape[-1], self.target_feature_dim)
            padded_targets[:visible_targets, :feature_dim] = target_deltas[
                :visible_targets, :feature_dim
            ]
            target_mask[:visible_targets] = 1.0

        flat_obs = [
            attitude,
            padded_targets.reshape(-1),
            target_mask,
        ]

        if self.include_task_context:
            env = self.unwrapped
            waypoints = getattr(env, "waypoints", None)
            flat_obs.append(
                np.array(
                    [
                        visible_targets / max(self.max_observed_targets, 1),
                        float(getattr(waypoints, "use_yaw_targets", False)),
                        float(getattr(env, "flight_dome_size", 0.0)) / 10.0,
                        float(getattr(waypoints, "goal_reach_distance", 0.0)),
                    ],
                    dtype=np.float32,
                )
            )

        return np.concatenate(flat_obs).astype(np.float32, copy=False)


def maybe_wrap_observation(env: gym.Env, cfg: Config) -> gym.Env:
    if (
        isinstance(env.observation_space, gym.spaces.Dict)
        and "target_deltas" in env.observation_space.spaces
        and isinstance(
            env.observation_space.spaces["target_deltas"],
            gym.spaces.Sequence,
        )
    ):
        return WaypointObservationWrapper(
            env,
            max_observed_targets=cfg.max_observed_targets,
            target_feature_dim=cfg.target_feature_dim,
            include_task_context=cfg.include_task_context,
        )
    return env


def build_env(
    env_id: str,
    env_kwargs: dict,
    cfg: Config,
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    env = maybe_wrap_observation(env, cfg)
    return env


class DomainRandomizedQuadXWaypointsEnv(gym.Env):
    """Recreates the underlying env on reset so each episode can sample a new task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, env_id: str, cfg: Config, seed: int):
        super().__init__()
        self.env_id = env_id
        self.cfg = cfg
        self.random_state = np.random.default_rng(seed)
        self.env: gym.Env | None = None
        self.current_env_kwargs: dict | None = None

        initial_env = build_env(
            env_id=self.env_id,
            env_kwargs=sample_train_task(self.random_state, self.cfg),
            cfg=self.cfg,
        )
        self.action_space = initial_env.action_space
        self.observation_space = initial_env.observation_space
        initial_env.close()

    def _new_env(self):
        if self.env is not None:
            self.env.close()
        self.current_env_kwargs = sample_train_task(self.random_state, self.cfg)
        self.env = build_env(
            env_id=self.env_id,
            env_kwargs=self.current_env_kwargs,
            cfg=self.cfg,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._new_env()
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["task_config"] = dict(self.current_env_kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["task_config"] = dict(self.current_env_kwargs)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def render(self):
        if self.env is None:
            return None
        return self.env.render()


def make_train_env(env_id: str, seed: int, idx: int, cfg: Config):
    def thunk():
        env = DomainRandomizedQuadXWaypointsEnv(
            env_id=env_id,
            cfg=cfg,
            seed=seed + idx,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def make_eval_env(
    env_id: str,
    env_kwargs: dict,
    cfg: Config,
    render: bool = False,
) -> gym.Env:
    env = build_env(
        env_id=env_id,
        env_kwargs=env_kwargs,
        cfg=cfg,
        render_mode="human" if render else None,
    )
    return env


# ============================================================
# Actor-Critic
# ============================================================
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


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_suite(
    agent: ActorCritic,
    env_id: str,
    env_kwargs: dict,
    cfg: Config,
    device: str,
    episodes: int,
    render: bool = False,
):
    env = make_eval_env(env_id, env_kwargs, cfg, render=render)
    returns = []
    completions = []
    targets_reached = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + 10_000 + ep)
        done = False
        trunc = False
        ep_return = 0.0
        final_info = {}

        while not (done or trunc):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = agent.get_action_and_value(obs_t, deterministic=True)
            obs, reward, done, trunc, info = env.step(action.squeeze(0).cpu().numpy())
            ep_return += reward
            final_info = info

        returns.append(ep_return)
        completions.append(float(final_info.get("env_complete", False)))
        targets_reached.append(float(final_info.get("num_targets_reached", 0)))

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "completion_rate": float(np.mean(completions)),
        "mean_targets_reached": float(np.mean(targets_reached)),
    }


def evaluate_generalization(
    agent: ActorCritic,
    cfg: Config,
    device: str,
    render: bool = False,
):
    suite_metrics = []
    for idx, (suite_name, env_kwargs) in enumerate(build_eval_suites(cfg)):
        metrics = evaluate_suite(
            agent=agent,
            env_id=cfg.env_id,
            env_kwargs=env_kwargs,
            cfg=cfg,
            device=device,
            episodes=cfg.eval_episodes,
            render=render and idx == 0,
        )
        suite_metrics.append((suite_name, env_kwargs, metrics))
    return suite_metrics


# ============================================================
# Main
# ============================================================
def main():
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    envs = gym.vector.SyncVectorEnv(
        [make_train_env(cfg.env_id, cfg.seed, i, cfg) for i in range(cfg.num_envs)]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "PPO code expects a continuous Box action space."
    assert isinstance(
        envs.single_observation_space, gym.spaces.Box
    ), "PPO code expects a flat Box observation space."

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = ActorCritic(obs_dim, act_dim, cfg.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    obs_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, obs_dim),
        dtype=torch.float32,
        device=device,
    )
    actions_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, act_dim),
        dtype=torch.float32,
        device=device,
    )
    logprobs_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs),
        dtype=torch.float32,
        device=device,
    )
    rewards_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs),
        dtype=torch.float32,
        device=device,
    )
    dones_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs),
        dtype=torch.float32,
        device=device,
    )
    values_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs),
        dtype=torch.float32,
        device=device,
    )

    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).view(
        cfg.num_envs, -1
    )
    next_done = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

    episodic_returns = []
    completed_episodes = 0

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            values_buf[step] = value
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done_np = np.logical_or(terminations, truncations)

            rewards_buf[step] = torch.tensor(
                reward, dtype=torch.float32, device=device
            )
            next_obs = torch.tensor(
                next_obs_np, dtype=torch.float32, device=device
            ).view(cfg.num_envs, -1)
            next_done = torch.tensor(
                next_done_np, dtype=torch.float32, device=device
            )

            if "final_info" in infos:
                for item in infos["final_info"]:
                    if item is not None and "episode" in item:
                        ep_r = item["episode"]["r"]
                        if isinstance(ep_r, np.ndarray):
                            ep_r = float(ep_r.item())
                        episodic_returns.append(float(ep_r))
                        completed_episodes += 1

        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0

            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]

                delta = (
                    rewards_buf[t]
                    + cfg.gamma * next_values * next_nonterminal
                    - values_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + cfg.gamma
                    * cfg.gae_lambda
                    * next_nonterminal
                    * lastgaelam
                )

            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions_buf.reshape((-1, act_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0, device=device)

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

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
        avg_return_str = (
            f"{np.mean(episodic_returns[-20:]):8.3f}"
            if episodic_returns
            else " pending"
        )

        print(
            f"update={update:4d}/{num_updates} | "
            f"step={global_step:8d} | "
            f"avg_ep_return(20)={avg_return_str} | "
            f"completed_eps={completed_episodes:4d} | "
            f"approx_kl={approx_kl.item():8.5f} | "
            f"clipfrac={np.mean(clipfracs):8.3f} | "
            f"sps={sps}"
        )

        if update % cfg.eval_every_updates == 0:
            for suite_name, env_kwargs, metrics in evaluate_generalization(
                agent=agent,
                cfg=cfg,
                device=cfg.device,
                render=cfg.render_eval,
            ):
                print(
                    f"[eval:{suite_name}] "
                    f"mean_return={metrics['mean_return']:.3f} +/- "
                    f"{metrics['std_return']:.3f} | "
                    f"completion_rate={metrics['completion_rate']:.2%} | "
                    f"mean_targets_reached={metrics['mean_targets_reached']:.2f} | "
                    f"task={env_kwargs}"
                )

    envs.close()

    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "config": asdict(cfg),
            "watch_env_kwargs": default_watch_env_kwargs(cfg),
        },
        cfg.checkpoint_path,
    )
    print(f"Saved model to {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
