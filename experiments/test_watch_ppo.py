import numpy as np
import torch

from test_ppo import ActorCritic, Config, default_watch_env_kwargs, make_eval_env


checkpoint = torch.load("pyflyt_ppo.pt", map_location="cpu")
cfg = Config(**checkpoint["config"])
env_kwargs = checkpoint.get("watch_env_kwargs", default_watch_env_kwargs(cfg))

env = make_eval_env(cfg.env_id, env_kwargs, cfg, render=True)
obs, _ = env.reset(seed=cfg.seed)

obs_dim = int(np.prod(env.observation_space.shape))
act_dim = int(np.prod(env.action_space.shape))

agent = ActorCritic(obs_dim, act_dim, cfg.hidden_size)
agent.load_state_dict(checkpoint["model_state_dict"])
agent.eval()

done = False
trunc = False

while not (done or trunc):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(obs_t, deterministic=True)
    obs, reward, done, trunc, info = env.step(action.squeeze(0).numpy())

print(
    "episode_return_info:",
    {
        "num_targets_reached": info.get("num_targets_reached"),
        "env_complete": info.get("env_complete"),
        "task_config": env_kwargs,
    },
)

env.close()
