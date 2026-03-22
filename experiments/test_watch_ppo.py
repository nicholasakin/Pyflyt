import pprint

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

termination_reason = "unknown"
if info.get("collision", False):
    termination_reason = "collision"
elif info.get("out_of_bounds", False):
    termination_reason = "out_of_bounds"
elif info.get("env_complete", False):
    termination_reason = "env_complete"
elif trunc:
    termination_reason = "time_limit_or_truncation"
elif done:
    termination_reason = "termination_without_flag"

print("episode_end_summary:")
pprint.pprint(
    {
        "termination_reason": termination_reason,
        "done": done,
        "trunc": trunc,
        "num_targets_reached": info.get("num_targets_reached"),
        "env_complete": info.get("env_complete"),
        "collision": info.get("collision"),
        "out_of_bounds": info.get("out_of_bounds"),
        "task_config": env_kwargs,
        "final_info": info,
    },
    sort_dicts=False,
)

env.close()
