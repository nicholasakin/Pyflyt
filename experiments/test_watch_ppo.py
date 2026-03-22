import gymnasium as gym
import torch
import numpy as np
import PyFlyt.gym_envs  # noqa: F401
from test_ppo import ActorCritic

checkpoint = torch.load("pyflyt_ppo.pt", map_location="cpu")
cfg_dict = checkpoint["config"]

env_id = cfg_dict["env_id"]
hidden_size = cfg_dict["hidden_size"]

env = gym.make(env_id, render_mode="human")

if isinstance(env.observation_space, gym.spaces.Dict) and \
   any(isinstance(s, gym.spaces.Sequence) for s in env.observation_space.spaces.values()):

    num_targets = 1
    attitude_dim = env.observation_space["attitude"].shape[0]
    target_dim = env.observation_space["target_deltas"].feature_space.shape[0]
    flat_dim = attitude_dim + num_targets * target_dim
    flat_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float64)

    def flatten_obs(obs):
        attitude = obs["attitude"]
        targets = np.zeros((num_targets, target_dim), dtype=np.float64)
        n = min(len(obs["target_deltas"]), num_targets)
        if n > 0:
            targets[:n] = np.array(obs["target_deltas"][:n])
        return np.concatenate([attitude, targets.ravel()])

    env = gym.wrappers.TransformObservation(env, flatten_obs, flat_space)

obs, _ = env.reset()

obs_dim = int(np.prod(env.observation_space.shape))
act_dim = int(np.prod(env.action_space.shape))

agent = ActorCritic(obs_dim, act_dim, hidden_size)
agent.load_state_dict(checkpoint["model_state_dict"])
agent.eval()

done = False
trunc = False

while not (done or trunc):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(obs_t)
    obs, reward, done, trunc, info = env.step(action.squeeze(0).numpy())

env.close()