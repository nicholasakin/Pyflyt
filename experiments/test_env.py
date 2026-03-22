import gymnasium as gym
import PyFlyt.gym_envs  # noqa: F401

env = gym.make("PyFlyt/QuadX-Waypoints-v4")
print(type(env.observation_space))
print(env.observation_space)
env.close()