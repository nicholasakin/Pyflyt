import gymnasium
import PyFlyt.gym_envs
import time


env = gymnasium.make("PyFlyt/QuadX-Hover-v4",
                     render_mode="human")
obs, info = env.reset()






