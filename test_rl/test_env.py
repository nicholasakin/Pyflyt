import gymnasium
import PyFlyt.gym_envs
import time

env = gymnasium.make("PyFlyt/QuadX-Hover-v4",
                     render_mode="human")
obs, info = env.reset()

termination = False
truncation = False
reward = 0

N = 500
counter = 0
#while not (termination or truncation):
start = time.perf_counter()
for i in range(N):
    
    counter = counter+1
    observation, reward, truncation, termination, info = env.step(
            env.action_space.sample())
    if termination or truncation:
        obs, info = env.reset()

    #if counter % 5 == 0:
        #print("Iteration: " + str(counter))
        #print("observation: " + str(observation))
        #print("reward: " + str(reward))
        #print("truncation: " + str(truncation))
        #print("termination: " + str(termination))
        #print("info: + " + str(info))

end = time.perf_counter()
elapsed = end-start
print("DONE!")
print(f"Total time: {elapsed:.3f}s")
print(f"Steps/sec: {N/elapsed:.2f}")
print()
print("truncation: " + str(truncation))
print("termination: " + str(termination))
print("reward: " + str(reward))



