import robothor_env
import gym

envs = [gym.make("robothor-garbagecan"), gym.make("robothor-houseplant")]
import time
start_time = time.time()
for e in envs:
    e.reset()

n_step = 10000
for _ in range(n_step):
    for env in envs:
        obs, reward, terminate, truncate, info = env.step(env.action_space.sample())
        if terminate or truncate:
            env.reset()
end_time = time.time()
print("time elapse:", end_time - start_time)
print("fps:", n_step*len(envs) / (end_time - start_time))