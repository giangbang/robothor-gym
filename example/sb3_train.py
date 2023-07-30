import robothor_env
try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
import argparse

import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute-file", type=str, default=None,
        help="")
    parser.add_argument("--n-step", type=int, default=1e6,
        help="")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    file_path = args.precompute_file

    env = gym.make("robothor-precompute", precompute_file=file_path)
    
    
    model = RecurrentPPO("MlpLstmPolicy", env=env, verbose=1, tensorboard_log="./runs/")
    model.learn(args.n_step)

    vec_env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
    print(mean_reward)

    model.save("ppo_recurrent")
    del model # remove to demonstrate saving and loading

    model = RecurrentPPO.load("ppo_recurrent")

    obs = vec_env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    obses = []
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        episode_starts = dones
        obses.append(vec_env.render())
        if dones.any():
            break
    
    import cv2
    import os

    video_name = 'demo.avi'

    images = obses
    height, width = obses[0].shape[:2]

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()
