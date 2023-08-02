import robothor_env
try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
import argparse

import numpy as np
import cv2
import os

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ppo",
        choices=["lstm-ppo", "ppo", "dqn"])
    parser.add_argument("--precompute-file", type=str, default=None,
        help="")
    parser.add_argument("--n-step", type=int, default=1e6,
        help="")
    parser.add_argument("--eval-freq", type=int, default=10_000,
        help="")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    file_path = args.precompute_file

    env = gym.make("robothor-precompute", precompute_file=file_path)
    eval_env = gym.make("robothor-precompute", precompute_file=file_path)

    if args.algorithm == "lstm-ppo":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("CnnLstmPolicy", env=env, verbose=1, tensorboard_log="./runs/")
    elif args.algorithm == "ppo":
        from stable_baselines3 import PPO
        model = PPO("CnnPolicy", env=env, verbose=1, tensorboard_log="./runs/")
    elif args.algorithm == "dqn":
        from stable_baselines3 import DQN
        model = DQN("CnnPolicy", env=env, verbose=1, tensorboard_log="./runs/")


    eval_callback = EvalCallback(eval_env, best_model_save_path="./runs/",
                             log_path="./logs/", eval_freq=args.eval_freq,
                             deterministic=True, render=False)
    model.learn(args.n_step, callback=eval_callback)

    vec_env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
    print("Average reward of final policy:", mean_reward)

    model.save(args.algorithm)

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
        obses.append(np.transpose(obs[0], (1, 2, 0)))
        if dones.any():
            break

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
