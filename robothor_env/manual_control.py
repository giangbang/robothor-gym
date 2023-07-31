#!/usr/bin/env python3
# modified from https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/manual_control.py

from __future__ import annotations

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
import pygame
import cv2
import numpy as np

import robothor_env


class ManualControl:

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        env: gym.Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.window = None
        self.clock = None
        self.screen_size = 512
        self.screenshot = None

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        print(self.env.current_v)

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.render(self.env.render())

    def render(self, img):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_size, self.screen_size)
            )
            pygame.display.set_caption("robothor-env")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        img = cv2.resize(img, (self.screen_size, self.screen_size),
                interpolation = cv2.INTER_LINEAR)
        surf = pygame.surfarray.make_surface(img)

        offset = surf.get_size()[0] * 0.1
        # offset = 32 if self.agent_pov else 64
        bg = pygame.Surface(
            (int(surf.get_size()[0] + offset), 
            int(surf.get_size()[1] + offset))
        )
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))
        bg = pygame.transform.rotate(bg, -90)

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.render(self.env.render())
        
    def close(self):
        if self.window:
            pygame.quit()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": 2,
            "right": 3,
            "up": 4,
            "down": 5,
            "w": 0,
            "s": 1,
            "space": 6,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="robothor-apple",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--precompute-file",
        type=str,
        help="",
    )

    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        precompute_file=args.precompute_file,
        scene=args.scene,
    )
    print("Scene:", env.get_scene())

    manual_control = ManualControl(env, seed=args.seed)
    manual_control.start()
