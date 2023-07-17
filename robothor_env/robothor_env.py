import os
import ai2thor
import numpy as np
import random

from ai2thor.controller import Controller
import gym
from ai2thor.util.metrics import (
    path_distance
)

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

version = "AI2-THOR Version: " + ai2thor.__version__
print(version)


# more information at https://ai2thor.allenai.org/robothor/documentation/#initialization-scene
SCENE_ROBOTHOR = [
    f"FloorPlan_Train{i}_{j}"
    for i in range(1, 13, 1)
    for j in range(1, 6, 1)
]

TARGET_OBJECT_TYPES = [
    "AlarmClock",
    "Apple",
    "BaseballBat",
    "BasketBall",
    "Bowl",
    "GarbageCan",
    "HousePlant",
    "Laptop",
    "Mug",
    "RemoteControl",
    "SprayBottle",
    "Television",
    "Vase",
]


class AI2Thor(gym.Env):

    def randomize_controller(self):
        self.controller.step(
            action="RandomizeMaterials",
            useTrainMaterials=None,
            useValMaterials=None,
            useTestMaterials=None,
            inRoomTypes=None
        )
        self.controller.step(
            action="RandomizeLighting",
            brightness=(0.5, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False
        )
        self.controller.step(action="RandomizeColors")
        return self.controller

    def _choose_scene(self, scene: str):
        if scene: return scene
        return random.choice(SCENE_ROBOTHOR)

    def init_controller(self, scene=None, width=128, height=128, depth=True, target_object="Apple", randomize: bool=True, gridSize:float=0.25):
        """
        If scene = None, choose randomly from list of valid training scenes
        """
        self.scene = scene
        self.env_params = {
            "agentMode": "locobot",
            "visibilityDistance" : 1.5,
            "scene" : self._choose_scene(scene),
            "gridSize": gridSize,
            "snapToGrid": True,
            "rotateStepDegrees": 90,
            "renderDepthImage": depth,
            "renderInstanceSegmentation": False,
            "width": width,
            "height": height,
            "fieldOfView": 60,
            "rotateGaussianSigma": 0.5 if randomize else 0,
            "movementGaussianSigma": 0.005 if randomize else 0,
        }
        self.controller = Controller(
            **self.env_params
        )
        if randomize:
            self.randomize_controller()
        return self.controller


    def __init__(self, scene=None, width=128, height=128, depth=False, target_object="Apple", randomize: bool=True, gridSize:float=0.25):
        super().__init__()
        self.init_controller(
            scene=scene,
            width=width,
            height=height,
            depth=depth,
            target_object=target_object,
            randomize=randomize,
            gridSize=gridSize
        )
        self.randomize = randomize
        self.target_object=target_object
        assert target_object in TARGET_OBJECT_TYPES
        self.all_actions = [
            "MoveAhead",
            "MoveBack",
            "RotateRight",
            "RotateLeft",
            "LookUp",
            "LookDown",
            "Done"
        ]
        self.depth = depth
        self.action_space = gym.spaces.Discrete(len(self.all_actions))
        self.observation_space = gym.spaces.Box(shape=(width, height, int(depth) + 3), low=0, high=255, dtype=np.uint8)

    def get_action_id(self, action_str):
        return self.all_actions.index(action_str)

    def get_last_obs(self):
        return self._obs(self.controller.last_event)

    def _obs(self, event):
        obs = event.frame
        if self.depth:
            obs = np.concatenate((obs, event.depth_frame), -1)
        return obs

    def step(self, action):
        action = self.all_actions[action]
        event = self.controller.step(action = action)

        obs = self._obs(event)
        done = self.check_success(event.metadata)
        reward = float(done)
        return obs, reward, done, False, event.metadata

    def get_scene(self):
        return self.controller.last_event.metadata["sceneName"]

    def get_current_agent_state(self):
        metadata = self.controller.last_event.metadata["agent"]
        return (metadata["position"], metadata["rotation"], metadata["cameraHorizon"])

    def reset(self, scene=None, randomize=None):
        if scene:
            self.scene = scene
        if randomize:
            self.randomize = randomize
        self.controller.reset(scene=self._choose_scene(self.scene))
        if self.randomize:
            self.randomize_controller()
        event = self.controller.last_event
        return self._obs(event), event.metadata

    def check_success(self, metadata):
        return metadata["lastAction"] == "Done" and self.check_find_target(metadata)

    def check_find_target(self, metadata):
        for obj in metadata["objects"]:
            if obj["objectType"].lower() == self.target_object.lower():
                if obj["distance"] <= 1 and obj["visible"]:
                    return True
        return False

from gym.envs.registration import register

for obj in TARGET_OBJECT_TYPES:
    register(
        id=f"robothor-{obj.lower()}",
        entry_point=__name__ + ":AI2Thor",
        max_episode_steps=1000,
        kwargs={"target_object": obj, "height": 84, "width": 84}
    )


if __name__ == "__main__":
    env = gym.make("robothor-apple")
    print(env.reset())
    t = env.step(env.action_space.sample())
    print(len(t))
    print(t[0].shape)

    env = gym.make("robothor-apple", kwargs={"depth": True})
