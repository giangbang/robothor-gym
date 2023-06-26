import os
import ai2thor
import ai2thor_colab
import numpy as np
import random

from ai2thor.controller import Controller
import gym
import ai2thor_colab
from ai2thor.util.metrics import (
    path_distance
)
ai2thor_colab.start_xserver()

version = "AI2-THOR Version: " + ai2thor.__version__
print(version)


# more information at https://ai2thor.allenai.org/robothor/documentation/#initialization-scene
SCENE_ROBOTHOR = [
    f"FloorPlan_Train{i}_{j}" 
    for i in range(1, 12, 1)
    for j in range(1, 5, 1)
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
        self.controller.controller.step(
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

    def init_controller(self, scene=None, width=128, height=128, depth=True, target_object="Apple"):
        """
        If scene = None, choose randomly from list of valid training scenes
        """
        self.scene = scene
        self.env_params = {
            "agentMode": "locobot",
            "visibilityDistance" : 1.5,
            "scene" : self._choose_scene(scene),
            "gridSize": .25,
            "snapToGrid": False,
            "rotateStepDegrees": 90,
            "renderDepthImage": depth,
            "renderInstanceSegmentation": False,
            "width": width,
            "height": height,
            "fieldOfView": 60,
            "rotateGaussianSigma": 0.5,
            "movementGaussianSigma": 0.005,
        }
        if self.env_params is None:
            self.env_params = 
        self.controller = Controller(
            **self.env_params
        )
        self.randomize_controller()
        return self.controller


    def __init__(self, scene=None, width=128, height=128, depth=False, target_object="Apple"):
        self.init_controller(
            scene=scene,
            width=width,
            height=height,
            depth=depth,
            target_object=target_object
        )
        self.target_object=target_object
        assert target_object in TARGET_OBJECT_TYPES
        self.all_actions = [
            "MoveAhead",
            "RotateRight",
            "MoveLeft",
            "MoveBack",
            "LookUp", 
            "LookDown",
            "Done"
        ]
        self.depth = depth
        self.action_space = gym.spaces.Categorical(len(self.all_actions))
        self.observation_space = gym.spaces.Box(shape=(width, height, int(depth) + 1), low=0, high=255, dtype=np.uint8)

        from sklearn import preprocessing
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.all_actions)

    def _obs(self, event):
        obs = event.frame
        if self.depth:
            obs = np.concatenate((obs, event.depth_frame), -1)
        return obs

    def step(self, action):
        action = self.le.inverse_transform(action)
        event = self.controller.step(action = action).last_event

        obs = self._obs(event)
        done = check_success(event.metadata)
        reward = float(done)
        return obs, reward, done, False, {}
    
    def reset(self):
        self.controller.reset(scene=self._choose_scene(self.scene))
        event = self.randomize_controller().last_event
        return self._obs(event), {}

    def check_success(self, metadata):
        return metadata["lastAction"] == "Done" and self.check_find_target(metadata)

    def check_find_target(self, metadata):
        for obj in metadata["objects"]:
            if lower(obj["objectType"]) == lower(self.target_object):
                if obj["distance"] < 1 and obj["visible"]:
                    return True
        return False

from gym.envs.registration import register

for obj in TARGET_OBJECT_TYPES:
    register(
        id=f"robothor-{lower(obj)}",
        entry_point=__name__ + ":AI2Thor",
        max_episode_steps=1000,
        kwargs={"target_object": obj}
    )
