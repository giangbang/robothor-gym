# Robothor-gym

## RoboThor

install the following denpendencies, vulkan can be redundant but i have not checked it yet

```
apt --fix-broken install
apt install libvulkan1 mesa-vulkan-drivers vulkan-utils
apt update
apt upgrade
sudo apt install  pciutils
```

If error relating to X server with GLX occurs, try to run the above code several times might fix it. 

Finally, install the python requirement
```
git clone https://github.com/giangbang/robothor-gym
cd robothor_gym
pip install .
```
or directly from pip 
```
pip install git+https://github.com/giangbang/robothor-gym
```

The currently available environments are object navigation tasks, using mostly default parameters from `Robothor`.

### Observation space
Rgb image of the egocentric view of the robot, the robot has the field of view of 60 degree, can be changed in the code.
The environment support the depth mask of the agent observation, but it is disabled by default, set `depth=True` when creating the environment to enable

By default, the scene of the task is randomized at each reset call, with random material and rondom color texture of objects, more info can be found in [this doc](https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization).
To disable this randomization, set the `scene` to some specific scene, for example `FloorPlan_Train1_1`, see the code for the list of all available scenes or robothor documentation.
### Action space
The list of all actions are `MoveAhead`,
            `RotateRight`,
            `MoveLeft`,
            `MoveBack`,
            `LookUp`, 
            `LookDown`,
            `Done`.
Rotate actions rotate the robot's camera 90 degrees (can be changed). Move actions move the robot forward/backward a small distance depending on the `gridSize`, LookUp/Down shift the view of vertical angle camera 30 degree up/down

### Rewards
All environments are spare reward, agent receives reward = 1 when it finds the target object in the scene, each task requires the robot to find a specific object that is encoded in the name of the environment, for example `robothor-apple`. 

The success criteria is defined to be similar to the criteria in `Robothor` challenge, more detail can be found in  [this doc](https://ai2thor.allenai.org/robothor/documentation/#evaluation).
To summarize, a navigation episode is considered successful if both of the following criteria are met:
- The specified object category is within 1 meter (Euclidean distance) from the agent's camera, and the agent issues the STOP action, which indicates the termination of the episode.
- The object is visible from in the final action's frame.

### Code example
```python
import gym
import robothor_env # required to register new gym envs

env = gym.make("robothor-apple")
env.reset()  # unlike other gym env, reset is not really required in robothor, this step is only an abidance to gym API
n_env_step = 0
tot_reward = 0
while True:
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    n_env_step += 1
    tot_reward += reward
    if terminated or truncated:
        break
print(f"Total number of timestep: {n_env_step}")
print(f"Total reward: {tot_reward}")
```