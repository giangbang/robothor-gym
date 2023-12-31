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
SETUP_ROBOTHOR=1 pip install .
```
or directly from pip
```
SETUP_ROBOTHOR=1 pip install git+https://github.com/giangbang/robothor-gym
```
`SETUP_ROBOTHOR=1` will create a virtual display and connect it to the display output of `robothor`. If you do not want to set this up (e.g when using precomputed environments only), omit this variable.

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


In the precomputed environments, the reward is similar to [pointnav env](https://allenact.org/tutorials/training-a-pointnav-model/#config-file-setup), with additional reward signal from the shortest distance from each point in the scene.
However, instead of the geodesic distance in meter, the number of (bfs) steps needed to reach goal is used instead.

### Code example
```python
import gym # if you have gymnasium, priotize using gymnasium
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

## Precompute environment

Since rendering frames in `robothor` takes a _long time_, a pre-rendered version of this environment is provided in `robothor_preload.py`. In this version, all the states of the environment is visited by brute force and all the observations are cached, a graph of the underlying dynamic is also built. At training time, we simply output the cached image observations. In this way, raw performance can reach about 60k fps on Google Colab (compared to 15fps running simulation on the same machine).
```python
import robothor_env
import gym

env = gym.make("robothor-precompute")
env.build_graph(scene=None) # build the graph takes roughly 1 hour on Google Colab
env.save_graph("graph.pkl")

del env

env = gym.make("robothor-precompute", precompute_file="graph.pkl", \
                random_start=True) # starting the episode at a random position
# or the graph can be loaded by using `env.load_graph("graph.pkl")`
tot_reward=0
while True:
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    tot_reward += reward
    if terminated or truncated:
        break

print(f"Total reward: {tot_reward}")
```
or run the example script with the target object in argument
```bash
python example/generate_graph.py --target-obj Mug
```
Examples of the pre-built graph files can be downloaded from this [kaggle dataset](https://www.kaggle.com/datasets/banggiangle/robothor-graph-files).
Using precomputed files, we can gain access to the (precomputed) shortest distances from each state to the goal states, and use them to provide more instructive reward signal.
![Position Grid](./images/grids.png)
## Manual control
Run the script
```
python -m robothor_env.manual_control --env-id robothor-apple
```
To open a new window that allow user input from keyboard. 
Manually control the robot by pressing `up`, `down`, `left`, and `right` buttons to rotate and `s`, `w` to move backward/forward, respectively.

## RL Training
LSTM-PPO with stable-baselines3
```
python ./example/sb3_train.py --precompute-file <precompute-file-path>
```
Using the precompute environments (need precomputed graph path). Training PPO from stable-baselines3 converge after around 150k environment steps (about 30 minutes of training on Kaggle, with peak performance about 320 fps).
![PPO result](./images/ppo_sb3_results.png)
The [link](https://www.kaggle.com/code/banggiangle/robothor-with-stable-baselines3) to the training notebook on kaggle.
