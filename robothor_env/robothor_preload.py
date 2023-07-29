import os
import numpy as np
import random

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

import copy


class AI2Thor_Preload(gym.Env):
    # taken from https://allenact.org/tutorials/training-a-pointnav-model/
    REWARD_CONFIG = {
        "step_penalty": -0.01,
        "goal_success_reward": 10.0,
        "failed_stop_reward": 0.0, # not used
        "shaping_weight": 1.0,
    }

    def __init__(self, precompute_file=None):
        super().__init__()
        self.graph = EnvGraph()
        if precompute_file:
            self.load_graph(precompute_file)

    @property
    def action_space(self):
        return self.graph.action_space

    @property
    def observation_space(self):
        return self.graph.observation_space

    @property
    def scene(self):
        return self.graph.scene

    @property
    def target_obj(self):
        return self.graph.target_obj

    def screenshot(self, graph, filename=None, env = None):
        if env is None:
            import robothor_env
            env = gym.make("robothor-"+graph.target_obj.lower(), scene=graph.scene, width=256, height=132, randomize=False)

        event = env.controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=5.5, y=15, z=-2.65),
            rotation=dict(x=90, y=0, z=0),
            fieldOfView=20
        )
        center_grid = np.array([5.5, -2.65])

        graph.raw_screenshot = event.third_party_camera_frames[0].astype(np.uint8).copy() 
        all_points = list(map(lambda x : np.array(x[0])/10000., graph.obs.keys()))
        terminal_points = list(map(lambda x : np.array(x[0])/10000., graph.terminal))
        
        def transform(p):
            return np.abs(np.array([(p[0]-0.4) * 265/2/5.3, (p[2]+0.15) * 140/2/2.61])).astype(np.int)

        import cv2
        for point in all_points:
            graph.raw_screenshot = cv2.circle(graph.raw_screenshot, transform(point), radius=1, color=(0, 0, 255), thickness=-1)
        for point in terminal_points:
            graph.raw_screenshot = cv2.circle(graph.raw_screenshot, transform(point), radius=1, color=(255, 0, 0), thickness=-1)

        if filename is None:
            filename = f"{graph.scene}_{graph.target_obj}.png"
        cv2.imwrite(filename, cv2.cvtColor(graph.raw_screenshot, cv2.COLOR_RGB2BGR))

    def build_graph(self, target_object="Apple", **kwargs):
        import robothor_env
        from tqdm import tqdm

        self.graph = EnvGraph()
        kwargs.update(randomize=False)
        env = gym.make("robothor-"+target_object.lower(), **kwargs)
        print("Scene:", env.get_scene())

        self.graph.action_space = copy.deepcopy(env.action_space)
        self.graph.observation_space = copy.deepcopy(env.observation_space)
        self.graph.scene = copy.deepcopy(env.get_scene())
        self.graph.target_obj = target_object.lower()
        self.graph.env_params = copy.deepcopy(env.env_params)
        self.graph.init_v = env.get_current_agent_state()

        print("Doing breath first search on the environment...")
        vertices = breath_first_search(env, self.graph)
        with open('agent_states.txt', 'w') as f:
            f.write("\n".join(map(str, vertices)))
        print("Done.")

        print("Total number of vertices:", len(self.graph.obs))

        print("Finding shortest all vertices distances...")
        self.graph.calculate_shortest_distance_to_goal()
        print("Done.")

        self.screenshot(self.graph)

        return self.graph

    def load_graph(self, path_to_save_file: str):
        self.graph = EnvGraph()
        self.graph.load(path_to_save_file)

    def save_graph(self, path_to_save_file: str):
        self.graph.save(path_to_save_file)

    def reset(self, seed=None, **kwargs):
        self.current_v = self.graph.init_v
        return self.graph.get_obs(self.current_v), {}

    def step(self, action):
        self.current_v = self.graph.next_vertex(self.current_v, action)
        done = action == self.action_space.n - 1 and self.graph.check_terminal(self.current_v)
        reward = self._reward(self.current_v, done)
        return self.graph.get_obs(self.current_v), reward, done, False, {}

    def render(self, render_mode="rgb_array"):
        return self.graph.get_obs(self.current_v)

    def _reward(self, current_vertex, done):
        reward = self.REWARD_CONFIG["step_penalty"] if not done else self.REWARD_CONFIG["goal_success_reward"]
        reward -= self.graph.get_distance_to_goal(current_vertex) * \
                self.REWARD_CONFIG["shaping_weight"] * self.graph.env_params["gridSize"]
        return reward

class EnvGraph:
    def __init__(self):
        self.adj = {}
        self.obs = {}
        self.terminal = set()
        # dummy spaces
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)
        self.init_v = None
        self.scene = None
        self.target_obj = None
        self.env_params = None
        self.distances = None

    def get_obs(self, vertex):
        vertex = self.convert_vertex(vertex)
        return self.obs[vertex]

    def get_distance_to_goal(self, vertex):
        vertex = self.convert_vertex(vertex)
        return self.distances[vertex]

    def calculate_shortest_distance_to_goal(self):
        print("Finding shortest distances from each state to goal...")
        from collections import deque
        q = deque()
        self.distances = dict()
        assert self.terminal is not None

        q.extend(self.terminal)
        current_distance = 0
        while q:
            q_size = len(q)
            for _ in range(q_size):
                v = q.popleft()
                self.distances[v] = current_distance
                for nei in self.adj[v].values():
                    nei = self.convert_vertex(nei)
                    if nei not in self.distances:
                        self.distances[nei] = 0 # explored, in queue, overrided later
                        q.append(nei)
            current_distance += 1
        print("Done.")

    def next_vertex(self, vertex, edge):
        vertex = self.convert_vertex(vertex)
        return self.adj[vertex][edge]

    def add_vertex(self, vertex, obs):
        vertex = self.convert_vertex(vertex)
        assert vertex not in self.obs
        self.adj[vertex] = dict()
        self.obs[vertex] = obs.astype(np.uint8)

    def add_terminal(self, vertex):
        vertex = self.convert_vertex(vertex)
        self.terminal.add(vertex)

    def check_terminal(self, vertex):
        vertex = self.convert_vertex(vertex)
        return vertex in self.terminal

    def add_edge(self, v1, v2, edgetype):
        v1 = self.convert_vertex(v1)
        self.adj[v1][edgetype] = v2

    def save(self, path):
        with open(path, 'wb') as outp:
            pickle.dump(self.__dict__, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as inp:
            tmp_dict = pickle.load(inp)
        self.__dict__.update(tmp_dict)
        # resolve conflicting library
        # converting between gym.spaces and gymnasium.spaces and vice versa
        self.action_space = gym.spaces.Discrete(self.action_space.n)
        self.observation_space = gym.spaces.Box(low=self.observation_space.low,
                                                high=self.observation_space.high)
        # calculate the distances, if not available
        if self.distances is None or len(self.distances) == 0:
            print("Precompute shortest distances are missing in the saved file.")
            self.calculate_shortest_distance_to_goal()
        return self

    def __contains__(self, vertex):
        vertex = self.convert_vertex(vertex)
        return vertex in self.obs

    def convert_vertex(self, vertex):
        vertex[1]["y"] = int(round(vertex[1]["y"] / 90) * 90) % 360
        t = (tuple(vertex[0].values()), tuple(map(int, vertex[1].values())), (round(vertex[2]/30)*30,))
        return tuple(tuple(map(lambda x : int(x*10000), tup)) for tup in t)

    def __len__(self):
        return len(self.obs)

def breath_first_search(env, graph):
    """
    Perform dfs on the robothor environment, return a list of all possible ```(position, rotation, horizon)```
    """
    # reset env to the current scene, if `scene` is None, then
    # the env will reset to other random scene
    env.reset(scene=env.get_scene(), randomize=False)
    actions = range(len(env.all_actions))
    res = []

    from collections import deque
    q = deque()
    q.append(env.get_current_agent_state())
    graph.add_vertex(env.get_current_agent_state(), env.get_last_obs())

    while len(q):
        v = q.popleft()
        res.append(v)
        for action in actions:
            env.controller.step(
                action="Teleport",
                position=v[0],
                rotation=v[1],
                horizon=v[2]
            )
            env.step(action)
            pos, rot, hor = env.get_current_agent_state()
            next  = (pos, rot, hor)
            graph.add_edge(v, next, action)
            if env.check_success(env.controller.last_event.metadata):
                graph.add_terminal(v) # v and next are the same, since "Done" does not alter env state
                graph.add_terminal(next)
            if next in graph:
                continue
            q.append(next)
            graph.add_vertex(next, env.get_last_obs())
    return res

gym.envs.registration.register(
    id="robothor-precompute",
    entry_point=__name__ + ":AI2Thor_Preload",
    max_episode_steps=500,
)

if __name__ == "__main__":
    env = gym.make("robothor-precompute")

    import time
    start = time.time()
    # change gridSize to 1 for faster excution
    # in practice, set this to 0.25
    # the total time to build graph in this case is around 1 hour
    env.build_graph(gridSize=1)
    end = time.time()
    print("total time to build graph: {:.3f}s".format(end-start))
    env.reset()

    n_step = 1000
    start_time = time.time()
    for _ in range(n_step):
        obs, reward, terminate, truncate, info = env.step(env.action_space.sample())
        if terminate or truncate:
            env.reset()
    end_time = time.time()
    print("fps: {:.3f}".format(n_step / (end_time - start_time)))

    graph_file = f"robothor-{env.scene}-{env.target_obj}.pkl"
    env.save_graph(graph_file)

    del env

    env = gym.make("robothor-precompute")
    env.load_graph(graph_file)
    env.reset()
    start_time = time.time()
    for _ in range(n_step):
        obs, reward, terminate, truncate, info = env.step(env.action_space.sample())
        if terminate or truncate:
            env.reset()
    end_time = time.time()
    print("fps: {:.3f}".format(n_step / (end_time - start_time)))
