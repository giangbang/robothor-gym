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
        "shaping_weight": .1,
    }

    def __init__(self, 
        precompute_file=None, 
        reward_shaping:bool=True, 
        random_start=True, 
        **kwargs
    ):
        super().__init__()
        self.graph = EnvGraph()
        self.reward_shaping = reward_shaping
        self.random_start = random_start
        self.all_vertices = None # cached all vertices in a list
        if precompute_file:
            self.load_graph(precompute_file)

        self.step_count = 0

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

    def get_scene(self):
        return self.scene

    def screenshot(self, graph, filename=None):
        w, h = 512, 256
        import robothor_env
        env = gym.make("robothor-"+graph.target_obj.lower(), scene=graph.scene, 
                width=w, height=h, randomize=False)

        event = env.controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=5.5, y=15, z=-2.65),
            rotation=dict(x=90, y=0, z=0),
            fieldOfView=20
        )

        graph.raw_screenshot = event.third_party_camera_frames[0].astype(np.uint8).copy() 
        all_points = list(map(lambda x : np.array(x[0])/10000., graph.obs.keys()))
        terminal_points = list(map(lambda x : np.array(x[0])/10000., graph.terminal))
        
        def transform(p):
            return np.abs(np.array([(p[0]-0.3) * w/2/5.2, 
                    (p[2]+0.1) * h/2/2.5])).astype(int)

        import cv2
        graph.screenshot = graph.raw_screenshot
        r = round(w/256.)
        for point in all_points:
            graph.screenshot = cv2.circle(graph.screenshot, 
                    transform(point), radius=r, color=(0, 0, 255), thickness=-1)
        for point in terminal_points:
            graph.screenshot = cv2.circle(graph.screenshot, 
                    transform(point), radius=r, color=(255, 0, 0), thickness=-1)

        if filename is None:
            filename = f"{graph.scene}_{graph.target_obj}.png"
        cv2.imwrite(filename, cv2.cvtColor(graph.screenshot, cv2.COLOR_RGB2BGR))

    def build_graph(self, target_object="Apple", **kwargs):
        import robothor_env

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

        if self.graph.terminal:
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

    def reset(self, seed=None, reward_shaping=None, **kwargs):
        self.step_count = 0
        self.prev_v = None
        if self.random_start:
            if self.all_vertices is None:
                self.all_vertices = list(map(
                    lambda v : self.graph.inverse_vertex(v), self.graph.obs.keys()
                ))
            self.current_v = random.choice(self.all_vertices)
            # set camera horizon to 0, agent always look directly ahead
            # when starting a new episode
            self.current_v = (*self.current_v[:-1], 0)
        else:
            self.current_v = self.graph.init_v
        if reward_shaping is not None:
            self.reward_shaping = reward_shaping
        return self.graph.get_obs(self.current_v), {"is_success": False}

    def step(self, action, reward_shaping = None):
        self.step_count += 1
        self.prev_v = self.current_v
        self.current_v = self.graph.next_vertex(self.current_v, action)
        done = action == self.action_space.n - 1 and self.graph.check_terminal(self.current_v)
        reward = self._reward(self.prev_v, self.current_v, done, reward_shaping \
                if reward_shaping is not None else self.reward_shaping)
        return self.graph.get_obs(self.current_v), reward, done, False, {"is_success": done}

    def render(self, render_mode="rgb_array"):
        return self.graph.get_obs(self.current_v)

    def _reward(self, prev_vertex, current_vertex, done, reward_shaping):
        reward = self.REWARD_CONFIG["step_penalty"] \
                if not done else self.REWARD_CONFIG["goal_success_reward"]
        if reward_shaping and self.graph.distances:
            reward += self._reward_shaping_2(prev_vertex, current_vertex)
        return reward

    def _reward_shaping_1(self, current_v, **kwargs):
        """Reward shaping = minus distance to goal"""
        return -self.graph.get_distance_to_goal(current_vertex) * \
                    self.REWARD_CONFIG["shaping_weight"] * self.graph.env_params["gridSize"]

    def _reward_shaping_2(self, prev_v, current_v):
        """Reward shaping = changes in the distance to goal"""
        return (self.graph.get_distance_to_goal(current_v) - 
                self.graph.get_distance_to_goal(prev_v)) * \
                self.REWARD_CONFIG["shaping_weight"] * self.graph.env_params["gridSize"]

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
        assert self.terminal, "Empty termination states."

        q.extend(self.terminal)
        current_distance = 0
        while len(q) > 0:
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
        assert len(self.distances) == len(self), f"{len(self.distances)}, {len(self)}"
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
                                                high=self.observation_space.high, 
                                                dtype=np.uint8)
        # calculate the distances, if not available
        if self.distances is None or len(self.distances) == 0:
            print("Precompute shortest distances are missing in the saved file.")
            self.calculate_shortest_distance_to_goal()
        return self

    def __contains__(self, vertex):
        vertex = self.convert_vertex(vertex)
        return vertex in self.obs

    def convert_vertex(self, vertex):
        """Convert actual positions of the robot into hashable (immutable) objects."""
        rotate_degree = self.env_params["rotateStepDegrees"]
        vertex[1]["y"] = int(round(vertex[1]["y"]/rotate_degree)*rotate_degree) % 360
        t = (tuple(vertex[0].values()), tuple(map(int, vertex[1].values())), \
                (round(vertex[2]/30)*30,))
        return tuple(tuple(map(lambda x : int(x*10000), tup)) for tup in t)

    def inverse_vertex(self, vertex):
        """Inverse function of ``convert_vertex`` """
        ret = list(tuple(map(lambda x : x/10000, tup)) for tup in vertex)
        ret[-1] = ret[-1][0]
        for i in range(2):
            ret[i] = dict(zip(["x", "y", "z"], ret[i]))
        ret = tuple(ret)
        return ret

    def __len__(self):
        return len(self.obs)

def breath_first_search(env, graph):
    """
    Perform dfs on the robothor environment, return a list of all possible 
    ```(position, rotation, horizon)```
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
    
    env = env.unwrapped
    assert type(env) is not gym.wrappers.TimeLimit

    while len(q) > 0:
        v = q.popleft()
        res.append(v)
        for action in actions:
            env.controller.step(
                action="Teleport",
                position=v[0],
                rotation=v[1],
                horizon=np.clip(v[2], a_min=-29.9, a_max=59.9)
            )

            step_result = env.step(action)
            obs, metadata = step_result[::len(step_result)-1]

            pos, rot, hor = env.get_current_agent_state(metadata)
            next_v = (pos, rot, hor)
            graph.add_edge(v, next_v, action)
            if env.check_success(metadata):
                graph.add_terminal(v) # v and next are the same, since "Done" does not alter env state
                graph.add_terminal(next_v)
            if next_v not in graph:
                q.append(next_v)
                graph.add_vertex(next_v, obs)

    if not graph.terminal:
        print("WARN: Empty terminal states. Decrease the GridSize")
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
