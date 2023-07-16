try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import os
import numpy as np
import random
import gym
import copy


class AI2Thor_Preload(gym.Env):
    def __init__(self, precompute_file=None):
        super().__init__()
        self.graph = None
        if precompute_file:
            self.load_graph(precompute_file)

    def build_graph(self, target_object="Apple", **kwargs):
        import robothor_env
        self.graph = EnvGraph()
        kwargs.update(randomize=False)
        env = gym.make("robothor-"+target_object, kwargs=kwargs)

        self.graph.action_space = copy.deepcopy(env.action_space)
        self.graph.observation_space = copy.deepcopy(env.observation_space)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # get positions (x, y, z)
        positions = controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]

        rotations = [dict(x=0, y=i*90, z=0) for i in range(4)]
        horizons = [-30, 0, 30]
        self.graph.init_v = (env.controller.last_event.metadata.agent["location"],
            env.controller.last_event.metadata.agent["rotation"],
            env.controller.last_event.metadata.agent["cameraHorizon"])

        # get rotations and camera angle of each position
        print("Extract total of", len(positions), "positions.")
        for position in positions:
            for rotation in rotations:
                for horizon in horizons:
                    env.controller.step(
                        action="Teleport",
                        position=position,
                        rotation=rotation,
                        horizon=horizon
                    )
                    observation = env._obs(env.controller.last_event)
                    self.graph.add_vertex((position, rotation, horizon), observation)

        for v in self.graph:
            for action in range(len(env.all_actions)):
                env.controller.step(
                    action="Teleport",
                    position=v[0],
                    rotation=v[1],
                    horizon=v[2]
                )
                ..., terminate, truncate, metadata = env.step(action)
                pos = metadata["agent"]["position"]
                rot = metadata["agent"]["rotation"]
                hor = metadata["agent"]["cameraHorizon"]
                v2  = (pos, rot, hor)
                assert v2 in self.graph

                self.graph.add_edge(v, v2, action)
                if terminate or truncate:
                    self.graph.add_terminal(v)

        return self.graph

    def load_graph(self, path_to_save_file: str):
        self.graph = EnvGraph().load(path_to_save_file)
        self.action_space = self.graph.action_space
        self.observation_space = self.graph.observation_space

    def save_graph(self, path_to_save_file: str):
        self.graph.save(path_to_save_file)

    def reset(self):
        self.current_v = self.graph.init_v
        return self.graph.obs[self.current_v], {}

    def step(self, action):
        self.current_v = self.graph.adj[self.current_v][action]
        done = action == self.action_space.n - 1 and self.current_v in self.graph.terminal
        return self.graph.obs[self.current_v], float(done), done, False, {}

    def render(self, render_mode="rgb_array"):
        pass

class EnvGraph:
    def __init__(self):
        self.adj = {}
        self.obs = {}
        self.terminal = set()
        self.observation_space = None
        self.action_space = None
        self.init_v = None

    def add_vertex(self, vertex, obs):
        self.adj[vertex] = dict()
        self.obs[vertex] = obs

    def add_terminal(self, vertex):
        self.terminal.add(vertex)

    def check_terminal(self, vertex):
        return vertex in self.terminal

    def add_edge(self, v1, v2, edgetype):
        self.adj[v1][edgetype].add(v2)

    def save(self, path):
        with open(path, 'wb') as outp:
            pickle.dump(self.__dict__, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as inp:
            tmp_dict = pickle.load(inp)
        self.__dict__.update(tmp_dict)

    def __contains__(self, vertex):
        return vertex in self.obs

    def __iter__(self):
        return self.obs.keys()


from gym.envs.registration import register
register(
    id="robothor-precompute",
    entry_point=__name__ + ":AI2Thor_Preload",
    max_episode_steps=1000,
)

if __name__ == "__main__":
    env = gym.make("robothor-precompute")

    import time
    start = time.time()
    env.build_graph()
    end = time.time
    print("total time to build graph:", end-start)

    n_step = 1000
    start = time.time()
    for _ in range(n_step):
        obs, reward, terminate, truncate, info = env.step(env.action_space.sample())
        if terminate or truncate:
            env.reset()
    end = time.time()
    print("fps:", n_step / (end - start))
