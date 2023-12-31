import robothor_env
try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
import argparse


list_target_object = [
    "GarbageCan",
    "HousePlant",
    "Laptop",
    "Mug"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-obj", type=str, default=None,
        help="target object")
    parser.add_argument("--grid-size", type=float, default=0.25,
        help="grid size")
    parser.add_argument("--scene", type=str, default="FloorPlan_Train1_1",
        help="scene")
    parser.add_argument("--field-of-view", type=int, default=90,
        help="fieldOfView")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    env = gym.make("robothor-precompute")
    args = parse_args()

    if args.target_obj is not None:
        list_target_object = [args.target_obj]

    for obj in list_target_object:
        env.build_graph(target_object=obj, scene=args.scene, 
                gridSize=args.grid_size, fieldOfView=args.field_of_view)
        graph_file = f"robothor-{env.scene}-{env.target_obj}.pkl"
        env.save_graph(graph_file)

    n_step = 10000
    import time
    start_time = time.time()
    env.reset()
    for _ in range(n_step):
        env.step(env.action_space.sample())
    end_time = time.time()
    print("fps: {:.3f}".format(n_step / (end_time - start_time)))
