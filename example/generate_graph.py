import robothor_env
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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    env = gym.make("robothor-precompute")
    args = parse_args()

    if args.target_obj is not None:
        list_target_object = [args.target_obj]

    for obj in list_target_object:
        env.build_graph(target_object=obj)
        graph_file = f"robothor-{env.scene}-{env.target_obj}.pkl"
        env.save_graph(graph_file)
