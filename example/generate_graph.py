import robothor_env
import gym


list_target_object = [
    "GarbageCan",
    # "HousePlant",
    # "Laptop",
    # "Mug"
]

if __name__ == "__main__":
    env = gym.make("robothor-precompute")
    for obj in list_target_object:
        env.build_graph(target_object=obj)
        graph_file = f"robothor-{env.scene}-{env.target_obj}.pkl"
        env.save_graph(graph_file)
