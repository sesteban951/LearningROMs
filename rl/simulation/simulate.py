# for importing saved policies
import pickle

# jax imports
import jax

# brax imports
from brax.training.agents.ppo import make_inference_fn

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig

#################################################################

if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    print(jax.devices())

    # load the saved policy
    policy_path = "./rl/policy/cart_pole_policy_2025_08_26_22_45_17.pkl"
    with open(policy_path, "rb") as f:
        params = pickle.load(f)

    # creat the inference function
    inference_fn = make_inference_fn(CartPoleEnv)

