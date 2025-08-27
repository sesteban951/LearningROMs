# standard imports
import time

# jax imports
import jax

# brax improts
from brax import envs
from brax.training.agents.ppo.networks import make_inference_fn
from brax.training.agents.ppo import networks as ppo_networks

# for importing policy
import pickle

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig

#################################################################

if __name__ == "__main__":

    # Load environment
    env = envs.get_environment("cart_pole")

    # Load trained params
    # NOTE: change the path to your saved policy
    with open("./rl/policy/cart_pole_policy_2025_08_27_15_28_19.pkl", "rb") as f:
        params = pickle.load(f)

    # Rebuild the policy fn
    #   - In your training script, the first thing returned was `make_policy`
    #   - So we need to rebuild that the same way PPO does:

    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
    )
    
    print("Rebuilding policy function...")

    # Step 1: build fn factory
    inference_fn_factory = ppo_networks.make_inference_fn(networks)

    # Step 2: create an actual policy fn with trained params
    policy_fn = inference_fn_factory(params)

    print("Policy function built.")

    print("Jitting Functions.")
    # jit important functions
    policy_fn_jit = jax.jit(policy_fn)
    env_step_fn_jit = jax.jit(env.step)
    env_reset_jit = jax.jit(env.reset)

    print("Warm up...")
    # Warm up State
    key = jax.random.PRNGKey(0)
    state = env_reset_jit(rng=key)

    # Warm up policy and step
    # 2. Warm up policy & step multiple times
    for _ in range(5):   # a few steps is enough
        key, subkey = jax.random.split(key)
        act, _ = policy_fn_jit(state.obs, subkey)
        state = env_step_fn_jit(state, act)

    print("Running a single episode with the trained policy...")
    
    t0 = time.time()

    key  = jax.random.PRNGKey(0)
    state = env_reset_jit(rng=key)
    for step in range(10):
        print(f"Step {step}, Time {time.time() - t0} seconds")
        key, subkey = jax.random.split(key)
        act, _ = policy_fn_jit(state.obs, subkey)
        state = env_step_fn_jit(state, act)

    print(f"Total Time elapsed: {time.time() - t0} seconds.")

    print("Episode reward:", float(state.reward))