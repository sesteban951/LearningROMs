# brax imports
import functools
import jax

from datetime import datetime
from jax import numpy as jp

from brax import envs
from brax.training.agents.ppo import train as ppo

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig

# for saving results
import pickle

#################################################################

if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    print(jax.devices())

    ############################## ENVIRONMENT / CONFIG ##############################

    # create the environment
    env = CartPoleEnv(CartPoleConfig())

    # define hyperparameters in one place
    ppo_config = dict(
        # num_timesteps=60_000_000,       # total training timesteps
        num_timesteps=10_000_000,       # total training timesteps
        num_evals=10,                  # number of evaluations
        reward_scaling=0.1,            # reward scale
        episode_length=100,            # max episode length
        normalize_observations=True,   # normalize observations
        # action_repeat=1,               # action repeats
        unroll_length=10,               # PPO unroll length
        num_minibatches=32,            # PPO minibatches
        num_updates_per_batch=8,       # PPO updates per batch
        discounting=0.97,              # gamma
        learning_rate=1e-3,            # optimizer LR
        clipping_epsilon=0.2,         # PPO clipping epsilon
        entropy_cost=1e-3,             # entropy bonus
        num_envs=2048,                  # parallel envs
        batch_size=1024,                # batch size
        seed=1,                        # RNG seed
        log_training_metrics=True      # log metrics
    )

    ################################# TRAIN #################################

    # make the train function with env + config
    train_fn = functools.partial(
        ppo.train,
        environment=env,
        **ppo_config
    )


    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())

        # prefer total reward if it exists
        if "reward_total" in metrics:
            reward = metrics["reward_total"]
            tag = "train"
        elif "eval/episode_reward" in metrics:
            reward = metrics["eval/episode_reward"]
            tag = "eval"
        else:
            # sum all reward-related metrics as a fallback
            reward_keys = [k for k in metrics if "reward" in k]
            if reward_keys:
                reward = sum(metrics[k] for k in reward_keys)
                tag = "train-fallback"
            else:
                reward, tag = 0.0, "none"

        print(f"[{tag}] step {num_steps:,}: reward={reward:.2f}")

    # train
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress,
    )

    ############################ RESULTS #################################

    # print timing info
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # print all the metrics
    for key, val in metrics.items():
        print(f"{key}: {val}")

    # save the trained policy
    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"./rl/policy/cart_pole_policy_{current_datetime}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(params, f)

    print(f"Saved trained policy to: {file_name}")