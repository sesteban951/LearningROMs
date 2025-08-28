# brax imports
import functools
import jax

# standard imports
from datetime import datetime

from brax import envs
from brax.training.agents.ppo import train as ppo

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig

# for saving results
import pickle

# for logging
from tensorboardX import SummaryWriter

#################################################################

if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    print(jax.devices())

    ############################## ENVIRONMENT / LOGGING ##############################

    # Initialize the environment
    print("Initializing environment...")
    env = envs.get_environment("cart_pole")

    # A separate eval env is required if domain randomization is used
    eval_env = envs.get_environment("cart_pole")

    # Define a tensorboard logging callback
    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logdir = f"./rl/log/cart_pole_{current_datetime}"
    print(f"Logging to [{logdir}].")

    # Create a SummaryWriter for logging
    writer = SummaryWriter(logdir)
    times = [datetime.now()]

    # progress function for logging
    def progress(num_steps, metrics):
        
        # try to get eval reward if it exists
        reward = metrics.get("eval/episode_reward", None)
        if reward is not None:
            print(f"  Step: {num_steps}, Reward: {reward}")
        else:
            print(f"  Step: {num_steps}, Reward: N/A")

        times.append(datetime.now())

        # Write all metrics to tensorboard
        for key, val in metrics.items():

            # convert jax arrays to floats
            if isinstance(val, jax.Array):
                val = float(val)  # we need floats for logging

            # log to tensorboard
            writer.add_scalar(key, val, num_steps)

    ################################# TRAIN #################################

    # define hyperparameters in one place
    ppo_config = dict(
        num_timesteps=30_000_000,       # total training timesteps
        num_evals=10,                  # number of evaluations
        reward_scaling=0.1,            # reward scale
        episode_length=100,            # max episode length
        normalize_observations=True,   # normalize observations
        unroll_length=10,               # PPO unroll length
        num_minibatches=32,            # PPO minibatches
        num_updates_per_batch=8,       # PPO updates per batch
        discounting=0.97,              # gamma
        learning_rate=1e-3,            # optimizer LR
        clipping_epsilon=0.2,         # PPO clipping epsilon
        entropy_cost=1e-3,             # entropy bonus
        num_envs=2048,                  # parallel envs
        batch_size=1024,                # batch size
        seed=0,                        # RNG seed
    )

    # create the training function with the environment and hyperparameters
    train_fn = functools.partial(
        ppo.train,
        environment=env,
        **ppo_config
    )

    # train the  PPO agent
    #   - (make_policy) makes the policy function
    #   - (params) are the trained parameters
    #   - (metrics) final training metrics
    print("Training...")
    make_policy, params, metrics = train_fn(
        environment=env,
        progress_fn=progress,
    )

    ############################ RESULTS #################################

    # print timing info
    print("Training complete.")
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # save the trained policy (NOTE: make sure "/rl/policy" directory exists)
    print("Saving trained policy...")
    save_path = f"./rl/policy/cart_pole_policy_{current_datetime}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved trained policy to: {save_path}")
