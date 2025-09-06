##
#
#  Simple Training Script for BRAX environments
#  NOTE: For some reason, my desktop (4090 GPU) is better at training.  
#        and produces better policies for the same hyperparameters.
#
##

# jax imports
import jax

# brax imports
from brax import envs

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig
from envs.acrobot_env import AcrobotEnv, AcrobotConfig
from envs.hotdog_man_env import HotdogManEnv, HotdogManConfig
from envs.hopper_env import HopperEnv, HopperConfig
from algorithms.ppo_train import PPO_Train

if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    print("JAX is using device:", jax.devices()[0])

    #----------------------- SETUP -----------------------#

    # Initialize the environment
    # env = envs.get_environment("cart_pole")
    
    # # define hyperparameters in one place
    # ppo_config = dict(
    #     num_timesteps=60_000_000,      # total training timesteps
    #     num_evals=10,                  # number of evaluations
    #     reward_scaling=0.1,            # reward scale
    #     episode_length=150,            # max episode length
    #     normalize_observations=True,   # normalize observations
    #     unroll_length=10,              # PPO unroll length
    #     num_minibatches=32,            # PPO minibatches
    #     num_updates_per_batch=8,       # PPO updates per batch
    #     discounting=0.97,              # gamma
    #     learning_rate=1e-3,            # optimizer LR
    #     clipping_epsilon=0.2,          # PPO clipping epsilon
    #     entropy_cost=1e-3,             # entropy bonus
    #     num_envs=1024,                 # parallel envs
    #     batch_size=1024,               # batch size
    #     seed=0,                        # RNG seed
    # )

    # # Initialize the environment
    # env = envs.get_environment("acrobot")

    # # define hyperparameters in one place
    # ppo_config = dict(
    #     num_timesteps=60_000_000,      # total training timesteps
    #     num_evals=10,                  # number of evaluations
    #     reward_scaling=0.1,            # reward scale
    #     episode_length=300,            # max episode length
    #     normalize_observations=True,   # normalize observations
    #     unroll_length=10,              # PPO unroll length
    #     num_minibatches=32,            # PPO minibatches
    #     num_updates_per_batch=8,       # PPO updates per batch
    #     discounting=0.97,              # gamma
    #     learning_rate=5e-4,            # optimizer LR
    #     clipping_epsilon=0.2,          # PPO clipping epsilon
    #     entropy_cost=3e-4,             # entropy bonus
    #     num_envs=2048,                 # parallel envs
    #     batch_size=2048,               # batch size
    #     seed=0,                        # RNG seed
    # )

    # # Initialize the environment
    # env = envs.get_environment("hotdog_man")

    # # define hyperparameters in one place
    # ppo_config = dict(
    #     num_timesteps=100_000_000,      # total training timesteps
    #     num_evals=10,                  # number of evaluations
    #     reward_scaling=0.1,            # reward scale
    #     episode_length=200,            # max episode length
    #     normalize_observations=True,   # normalize observations
    #     unroll_length=10,              # PPO unroll length
    #     num_minibatches=32,            # PPO minibatches
    #     num_updates_per_batch=8,       # PPO updates per batch
    #     discounting=0.97,              # gamma
    #     learning_rate=5e-4,            # optimizer LR
    #     clipping_epsilon=0.2,          # PPO clipping epsilon
    #     entropy_cost=3e-4,             # entropy bonus
    #     num_envs=4096,                 # parallel envs
    #     batch_size=4096,               # batch size
    #     seed=0,                        # RNG seed
    # )

    # Initialize the environment
    env = envs.get_environment("hopper")

    # define hyperparameters in one place
    ppo_config = dict(
        num_timesteps=40_000_000,      # total training timesteps
        num_evals=10,                  # number of evaluations
        reward_scaling=0.1,            # reward scale
        episode_length=300,            # max episode length
        normalize_observations=True,   # normalize observations
        unroll_length=10,              # PPO unroll length
        num_minibatches=64,            # PPO minibatches
        num_updates_per_batch=8,       # PPO updates per batch
        discounting=0.97,              # gamma
        learning_rate=5e-4,            # optimizer LR
        clipping_epsilon=0.2,          # PPO clipping epsilon
        entropy_cost=3e-4,             # entropy bonus
        num_envs=4096,                 # parallel envs
        batch_size=4096,               # batch size
        seed=0,                        # RNG seed
    )
    
    #----------------------- TRAIN -----------------------#

    # Create PPO training instance
    ppo_trainer = PPO_Train(env, ppo_config)

    # start training
    ppo_trainer.train()