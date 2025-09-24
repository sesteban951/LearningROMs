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
from envs.biped_env import BipedEnv, BipedConfig
from envs.biped_basic_env import BipedBasicEnv, BipedBasicConfig
from envs.hopper_env import HopperEnv, HopperConfig
from envs.paddle_ball_env import PaddleBallEnv, PaddleBallConfig
from algorithms.ppo_train import PPO_Train

if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    print("JAX is using device:", jax.devices()[0])

    #----------------------- SETUP -----------------------#

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("cart_pole")
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

    # # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("acrobot")
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

    # # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("paddle_ball")
    # ppo_config = dict(
    #     num_timesteps=30_000_000,      # total training timesteps
    #     num_evals=10,                  # number of evaluations
    #     reward_scaling=0.1,            # reward scale
    #     episode_length=500,            # max episode length
    #     normalize_observations=True,   # normalize observations
    #     unroll_length=5,              # PPO unroll length
    #     num_minibatches=32,            # PPO minibatches
    #     num_updates_per_batch=8,       # PPO updates per batch
    #     discounting=0.97,              # gamma
    #     learning_rate=5e-4,            # optimizer LR
    #     clipping_epsilon=0.2,          # PPO clipping epsilon
    #     entropy_cost=1e-3,             # entropy bonus
    #     num_envs=2048,                 # parallel envs
    #     batch_size=2048,               # batch size
    #     seed=0,                        # RNG seed
    # )

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("hopper")
    # ppo_config = dict(
    #     num_timesteps=25_000_000,      # total training timesteps
    #     num_evals=10,                  # number of evaluations
    #     reward_scaling=0.1,            # reward scale
    #     episode_length=600,            # max episode length
    #     normalize_observations=True,   # normalize observations
    #     unroll_length=5,              # PPO unroll length
    #     num_minibatches=64,            # PPO minibatches
    #     num_updates_per_batch=8,       # PPO updates per batch
    #     discounting=0.97,              # gamma
    #     learning_rate=5e-4,            # optimizer LR
    #     clipping_epsilon=0.2,          # PPO clipping epsilon
    #     entropy_cost=3e-4,             # entropy bonus
    #     num_envs=4096,                 # parallel envs
    #     batch_size=4096,               # batch size
    #     seed=0,                        # RNG seed
    # )

    # Initialize the environment and PPO hyperparameters
    env = envs.get_environment("biped")
    # env = envs.get_environment("biped_basic")
    ppo_config = dict(
        num_timesteps=15_000_000,      # total training timesteps
        num_evals=20,                  # number of evaluations
        reward_scaling=1.0,            # reward scale
        episode_length=600,            # max episode length
        normalize_observations=True,   # normalize observations
        unroll_length=20,              # PPO unroll length
        num_minibatches=32,            # PPO minibatches
        num_updates_per_batch=4,       # PPO updates per batch
        discounting=0.97,              # gamma
        learning_rate=3e-4,            # optimizer LR
        clipping_epsilon=0.2,          # PPO clipping epsilon
        entropy_cost=1e-3,             # entropy bonus
        num_envs=8192,                 # parallel envs
        batch_size=256,                 # batch size
        seed=0,                         # RNG seed
    )

    #----------------------- TRAIN -----------------------#

    # Create PPO training instance
    ppo_trainer = PPO_Train(env, ppo_config)

    # start training
    ppo_trainer.train()