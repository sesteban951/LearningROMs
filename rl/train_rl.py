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
import brax.training.distribution as distribution

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig
from envs.cart_pole_tracking_env import CartPoleTrackingEnv, CartPoleTrackingConfig
from envs.acrobot_env import AcrobotEnv, AcrobotConfig
from envs.biped_env import BipedEnv, BipedConfig
from envs.biped_basic_env import BipedBasicEnv, BipedBasicConfig
from envs.hopper_env import HopperEnv, HopperConfig
from envs.paddle_ball_env import PaddleBallEnv, PaddleBallConfig
from algorithms.ppo_train import PPO_Train
from algorithms.custom_networks import BraxPPONetworksWrapper, MLPConfig, MLP

if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    device = jax.devices()[0]
    print("Device type:", device.platform)      # e.g. 'gpu' or 'cpu'
    print("Device name:", device.device_kind)   # e.g. 'NVIDIA GeForce RTX 4090'

    #--------------------------------- SETUP ---------------------------------#

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("cart_pole")
    env = envs.get_environment("cart_pole_tracking")
    policy_network_config = MLPConfig(
        layer_sizes=(32, 32, 32, 32, 2*env.action_size),   # policy hidden layer sizes
        activation_fn_name="relu",                         # activation function
    )
    value_network_config = MLPConfig(
        layer_sizes=(256, 256, 256, 256, 256, 1),  # value hidden layer sizes
        activation_fn_name="swish",                # activation function
    )
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(policy_network_config),
        value_network=MLP(value_network_config),
        action_distribution=distribution.NormalTanhDistribution
    )
    ppo_config = dict(
        num_timesteps=100_000_000,      
        num_evals=10,                  
        reward_scaling=0.1,            
        episode_length=200,            
        normalize_observations=True,   
        unroll_length=10,              
        num_minibatches=32,            
        num_updates_per_batch=8,       
        discounting=0.97,              
        learning_rate=1e-3,            
        clipping_epsilon=0.2,          
        entropy_cost=1e-3,             
        num_envs=1024,                 
        batch_size=1024,               
        seed=0,                        
    )

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("acrobot")
    # policy_network_config = MLPConfig(
    #     layer_sizes=(32, 32, 32, 2*env.action_size),   # policy hidden layer sizes
    #     activation_fn_name="swish",                        # activation function
    # )
    # value_network_config = MLPConfig(
    #     layer_sizes=(256, 256, 256, 256, 256, 1),  # value hidden layer sizes
    #     activation_fn_name="swish",                # activation function
    # )
    # network_wrapper = BraxPPONetworksWrapper(
    #     policy_network=MLP(policy_network_config),
    #     value_network=MLP(value_network_config),
    #     action_distribution=distribution.NormalTanhDistribution
    # )
    # ppo_config = dict(
    #     num_timesteps=100_000_000,      
    #     num_evals=10,                  
    #     reward_scaling=0.1,            
    #     episode_length=300,            
    #     normalize_observations=True,   
    #     unroll_length=10,              
    #     num_minibatches=32,            
    #     num_updates_per_batch=8,       
    #     discounting=0.98,              
    #     learning_rate=5e-4,            
    #     clipping_epsilon=0.2,          
    #     entropy_cost=3e-4,             
    #     num_envs=4096,                 
    #     batch_size=4096,               
    #     seed=0,                        
    # )

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("paddle_ball")
    # policy_network_config = MLPConfig(
    #     layer_sizes=(32, 32, 32, 32, 2*env.action_size),   # policy hidden layer sizes
    #     activation_fn_name="swish",                        # activation function
    # )
    # value_network_config = MLPConfig(
    #     layer_sizes=(256, 256, 256, 256, 256, 1),  # value hidden layer sizes
    #     activation_fn_name="swish",                # activation function
    # )
    # network_wrapper = BraxPPONetworksWrapper(
    #     policy_network=MLP(policy_network_config),
    #     value_network=MLP(value_network_config),
    #     action_distribution=distribution.NormalTanhDistribution
    # )
    # ppo_config = dict(
    #     num_timesteps=80_000_000,      
    #     num_evals=10,                  
    #     reward_scaling=0.1,            
    #     episode_length=500,            
    #     normalize_observations=True,   
    #     unroll_length=10,              
    #     num_minibatches=32,            
    #     num_updates_per_batch=8,       
    #     discounting=0.97,              
    #     learning_rate=5e-4,            
    #     clipping_epsilon=0.2,          
    #     entropy_cost=1e-3,             
    #     num_envs=2048,                 
    #     batch_size=2048,               
    #     seed=0,                        
    # )

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("hopper")
    # policy_network_config = MLPConfig(
    #     layer_sizes=(32, 32, 32, 32, 2*env.action_size),   # policy hidden layer sizes
    #     activation_fn_name="swish",                        # activation function
    # )
    # value_network_config = MLPConfig(
    #     layer_sizes=(256, 256, 256, 256, 256, 1),  # value hidden layer sizes
    #     activation_fn_name="swish",                # activation function
    # )
    # network_wrapper = BraxPPONetworksWrapper(
    #     policy_network=MLP(policy_network_config),
    #     value_network=MLP(value_network_config),
    #     action_distribution=distribution.NormalTanhDistribution
    # )
    # ppo_config = dict(
    #     num_timesteps=50_000_000,      
    #     num_evals=10,                  
    #     reward_scaling=0.1,            
    #     episode_length=600,            
    #     normalize_observations=True,   
    #     unroll_length=5,              
    #     num_minibatches=64,            
    #     num_updates_per_batch=8,       
    #     discounting=0.97,              
    #     learning_rate=5e-4,            
    #     clipping_epsilon=0.2,          
    #     entropy_cost=3e-4,             
    #     num_envs=4096,                 
    #     batch_size=4096,               
    #     seed=0,                        
    # )

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("biped_basic")
    # env = envs.get_environment("biped")
    # policy_network_config = MLPConfig(
    #     layer_sizes=(256, 256, 256, 2*env.action_size), # policy hidden layer sizes
    #     activation_fn_name="tanh",                           # activation function
    # )
    # value_network_config = MLPConfig(
    #     layer_sizes=(512, 512, 512, 1),  # value hidden layer sizes
    #     activation_fn_name="tanh",                 # activation function
    # )
    # network_wrapper = BraxPPONetworksWrapper(
    #     policy_network=MLP(policy_network_config),
    #     value_network=MLP(value_network_config),
    #     action_distribution=distribution.NormalTanhDistribution
    # )
    # ppo_config = dict(
    #     num_timesteps=250_000_000,      # total training timesteps
    #     num_evals=20,                  # number of evaluations
    #     reward_scaling=1.0,            # reward scale
    #     episode_length=800,            # max episode length
    #     normalize_observations=True,   # normalize observations
    #     action_repeat=1,               # action repeat
    #     unroll_length=20,              # PPO unroll length
    #     num_minibatches=32,            # PPO minibatches
    #     num_updates_per_batch=4,       # PPO updates per batch
    #     discounting=0.98,              # gamma
    #     learning_rate=3e-4,            # optimizer LR
    #     entropy_cost=0.005,             # entropy bonus
    #     clipping_epsilon=0.2,          # PPO clipping epsilon
    #     num_envs=8192,                 # parallel envs
    #     batch_size=256,                # batch size
    #     max_grad_norm=1.0,             # gradient clipping
    #     seed=0,                        # RNG seed
    # )

    #--------------------------------- TRAIN ---------------------------------#

    # Create PPO training instance
    ppo_trainer = PPO_Train(env, network_wrapper, ppo_config)

    # start training
    ppo_trainer.train()