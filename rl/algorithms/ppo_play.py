# standard imports
import numpy as np
import time

# jax imports
import jax

# brax improts
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training import types

# for importing policy
import pickle

# custom imports
from rl.algorithms.custom_networks import MLP, MLPConfig, BraxPPONetworksWrapper
from rl.algorithms.custom_networks import make_policy_function
from brax.training import distribution


# PPO Training class
class PPO_Play:

    def __init__(self, env, policy_data_path):

        # make copy of the env
        self.env = env

        # Load trained policy params
        with open(policy_data_path, "rb") as f:
            self.policy_data = pickle.load(f)

        # print info
        print(f"Loaded policy data from: [{policy_data_path}]")

    # build the policy function
    def build_policy_fn(self):

        # unpack the policy data
        policy_network_config = self.policy_data["policy_config"]
        value_network_config = self.policy_data["value_config"]
        action_dist_name = self.policy_data["action_dist_class"]
        params = self.policy_data["params"]

        #  rebuild the action distribution
        if action_dist_name == "NormalTanhDistribution":
            action_dist = distribution.NormalTanhDistribution
        else:
            raise ValueError(f"Unknown action or not implemented distribution: [{action_dist_name}]")
        
        # rebuild the network wrapper
        network_wrapper = BraxPPONetworksWrapper(
            policy_network=MLP(policy_network_config),
            value_network=MLP(value_network_config),
            action_distribution=action_dist
        )
        
        # observation and action size
        obs_size = self.env.observation_size
        act_size = self.env.action_size

        # set some hardcoded params
        normalize_obs = True   # whether to normalize observations WARNING: make sure this matches PPO config
        deterministic = True   # for inference, we want deterministic actions

        # rebuild the networks
        policy_fn = make_policy_function(
            network_wrapper=network_wrapper,
            params=params,
            obs_size=obs_size,
            act_size=act_size,
            normalize_observations=normalize_obs,
            deterministic=deterministic
        )

        # # build the function factory
        # inference_fn_factory = ppo_networks.make_inference_fn(networks)

        # # create the actual inference function using the params
        # deterministic = True   # for inference, we want deterministic actions
        # self.policy_fn = inference_fn_factory(params=self.params,
        #                                       deterministic=deterministic)

        print("Rebuilt policy function.")
        
        return policy_fn
        
    # build the observation function
    def build_obs_fn(self):
        
        # build the observation function
        obs_fn = self.env._compute_obs

        return obs_fn

    # jit the policy and obs functions for speed
    def policy_and_obs_functions(self):

        # rebuild the policy and obs functions
        policy_fn = self.build_policy_fn()
        obs_fn = self.build_obs_fn()

        # jit the policy and observation functions
        policy_jit = jax.jit(lambda obs: policy_fn(obs, jax.random.PRNGKey(0))[0])
        obs_jit = jax.jit(obs_fn)

        print("Jitted policy and observation functions.")

        return policy_jit, obs_jit