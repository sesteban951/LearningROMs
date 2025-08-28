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

# PPO Training class
class PPO_Play:

    def __init__(self, env, params_path):

        # Load trained params
        with open(params_path, "rb") as f:
            self.params = pickle.load(f)

        # print info
        print(f"Loaded policy params from: [{params_path}]")

        # make copy of the envand env config
        self.env = env
        self.env_config = env.config

    # function that rebuilds the policy function, and return jitted policy and obs functions
    def build_policy_fn(self):
        
        # get relevant env params
        obs_size = self.env.observation_size
        act_size = self.env.action_size

        # set the observation preprocessing function
        normalize_obs = True  # WARNING: this is hardcoded, make sure it matches ppo config!
        if normalize_obs == True:
            # Normalize observations
            preprocess_observations_fn = running_statistics.normalize
        else: 
            # Identity function (no preprocessing, just pass through)
            preprocess_observations_fn = types.identity_observation_preprocessor        

        # make the ppo networks
        networks = ppo_networks.make_ppo_networks(
            observation_size=obs_size,
            action_size=act_size,
            preprocess_observations_fn=preprocess_observations_fn,
        )

        print("Rebuilding policy function...")

        # build the function factory
        inference_fn_factory = ppo_networks.make_inference_fn(networks)

        # create the actual inference function using the params
        deterministic = True   # for inference, we want deterministic actions
        self.policy_fn = inference_fn_factory(params=self.params,
                                              deterministic=deterministic)
        
        print("Rebuilt policy function.")

    # jit the policy and obs functions for speed
    def policy_and_obs_functions(self):

        # rebuild the policy and obs functions
        self.build_policy_fn()

        print("Jitting the policy and observation functions...")

        policy_jit = jax.jit(lambda obs: self.policy_fn(obs, jax.random.PRNGKey(0))[0])
        obs_jit = jax.jit(self.env._compute_obs)

        print("Jitted functions.")

        return policy_jit, obs_jit