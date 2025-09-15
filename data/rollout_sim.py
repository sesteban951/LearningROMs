##
#
#  Playback and record data
#  
##

# standard imports
import os, sys
import numpy as np
import time

# jax imports
import jax
import jax.numpy as jnp
from jax import lax

# brax imports
from brax import envs

# mujoco imports
import mujoco
import mujoco.mjx as mjx

# change directories to project root (so `from rl...` works even if run from /data)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# custom imports
from rl.envs.cart_pole_env import CartPoleEnv
from rl.envs.acrobot_env import AcrobotEnv
from rl.envs.biped_env import BipedEnv
from rl.envs.biped_basic_env import BipedBasicEnv
from rl.envs.hopper_env import HopperEnv
from rl.envs.paddle_ball_env import PaddleBallEnv
from rl.algorithms.ppo_play import PPO_Play

##################################################################################
# DYNAMICS ROLLOUT CLASS
##################################################################################

# MJx Rollout class
class DynamicsRollout():

    # initialize the class
    def __init__(self, rng, batch_size, env_name, policy_params_path=None):

        # assign the random seed
        self.rng = rng

        # assign the batch size
        self.batch_size = batch_size

        # load the enviornment
        self.env = envs.get_environment(env_name)
        config = self.env.config
        
        # load some parameters
        self.control_step_size = config.physics_steps_per_control_step

        # load the mujoco model for parallel sim
        model_path = config.model_path
        self.initialize_model(model_path)

        # poly and observation function
        self.policy_params_path = policy_params_path
        if policy_params_path is not None:
            self.initialize_policy_and_obs_fn(policy_params_path)
        else:
            print("No policy path provided.")
            self.step_fn_batched = None
            self.policy_fn_batched = None
            self.obs_fn_batched = None

        # zeros vector to use with zero input rollouts
        self.u_zero = jnp.zeros((batch_size, self.nu)) # (batch, nu) zeros

    # initialize model
    def initialize_model(self, model_path):

        # import the mujoco model
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        mj_data = mujoco.MjData(mj_model)

        # put the model and data on GPU
        self.mjx_model = mjx.put_model(mj_model)
        self.mjx_data = mjx.put_data(mj_model, mj_data)

        # load sizes
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        # create zeroes data to initialize the parallel model
        q0_dummy_batch = jnp.zeros((self.batch_size, self.nq))  # shape (batch, nq)
        v0_dummy_batch = jnp.zeros((self.batch_size, self.nv))  # shape (batch, nv)

        # create the batched model data
        self.mjx_data_batched = jax.vmap(lambda i: self.mjx_data.replace(qpos=q0_dummy_batch[i], qvel=v0_dummy_batch[i]))(jnp.arange(self.batch_size))

        # simulation parameters
        self.sim_dt = self.mjx_model.opt.timestep  # simulation time step

        # print message
        print(f"Initialized batched MJX model from [{model_path}].")
        print(f"   sim_dt: {self.sim_dt:.4f}")
        print(f"   nq: {self.nq}")
        print(f"   nv: {self.nv}")
        print(f"   nu: {self.nu}")

    # initialize the policy and observation functions
    def initialize_policy_and_obs_fn(self, policy_params_path):

        # create the policy and observation function
        ppo_player = PPO_Play(self.env, policy_params_path)

        # get the policy and observation functions
        policy_fn, obs_fn = ppo_player.policy_and_obs_functions()

        # create the batched step function
        self.step_fn_batched = jax.jit(jax.vmap(lambda d: mjx.step(self.mjx_model, d), in_axes=(0)))

        # create the batched policy and observation functions
        self.policy_fn_batched = jax.vmap(policy_fn, in_axes=0)
        self.obs_fn_batched = jax.vmap(obs_fn, in_axes=0)

        # print message
        print(f"Initialized batched policy and observation functions from:")
        print(f"   env:    [{self.env.robot_name}]")
        print(f"   policy: [{policy_params_path}]")

    # sample initial conditions
    def sample_initial_conditions(self, batch_size, 
                                        q_lb, q_ub, 
                                        v_lb, v_ub):
        """
        Sample initial conditions for the system.

        Args:
            rng: jax.random.PRNGKey, random number generator key
            batch_size: int, number of initial conditions to sample
            q_lb: jnp.array, lower bound for position (nq,)
            q_ub: jnp.array, upper bound for position (nq,)
            v_lb: jnp.array, lower bound for velocity (nv,)
            v_ub: jnp.array, upper bound for velocity (nv,)
        Returns:
            q0_batch: jnp.array, sampled initial positions (batch_size, nq)
            v0_batch: jnp.array, sampled initial velocities (batch_size, nv)
        """

        # split the rng
        self.rng, key1, key2 = jax.random.split(rng, 3)

        # sample initial conditions
        q0_batch = jax.random.uniform(key1, (batch_size, self.nq), minval=q_lb, maxval=q_ub) # shape (batch, nq)
        v0_batch = jax.random.uniform(key2, (batch_size, self.nv), minval=v_lb, maxval=v_ub) # shape (batch, nv)

        return q0_batch, v0_batch
    
    # sample random actions
    def sample_random_inputs(self, batch_size, N):
        """
        Sample random sequence of inputs for the system.

        Args:
            rng: jax.random.PRNGKey, random number generator key
            batch_size: int, number of trajectories
            N: int, number of time steps
        Returns:
            u_bacth:
        """

        # split the rng
        self.rng, subkey = jax.random.split(self.rng)

        # sample random contorl inputs
        u_lb = -jnp.ones((self.nu,))  # lower bound is -1.0, shape (nu,)
        u_ub =  jnp.ones((self.nu,))  # upper bound is  1.0, shape (nu,)
        u_seq = jax.random.uniform(subkey, 
                                   (batch_size, N, self.nu), 
                                   minval=u_lb, 
                                   maxval=u_ub) # shape (batch, time, nu)
        
        return u_seq
    
    # 






        



##################################################################################
# EXAMPLE USAGE
##################################################################################

if __name__ == "__main__":

    # create a random number generator
    seed = int(time.time())
    rng = jax.random.PRNGKey(seed)

    # choose batch size
    batch_size = 512

    # choose environment and policy parameters
    env_name = "paddle_ball"

    # choose the policy parameters path
    params_path = "./rl/policy/paddle_ball_policy.pkl"
    
    # create the rollout instance
    # r = DynamicsRollout(rng, batch_size, env_name)
    r = DynamicsRollout(rng, batch_size, env_name, params_path)

