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
# PARALLEL DYNAMICS ROLLOUT CLASS
##################################################################################



##################################################################################
# PARALLEL DYNAMICS ROLLOUT CLASS
##################################################################################

# MJx Rollout class
class ParallelRollout():

    # initialize the class
    def __init__(self, rng, batch_size, env_name, state_bounds, policy_params_path=None):

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
        if policy_params_path is not None:
            self.initialize_policy_and_obs_fn(policy_params_path)
        else:
            print("No policy parameters path provided. Rollouts will use zero input.")
        
        # set the initial condition state bounds
        self.q_lb, self.q_ub, self.v_lb, self.v_ub = (
            jnp.asarray(state_bounds[0], dtype=jnp.float32),
            jnp.asarray(state_bounds[1], dtype=jnp.float32),
            jnp.asarray(state_bounds[2], dtype=jnp.float32),
            jnp.asarray(state_bounds[3], dtype=jnp.float32),
        )

        # zeros vector to use with zero input rollouts
        self.u_zero = jnp.zeros((batch_size, self.nu), dtype=jnp.float32)  # (batch, nu) zeros

        # sampling bounds. Should be [-1.0, 1.0] for all models
        self.u_lb = -jnp.ones((self.nu,), dtype=jnp.float32)  # lower bound is -1.0, shape (nu,)
        self.u_ub =  jnp.ones((self.nu,), dtype=jnp.float32)  # upper bound is  1.0, shape (nu,)

    # initialize model
    def initialize_model(self, model_path):
        """
        Initialize the mujoco model and data for parallel rollout.
        
        Args:
            model_path: str, path to the mujoco model xml file
        """

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
        q0_dummy_batch = jnp.zeros((self.batch_size, self.nq), dtype=jnp.float32)  # shape (batch, nq)
        v0_dummy_batch = jnp.zeros((self.batch_size, self.nv), dtype=jnp.float32)  # shape (batch, nv)

        # create the batched model data
        self.mjx_data_batched = jax.vmap(
            lambda q, v: self.mjx_data.replace(qpos=q, qvel=v)
        )(q0_dummy_batch, v0_dummy_batch)

        # create the batched step function
        self.step_fn_batched = jax.jit(jax.vmap(lambda d: mjx.step(self.mjx_model, d), in_axes=0))

        # simulation parameters
        self.sim_dt = float(self.mjx_model.opt.timestep)  # simulation time step

        # print message
        print(f"Initialized batched MJX model from [{model_path}].")
        print(f"   sim_dt: {self.sim_dt:.4f}")
        print(f"   nq: {self.nq}")
        print(f"   nv: {self.nv}")
        print(f"   nu: {self.nu}")


    # initialize the policy and observation functions
    def initialize_policy_and_obs_fn(self, policy_params_path):
        """
        Initialize the policy and observation functions.

        Args:
            policy_params_path: str, path to the policy parameters file
        Returns:
            policy_fn: function, policy function
            obs_fn: function, observation function
        """

        # create the policy and observation function
        ppo_player = PPO_Play(self.env, policy_params_path)

        # get the policy and observation functions
        policy_fn, obs_fn = ppo_player.policy_and_obs_functions()

        # create the batched policy and observation functions
        self.policy_fn_batched = jax.vmap(policy_fn, in_axes=0)
        self.obs_fn_batched = jax.vmap(obs_fn, in_axes=0)

        # print message
        print(f"Initialized batched policy and observation functions from:")
        print(f"   env:    [{self.env.robot_name}]")
        print(f"   policy: [{policy_params_path}]")


    # sample initial conditions
    def sample_random_uniform_initial_conditions(self):
        """
        Sample initial conditions for the system.

        Returns:
            q0_batch: jnp.array, sampled initial positions (batch_size, nq)
            v0_batch: jnp.array, sampled initial velocities (batch_size, nv)
        """

        # split the rng
        self.rng, key1, key2 = jax.random.split(self.rng, 3)

        # sample initial conditions
        q0_batch = jax.random.uniform(key1, 
                                      (self.batch_size, self.nq), 
                                      minval=self.q_lb, 
                                      maxval=self.q_ub).astype(jnp.float32)
        v0_batch = jax.random.uniform(key2, 
                                      (self.batch_size, self.nv), 
                                      minval=self.v_lb, 
                                      maxval=self.v_ub).astype(jnp.float32)

        return q0_batch, v0_batch
    

    # sample inputs from a uniform distribution
    def sample_random_uniform_inputs(self, N):
        """
        Sample random sequence of inputs for the system.

        Args:
            N: int, number of time steps
        Returns:
            u_batch: jnp.array, sampled input sequence (batch_size, N, nu)
        """

        # split the rng
        self.rng, subkey = jax.random.split(self.rng)

        # sample random control inputs
        u_seq_batch = jax.random.uniform(subkey, 
                                         (self.batch_size, N, self.nu), 
                                         minval=self.u_lb, 
                                         maxval=self.u_ub) # shape (batch, time, nu)
        
        return u_seq_batch
    

    # rollout with zero input sequence
    def rollout_zero(self, T):
        """
        Perform rollout with zero input sequence.
        Args:
            T: int, number of time steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T, nu) (all zeros)
        """

        # prealloc logs on device
        q_log = jnp.empty((self.batch_size, T, self.nq), dtype=jnp.float32)
        v_log = jnp.empty((self.batch_size, T, self.nv), dtype=jnp.float32)

        # sample initial conditions
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()

        # set the initial conditions in the batched data
        data_batch = jax.vmap(
            lambda q, v: self.mjx_data.replace(qpos=q, qvel=v)
        )(q0_batch, v0_batch)

        # set the control to zero in the batched data
        data_batch = data_batch.replace(ctrl=self.u_zero)

        # main step body
        def body(i, carry):

            # unpack the carry
            data, ql, vl = carry

            # apply the control and step
            data = self.step_fn_batched(data)

            # log data
            ql = ql.at[:, i, :].set(data.qpos)   # log q
            vl = vl.at[:, i, :].set(data.qvel)   # log v

            return (data, ql, vl)
        
        # loop over time steps
        data_batch,  q_log, v_log = lax.fori_loop(0, T, 
                                                  body,
                                                  (data_batch, q_log, v_log))
        
        # build zeros input log
        u_log = jnp.broadcast_to(self.u_zero[:, None, :], (self.batch_size, T, self.nu))

        return q_log, v_log, u_log


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

    # state space domain
    q_lb = jnp.array([ 1.0,  0.1])
    q_ub = jnp.array([ 3.0,  0.9])
    v_lb = jnp.array([-5.0, -5.0])
    v_ub = jnp.array([ 5.0,  5.0])
    state_bounds = (q_lb, q_ub, v_lb, v_ub)
    
    # create the rollout instance
    r = ParallelRollout(rng, batch_size, env_name, state_bounds, params_path)

    # number of simulation steps
    t_sim = 4.0                          # total simulation time
    num_steps = round(t_sim / r.sim_dt)  # number of simulation steps

    # rollout with zero inputs
    time_0 = time.time()
    q_log, v_log, u_log = r.rollout_zero(num_steps)
    time_1 = time.time()
    print(f"Rollout with zero input took (first): {(time_1-time_0):.3f}s")

    time_0 = time.time()
    q_log, v_log, u_log = r.rollout_zero(num_steps)
    time_1 = time.time()
    print(f"Rollout with zero input took (steady): {(time_1-time_0):.3f}s")

    # rollout with random inputs
    time_0 = time.time()
    q_log, v_log, u_log = r.rollout_zero(num_steps)
    time_1 = time.time()
    print(f"Rollout with random input took (steady): {(time_1-time_0):.3f}s")

    # save the data
    q_log_np = np.array(q_log)
    v_log_np = np.array(v_log)
    u_log_np = np.array(u_log)

    print(q_log_np.shape)
    print(v_log_np.shape)
    print(u_log_np.shape)

    # save the data
    save_path = "./data/paddle_ball_data.npz"
    np.savez(save_path, q_traj=q_log_np, v_traj=v_log_np, u_traj=u_log_np)
    print(f"Saved data to: {save_path}")
