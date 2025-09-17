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

# MJx Rollout class
class ParallelSimRollout():

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
        self.control_decimation = config.physics_steps_per_control_step

        # load the mujoco model for parallel sim
        model_path = config.model_path
        self.initialize_model(model_path)

        # poly and observation function
        if policy_params_path is not None:
            self.initialize_policy_and_obs_fn(policy_params_path)
        else:
            print("No policy parameters path provided. Rollouts will use zero input.")

        # initialize jit functions
        self.initialize_jit_functions()
        
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


    # initialize jit functions
    def initialize_jit_functions(self):
        """
        Initialize the jit functions for rollout.
        """

        # jit the rollout with zero inputs function
        # self.rollout_zero_input_jit = jax.jit(self._rollout_zero_input, 
        #                                       static_argnames=('T',))
        self.rollout_zero_input_jit = jax.jit(self._rollout_zero_input, 
                                              static_argnames=('T',), 
                                              donate_argnums=(0,1))
        
        # jit the rollout with policy inputs function
        # self.rollout_policy_input_jit = jax.jit(self._rollout_policy_input, 
        #                                       static_argnames=('T',))
        self.rollout_policy_input_jit = jax.jit(self._rollout_policy_input, 
                                               static_argnames=('T',), 
                                               donate_argnums=(0,1))


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
    def sample_random_uniform_inputs(self, T):
        """
        Sample random sequence of inputs for the system.

        Args:
            T: int, number of time steps
        Returns:
            u_batch: jnp.array, sampled input sequence (batch_size, T, nu)
        """

        # split the rng
        self.rng, subkey = jax.random.split(self.rng)

        # sample random control inputs
        u_seq_batch = jax.random.uniform(subkey, 
                                         (self.batch_size, T, self.nu), 
                                         minval=self.u_lb, 
                                         maxval=self.u_ub) # shape (batch, time, nu)
        
        return u_seq_batch
    

    # rollout with zero input sequence (thin wrapper to allow usage of jitted functions)
    def rollout_zero_input(self, T):
    
        # sample initial conditions
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()

        # perform rollout
        q_log, v_log, u_log = self.rollout_zero_input_jit(q0_batch, v0_batch, T)

        # perform rollout
        return q_log, v_log, u_log

    # rollout with zero input sequence (pure function to jit)
    def _rollout_zero_input(self, q0_batch, v0_batch, T):
        """
        Perform rollout with zero input sequence.
        Args:
            q0_batch: jnp.array, initial positions (batch_size, nq)
            v0_batch: jnp.array, initial velocities (batch_size, nv)
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T+1, nq)
            v_log: jnp.array, logged velocities (batch_size, T+1, nv)
            u_log: jnp.array, logged inputs (batch_size, T, nu) (all zeros)
        """

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q, v: self.mjx_data.replace(qpos=q, qvel=v))(q0_batch, v0_batch)

        # set the control to zero in the batched data
        data_0 = data_0.replace(ctrl=self.u_zero)

        # main step body
        def body(data, _):

            # take a step
            data = self.step_fn_batched(data)

            # state data
            ql = data.qpos   # log q
            vl = data.qvel   # log v

            return data, (ql, vl)

        # forward propagation
        data_last, (q_log, v_log) = lax.scan(body, data_0, None, length=T)

        # add the initial condition to the logs
        q0 = data_0.qpos   # initial q
        v0 = data_0.qvel   # initial v
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)  # shape (T+1, batch, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)  # shape (T+1, batch, nv)

        # swap axis to get (batch, T+1, dim)
        q_log  = jnp.swapaxes(q_log, 0, 1)  # shape (batch, T+1, nq)
        v_log  = jnp.swapaxes(v_log, 0, 1)  # shape (batch, T+1, nv)
        u_log = jnp.broadcast_to(self.u_zero[:, None, :], (self.batch_size, T, self.nu)) # shape (batch, T, nu)

        return q_log, v_log, u_log
    

    # rollout closed loop using RL policy (thin wrapper to allow usage of jitted functions)
    def rollout_policy_input(self, T):
    
        # sample initial conditions
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()

        # perform rollout
        q_log, v_log, u_log = self.rollout_policy_input_jit(q0_batch, v0_batch, T)

        # perform rollout
        return q_log, v_log, u_log

    # rollout closed loop using RL policy (pure function to jit)
    def _rollout_policy_input(self, q0_batch, v0_batch, T):
        """
        Perform rollout with inputs from RL policy.
        Args:
            q0_batch: jnp.array, initial positions (batch_size, nq)
            v0_batch: jnp.array, initial velocities (batch_size, nv)
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T+1, nq)
            v_log: jnp.array, logged velocities (batch_size, T+1, nv)
            u_log: jnp.array, logged inputs (batch_size, T, nu)
        """

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q, v: self.mjx_data.replace(qpos=q, qvel=v))(q0_batch, v0_batch)

        # start with zero action 
        u0 = jnp.zeros((self.batch_size, self.nu), dtype=jnp.float32)  # (batch_size, nu) zeros

        # main step body
        def body(carry, _):

            # unpack carry
            data, u_curr, t = carry

            # update control input at specified decimation
            def compute_control(_):
                obs = self.obs_fn_batched(data)    # get observations
                act = self.policy_fn_batched(obs)  # get actions from policy
                return act

            # if time to update control input
            u_next = lax.cond((t % self.control_decimation) == 0,
                               compute_control, 
                               lambda _: u_curr, operand=None)

            # apply control and take step
            data = data.replace(ctrl=u_next)
            data = self.step_fn_batched(data)

            # state data
            ql = data.qpos   # log q
            vl = data.qvel   # log v

            return (data, u_next, t + 1), (ql, vl, u_next)

        # do the forward propagation
        (data_last, u_last, _), (q_log, v_log, u_log) = lax.scan(body, (data_0, u0, 0), None, length=T)

        # add the initial condition to the logs
        q0 = data_0.qpos   # initial q
        v0 = data_0.qvel   # initial v
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)  # shape (T+1, batch_size, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)  # shape (T+1, batch_size, nv)

        # swap axis to get (batch, T+1, dim)
        q_log  = jnp.swapaxes(q_log, 0, 1)  # shape (batch_size, T+1, nq)
        v_log  = jnp.swapaxes(v_log, 0, 1)  # shape (batch_size, T+1, nv)
        u_log  = jnp.swapaxes(u_log, 0, 1)  # shape (batch_size, T, nu)

        return q_log, v_log, u_log


##################################################################################
# EXAMPLE USAGE
##################################################################################


if __name__ == "__main__":

    # create a random number generator
    seed = int(time.time())
    # seed = 0
    rng = jax.random.PRNGKey(seed)

    # choose batch size
    batch_size = 2048

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
    r = ParallelSimRollout(rng, batch_size, env_name, state_bounds, params_path)

    # number of simulation steps
    num_steps = 400

    # rollout with zero inputs
    time_0 = time.time()
    # q_log_1, v_log_1, u_log_1 = r.rollout_zero_input(num_steps)
    q_log_1, v_log_1, u_log_1 = r.rollout_policy_input(num_steps)
    time_1 = time.time()
    print(f"Rollout with zero input took (first): {(time_1-time_0):.3f}s")

    time_0 = time.time()
    # q_log_2, v_log_2, u_log_2 = r.rollout_zero_input(num_steps)
    q_log_2, v_log_2, u_log_2 = r.rollout_policy_input(num_steps)
    time_1 = time.time()
    print(f"Rollout with zero input took (steady): {(time_1-time_0):.3f}s")

    time_0 = time.time()
    # q_log_3, v_log_3, u_log_3 = r.rollout_zero_input(num_steps)
    q_log_3, v_log_3, u_log_3 = r.rollout_policy_input(num_steps)
    time_1 = time.time()
    print(f"Rollout with zero input took (steady): {(time_1-time_0):.3f}s")

    time_0 = time.time()
    # q_log_4, v_log_4, u_log_4 = r.rollout_zero_input(num_steps)
    q_log_4, v_log_4, u_log_4 = r.rollout_policy_input(num_steps)
    time_1 = time.time()
    print(f"Rollout with zero input took (steady): {(time_1-time_0):.3f}s")

    time_0 = time.time()
    # q_log_5, v_log_5, u_log_5 = r.rollout_zero_input(num_steps)
    q_log_5, v_log_5, u_log_5 = r.rollout_policy_input(num_steps)
    time_1 = time.time()
    print(f"Rollout with zero input took (steady): {(time_1-time_0):.3f}s")

    print(f"q_log_1 shape: {q_log_1.shape}")
    print(f"v_log_1 shape: {v_log_1.shape}")
    print(f"u_log_1 shape: {u_log_1.shape}")
    print(f"q_log_2 shape: {q_log_2.shape}")
    print(f"v_log_2 shape: {v_log_2.shape}")
    print(f"u_log_2 shape: {u_log_2.shape}")
    print(f"q_log_3 shape: {q_log_3.shape}")
    print(f"v_log_3 shape: {v_log_3.shape}")
    print(f"u_log_3 shape: {u_log_3.shape}")


    q_log_1 = np.array(q_log_1)
    v_log_1 = np.array(v_log_1)
    u_log_1 = np.array(u_log_1)
    q_log_2 = np.array(q_log_2)
    v_log_2 = np.array(v_log_2)
    u_log_2 = np.array(u_log_2)
    q_log_3 = np.array(q_log_3)
    v_log_3 = np.array(v_log_3)
    u_log_3 = np.array(u_log_3)
    
    q_err_1 = np.linalg.norm(q_log_1 - q_log_2)
    q_err_2 = np.linalg.norm(q_log_2 - q_log_3)
    q_err_3 = np.linalg.norm(q_log_1 - q_log_3)
    v_err_1 = np.linalg.norm(v_log_1 - v_log_2)
    v_err_2 = np.linalg.norm(v_log_2 - v_log_3)
    v_err_3 = np.linalg.norm(v_log_1 - v_log_3)
    u_err_1 = np.linalg.norm(u_log_1 - u_log_2)
    u_err_2 = np.linalg.norm(u_log_2 - u_log_3)
    u_err_3 = np.linalg.norm(u_log_1 - u_log_3)

    print(f"q_err_1: {q_err_1:.6e}, q_err_2: {q_err_2:.6e}, q_err_3: {q_err_3:.6e}")
    print(f"v_err_1: {v_err_1:.6e}, v_err_2: {v_err_2:.6e}, v_err_3: {v_err_3:.6e}")
    print(f"u_err_1: {u_err_1:.6e}, u_err_2: {u_err_2:.6e}, u_err_3: {u_err_3:.6e}")


    # save the data
    q_log_np = np.array(q_log_1)
    v_log_np = np.array(v_log_1)
    u_log_np = np.array(u_log_1)

    print(q_log_np.shape)
    print(v_log_np.shape)
    print(u_log_np.shape)

    # save the data
    save_path = "./data/paddle_ball_data.npz"
    np.savez(save_path, q_traj=q_log_np, v_traj=v_log_np, u_traj=u_log_np)
    print(f"Saved data to: {save_path}")
