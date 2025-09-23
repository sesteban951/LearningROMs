##
#
#  Perform Parallel MJX Rollouts
#  
##

# standard imports
import os, sys
import numpy as np
import time
from dataclasses import dataclass
from functools import partial

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
#AUXILLIARY FUNCTIONS
##################################################################################



##################################################################################
# PARALLEL DYNAMICS ROLLOUT CLASS
##################################################################################

# struct to hold the parallel sim config
@dataclass
class ParallelSimConfig:

    env_name: str                  # RL environment name
    batch_size: int                # batch size for parallel rollout
    state_bounds: tuple            # tuple of (q_lb, q_ub, v_lb, v_ub) for initial condition sampling
    rng: jax.random.PRNGKey        # random number generator key
    policy_params_path: str=None   # path to policy parameters file (if None, cannot run closed loop with policy)


# MJX Rollout class
class ParallelSim():
    """
    Class to perform parallel rollouts using mujoco mjx on GPU.

    Args:
        config: ParallelSimConfig, configuration for the parallel sim
    """

    # initialize the class
    def __init__(self, config: ParallelSimConfig):

        # assign the random seed
        self.rng = config.rng

        # assign the batch size
        self.batch_size = config.batch_size

        # load the enviornment
        self.env = envs.get_environment(config.env_name)
        env_config = self.env.config
        
        # load the dontrol decimation (number of sim steps per control step)
        self.control_decimation = env_config.physics_steps_per_control_step

        # load the mujoco model for parallel sim
        model_path = env_config.model_path
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        mj_data = mujoco.MjData(mj_model)
        self.initialize_model(mj_model, mj_data, model_path)

        # load in the touch sensors if any
        self.initialize_touch_sensors(mj_model)

        # policy and observation function
        if config.policy_params_path is not None:
            self.initialize_policy_and_obs_fn(config.policy_params_path)
        else:
            print("No policy parameters path provided. Rollouts will use zero input.")
            self.policy_fn_batched = None
            self.obs_fn_batched = None

        # initialize jit functions for speed
        self.initialize_jit_functions()
        
        # set the initial condition state bounds
        state_bounds = config.state_bounds
        self.q_lb, self.q_ub, self.v_lb, self.v_ub = (
            jnp.asarray(state_bounds[0], dtype=jnp.float32),
            jnp.asarray(state_bounds[1], dtype=jnp.float32),
            jnp.asarray(state_bounds[2], dtype=jnp.float32),
            jnp.asarray(state_bounds[3], dtype=jnp.float32),
        )

        # zeros vector to use with zero input rollouts
        self.u_zero = jnp.zeros((self.batch_size, self.nu), dtype=jnp.float32)  # (batch, nu) zeros

        # sampling bounds. Should be [-1.0, 1.0] for all models beause of how XML is set up
        self.u_lb = -jnp.ones((self.nu,), dtype=jnp.float32)  # lower bound is -1.0, shape (nu,)
        self.u_ub =  jnp.ones((self.nu,), dtype=jnp.float32)  # upper bound is  1.0, shape (nu,)

    ######################################### INITIALIZATION #########################################

    # initialize model
    def initialize_model(self, mj_model, mj_data, model_path):
        """
        Initialize the mujoco model and data for parallel rollout.
        
        Args:
            mj_model: mujoco.MjModel, the mujoco model
            mj_data: mujoco.MjData, the mujoco data
        """

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

    # initialize touch sensors
    def initialize_touch_sensors(self, mj_model):
        """
        Initialize touch sensors if any are present in the model.

        Args:
            mj_model: mujoco.MjModel, the mujoco model
        """
        
        # cache touch sensor IDs
        self.touch_sensor_ids = [
            i for i, stype in enumerate(mj_model.sensor_type)
            if stype == mujoco.mjtSensor.mjSENS_TOUCH
        ]
        self.nc = len(self.touch_sensor_ids)
        print(f"Found {self.nc} touch sensors: {self.touch_sensor_ids}")

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
        Initialize the jit functions for rollout for speed.
        """

        # jit the rollout with zero inputs function
        self.rollout_zero_input_jit = jax.jit(self._rollout_zero_input, 
                                              static_argnames=('T',), 
                                              donate_argnums=(0, 1))
        
        # jit the rollout with policy inputs function
        self.rollout_policy_input_jit = jax.jit(self._rollout_policy_input, 
                                               static_argnames=('T',), 
                                               donate_argnums=(0, 1))
        
        # jit the rollout with random inputs function
        self.rollout_random_input_jit = jax.jit(self._rollout_random_input,
                                               static_argnames=('T',),
                                               donate_argnums=(0, 1))


    ######################################### SAMPLING #########################################

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
    def sample_random_uniform_inputs(self, S):
        """
        Sample random sequence of inputs for the system.

        Args:
            S: int, length of input sequence
        Returns:
            u_batch: jnp.array, sampled input sequence (batch_size, S, nu)
        """

        # split the rng
        self.rng, subkey = jax.random.split(self.rng)

        # sample random control inputs
        u_seq_batch = jax.random.uniform(subkey, 
                                         (self.batch_size, S, self.nu), 
                                         minval=self.u_lb, 
                                         maxval=self.u_ub) # shape (batch, S, nu)
        
        return u_seq_batch


    ########################################## ZERO INPUT ROLLOUT ##########################################

    # rollout with zero input sequence (thin wrapper to allow usage of jitted functions)
    def rollout_zero_input(self, T):
        """
        Perform rollout with zero input sequence.

        Args:
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu) (all zeros)
        """
    
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
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu) (all zeros)
        """

        # number of integration steps
        S = T - 1

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q0, v0: self.mjx_data.replace(qpos=q0, qvel=v0))(q0_batch, v0_batch)

        # set the control to zero in the batched data
        data_0 = data_0.replace(ctrl=self.u_zero)

        # main step body
        def body(data, _):

            # take a step
            data = self.step_fn_batched(data)

            return data, (data.qpos, data.qvel)

        # forward propagation
        data_last, (q_log, v_log) = lax.scan(body, data_0, None, length=S)

        # add the initial condition to the logs
        q0 = data_0.qpos   # initial q
        v0 = data_0.qvel   # initial v
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)  # shape (T, batch_size, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)  # shape (T, batch_size, nv)

        # swap axis to get (batch_size, T, dim)
        q_log  = jnp.swapaxes(q_log, 0, 1)  # shape (batch_size, T, nq)
        v_log  = jnp.swapaxes(v_log, 0, 1)  # shape (batch_size, T, nv)
        u_log = jnp.broadcast_to(self.u_zero[:, None, :], (self.batch_size, S, self.nu)) # shape (batch_size, T-1, nu)

        return q_log, v_log, u_log
    

    ######################################### POLICY INPUT ROLLOUT #########################################

    # rollout closed loop using RL policy (thin wrapper to allow usage of jitted functions)
    def rollout_policy_input(self, T):
        """
        Perform rollout with inputs from RL policy.

        Args:
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu)
        """

        # check that policy function is available
        if (self.policy_fn_batched is None) or (self.obs_fn_batched is None):
            raise ValueError("Policy or Observation function is not set.")

        # sample initial conditions
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()

        # perform rollout
        q_log, v_log, u_log, c_log = self.rollout_policy_input_jit(q0_batch, v0_batch, T)

        # perform rollout
        return q_log, v_log, u_log, c_log

    # rollout closed loop using RL policy (pure function to jit)
    def _rollout_policy_input(self, q0_batch, v0_batch, T):
        """
        Perform rollout with inputs from RL policy.
        Args:
            q0_batch: jnp.array, initial positions (batch_size, nq)
            v0_batch: jnp.array, initial velocities (batch_size, nv)
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu)
            c_log: jnp.array, logged contact pairs (batch_size, T, nc)
        """

        # number of integration steps
        S = T - 1

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q0, v0: self.mjx_data.replace(qpos=q0, qvel=v0))(q0_batch, v0_batch)

        # start with zero action 
        u0 = jnp.zeros((self.batch_size, self.nu), dtype=jnp.float32)  # (batch_size, nu) zeros

        # main step body
        def body(carry, _):

            # unpack carry
            data, u_curr, t = carry

            # update control input at specified decimation
            def compute_control(_):
                obs = self.obs_fn_batched(data)    # get observations          (batch_size, obs_dim)
                act = self.policy_fn_batched(obs)  # get actions from policy   (batch_size, nu)
                return act

            # if time to update control input
            u_next = lax.cond((t % self.control_decimation) == 0,
                               compute_control, 
                               lambda _: u_curr, operand=None)

            # apply control and take step
            data = data.replace(ctrl=u_next)
            data = self.step_fn_batched(data)

            # extract contact pairs
            contact = self.parse_contact(data)

            return (data, u_next, t + 1), (data.qpos, data.qvel, u_next, contact)

        # do the forward propagation
        (data_last, u_last, _), (q_log, v_log, u_log, c_log) = lax.scan(body, (data_0, u0, 0), None, length=S)

        # add the initial condition to the logs
        q0 = data_0.qpos   # initial q
        v0 = data_0.qvel   # initial v 
        c0 = self.parse_contact(data_0)  # (batch_size, nc)
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)     # shape (T, batch_size, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)     # shape (T, batch_size, nv)
        c_log = jnp.concatenate((c0[None, ...], c_log), axis=0)  # (T, batch_size, nc)

        # swap axis to get (batch, T, dim)
        q_log  = jnp.swapaxes(q_log, 0, 1)  # shape (batch_size, T, nq)
        v_log  = jnp.swapaxes(v_log, 0, 1)  # shape (batch_size, T, nv)
        u_log  = jnp.swapaxes(u_log, 0, 1)  # shape (batch_size, T-1, nu)
        c_log = jnp.swapaxes(c_log, 0, 1)  # (batch_size, T, nc)

        return q_log, v_log, u_log, c_log


    ######################################### RANDOM INPUT ROLLOUT #########################################

    # rollout with random input sequence (thin wrapper to allow usage of jitted functions)
    def rollout_random_input(self, T):
        """
        Perform rollout with random input sequence.
        Args:
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu)
        """
    
        # sample initial conditions
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()

        # sample random input sequence
        u_seq_batch = self.sample_random_uniform_inputs(T-1)

        # perform rollout
        q_log, v_log, u_log = self.rollout_random_input_jit(q0_batch, v0_batch, u_seq_batch, T)

        # perform rollout
        return q_log, v_log, u_log

    # rollout with random input sequence (pure function to jit)
    def _rollout_random_input(self, q0_batch, v0_batch, u_seq_batch, T):
        """
        Perform rollout with random input sequence.
        Args:
            q0_batch: jnp.array, initial positions (batch_size, nq)
            v0_batch: jnp.array, initial velocities (batch_size, nv)
            u_seq_batch: jnp.array, input sequence (batch_size, T-1, nu)
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu) 
        """

        # number of integration steps
        S = T - 1

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q0, v0: self.mjx_data.replace(qpos=q0, qvel=v0))(q0_batch, v0_batch)

        # swap axis to get (T-1, batch, nu) for lax.scan
        u_seq_batch_swapped = jnp.swapaxes(u_seq_batch, 0, 1)  # (T-1, batch_size, nu)

        # main step body
        def body(data, u_t):

            # apply control and take step
            data = data.replace(ctrl=u_t)
            data = self.step_fn_batched(data)

            return data, (data.qpos, data.qvel, u_t)

        # forward propagation
        data_last, (q_log, v_log, u_log) = lax.scan(body, data_0, u_seq_batch_swapped, length=S)

        # add the initial condition to the logs
        q0 = data_0.qpos   # initial q
        v0 = data_0.qvel   # initial v
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)  # shape (T, batch_size, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)  # shape (T, batch_size, nv)

        # swap axis to get (batch_size, T, dim)
        q_log  = jnp.swapaxes(q_log, 0, 1)  # shape (batch_size, T, nq)
        v_log  = jnp.swapaxes(v_log, 0, 1)  # shape (batch_size, T, nv)
        u_log  = u_seq_batch                # shape (batch_size, T-1, nu)

        return q_log, v_log, u_log
    
    
    ######################################### UTILS #########################################

    def parse_contact(self, data_batch):
        """
        Extract touch sensor values for a batch of mjx.Data.

        Args:
            data_batch: mjx.Data, batched mjx data (batch_size, ...)

        Returns:
            jnp.ndarray, shape (batch_size, nc)
            Contact forces at each touch sensor site.
        """
        if self.nc == 0:
            # no touch sensors defined
            return jnp.zeros((self.batch_size, 0), dtype=jnp.float32)
        
        # data_batch.sensordata has shape (batch_size, nsensor)
        return data_batch.sensordata[:, self.touch_sensor_ids]
       
        
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

    # trajectory length
    T = 1000

    # choose environment, policy parameters, and state space bounds
    # env_name = "cart_pole"
    # params_path = "./rl/policy/cart_pole_policy.pkl"
    # q_lb = jnp.array([-1.0, -jnp.pi])  # cartpole
    # q_ub = jnp.array([ 1.0,  jnp.pi])  
    # v_lb = jnp.array([-5.0, -6.0])  
    # v_ub = jnp.array([ 5.0,  6.0])

    # env_name = "acrobot"
    # params_path = "./rl/policy/acrobot_policy.pkl"
    # q_lb = jnp.array([-jnp.pi, -jnp.pi])  # acrobot
    # q_ub = jnp.array([ jnp.pi,  jnp.pi])  
    # v_lb = jnp.array([-3.0, -3.0])  
    # v_ub = jnp.array([ 3.0,  3.0])

    env_name = "paddle_ball"
    params_path = "./rl/policy/paddle_ball_policy.pkl"
    q_lb = jnp.array([ 1.0,  0.1]) # paddle ball
    q_ub = jnp.array([ 3.0,  0.9])
    v_lb = jnp.array([-5.0, -5.0])
    v_ub = jnp.array([ 5.0,  5.0])

    # env_name = "hopper"
    # params_path = "./rl/policy/hopper_policy_2025_09_22_18_38_41.pkl"
    # q_lb = jnp.array([-0.001, 1.0, -jnp.pi, -0.3])  # hopper
    # q_ub = jnp.array([ 0.001, 1.5,  jnp.pi,  0.3])  
    # v_lb = jnp.array([-2.0, -2.0, -3.0, -5.0])
    # v_ub = jnp.array([ 2.0,  2.0,  3.0,  5.0])

    # assign the state bounds
    state_bounds = (q_lb, q_ub, v_lb, v_ub)

    # make the config
    config = ParallelSimConfig(env_name=env_name,
                               batch_size=batch_size,
                               state_bounds=state_bounds,
                               rng=rng,
                               policy_params_path=params_path)
    
    # create the rollout instance
    r = ParallelSim(config)

    # choose the type of rollout
    # rollout_fn = r.rollout_zero_input
    rollout_fn = r.rollout_policy_input
    # rollout_fn = r.rollout_random_input

    # rollout with chosen inputs
    time_0 = time.time()
    q_log_1, v_log_1, u_log_1, c_log_1 = rollout_fn(T)
    q_log_1.block_until_ready()
    v_log_1.block_until_ready()
    u_log_1.block_until_ready()
    c_log_1.block_until_ready()
    time_1 = time.time()
    print(f"Rollout with chosen inputs took (first): {(time_1-time_0):.3f}s")

    time_0 = time.time()
    q_log_2, v_log_2, u_log_2, c_log_2 = rollout_fn(T)
    q_log_2.block_until_ready()
    v_log_2.block_until_ready()
    u_log_2.block_until_ready()
    c_log_2.block_until_ready()
    time_1 = time.time()
    print(f"Rollout with chosen inputs took (steady): {(time_1-time_0):.3f}s")

    time_0 = time.time()
    q_log_3, v_log_3, u_log_3, c_log_3 = rollout_fn(T)
    q_log_3.block_until_ready()
    v_log_3.block_until_ready()
    u_log_3.block_until_ready()
    c_log_3.block_until_ready()
    time_1 = time.time()
    print(f"Rollout with chosen inputs took (steady): {(time_1-time_0):.3f}s")

    print(f"q_log_1 shape: {q_log_1.shape}")
    print(f"v_log_1 shape: {v_log_1.shape}")
    print(f"u_log_1 shape: {u_log_1.shape}")
    print(f"c_log_1 shape: {c_log_1.shape}")
    print(f"q_log_2 shape: {q_log_2.shape}")
    print(f"v_log_2 shape: {v_log_2.shape}")
    print(f"u_log_2 shape: {u_log_2.shape}")
    print(f"c_log_2 shape: {c_log_2.shape}")
    print(f"q_log_3 shape: {q_log_3.shape}")
    print(f"v_log_3 shape: {v_log_3.shape}")
    print(f"u_log_3 shape: {u_log_3.shape}")
    print(f"c_log_3 shape: {c_log_3.shape}")

    q_log_1 = np.array(q_log_1)
    v_log_1 = np.array(v_log_1)
    u_log_1 = np.array(u_log_1)
    c_log_1 = np.array(c_log_1)
    q_log_2 = np.array(q_log_2)
    v_log_2 = np.array(v_log_2)
    u_log_2 = np.array(u_log_2)
    c_log_2 = np.array(c_log_2)
    q_log_3 = np.array(q_log_3)
    v_log_3 = np.array(v_log_3)
    u_log_3 = np.array(u_log_3)
    c_log_3 = np.array(c_log_3)
    
    q_err_1 = np.linalg.norm(q_log_1 - q_log_2)
    q_err_2 = np.linalg.norm(q_log_2 - q_log_3)
    q_err_3 = np.linalg.norm(q_log_1 - q_log_3)
    v_err_1 = np.linalg.norm(v_log_1 - v_log_2)
    v_err_2 = np.linalg.norm(v_log_2 - v_log_3)
    v_err_3 = np.linalg.norm(v_log_1 - v_log_3)
    u_err_1 = np.linalg.norm(u_log_1 - u_log_2)
    u_err_2 = np.linalg.norm(u_log_2 - u_log_3)
    u_err_3 = np.linalg.norm(u_log_1 - u_log_3)
    c_err_1 = np.linalg.norm(c_log_1 - c_log_2)
    c_err_2 = np.linalg.norm(c_log_2 - c_log_3)
    c_err_3 = np.linalg.norm(c_log_1 - c_log_3)

    print(f"q_err_1: {q_err_1:.6e}, q_err_2: {q_err_2:.6e}, q_err_3: {q_err_3:.6e}")
    print(f"v_err_1: {v_err_1:.6e}, v_err_2: {v_err_2:.6e}, v_err_3: {v_err_3:.6e}")
    print(f"u_err_1: {u_err_1:.6e}, u_err_2: {u_err_2:.6e}, u_err_3: {u_err_3:.6e}")
    print(f"c_err_1: {c_err_1:.6e}, c_err_2: {c_err_2:.6e}, c_err_3: {c_err_3:.6e}")


    # save the data
    q_log_np = np.array(q_log_1)
    v_log_np = np.array(v_log_1)
    u_log_np = np.array(u_log_1)
    c_log_np = np.array(c_log_1)

    print(q_log_np.shape)
    print(v_log_np.shape)
    print(u_log_np.shape)
    print(c_log_np.shape)

    # save the data
    robot_name = r.env.robot_name
    save_path = f"./data/{robot_name}_data.npz"
    np.savez(save_path, q_log=q_log_np, v_log=v_log_np, u_log=u_log_np, c_log=c_log_np)
    print(f"Saved data to: {save_path}")
