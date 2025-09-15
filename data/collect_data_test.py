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

#################################################################

if __name__ == "__main__":

    # load in the enviorment and policy parameters
    env = envs.get_environment("paddle_ball")
    params_path = "./rl/policy/paddle_ball_policy.pkl"

    # create the policy and observation function here
    ppo_player = PPO_Play(env, params_path)
    policy_fn, obs_fn = ppo_player.policy_and_obs_functions()

    # get the enviornment config
    config = env.config

    # import the mujoco model
    model_path = config.model_path
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # convert mujoco model and data to jax
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # simulation parameters
    sim_dt = mj_model.opt.timestep                                  # sim timestep
    control_decimation = env.config.physics_steps_per_control_step  # sim steps per control update

    # simulation parameters
    batch_size = 1024
    t_max = 3.0
    num_sim_steps = round(t_max / sim_dt)

    print(f"sim_dt: {sim_dt}, control_decimation: {control_decimation}, num_sim_steps: {num_sim_steps}")

    # set some intial state bounds
    q_lb = jnp.array([ 1.0,  0.1])
    q_ub = jnp.array([ 3.0,  0.9])
    v_lb = jnp.array([-5.0, -5.0])
    v_ub = jnp.array([ 5.0,  5.0])

    # sample initial states
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)
    q0_batch = jax.random.uniform(key1, (batch_size, mjx_model.nq), minval=q_lb, maxval=q_ub) # shape (batch, nq)
    v0_batch = jax.random.uniform(key2, (batch_size, mjx_model.nv), minval=v_lb, maxval=v_ub) # shape (batch, nv)

    print(f"q0_batch: {q0_batch}")
    print(f"v0_batch: {v0_batch}")

    # create batched mj data, and set the initial states for each data instance
    mjx_data_batched = jax.vmap(lambda i: mjx_data.replace(qpos=q0_batch[i], qvel=v0_batch[i]))(jnp.arange(batch_size))

    # create batched step function, performs a single mujoco step
    step_fn_batched = jax.jit(jax.vmap(lambda data: mjx.step(mjx_model, data), in_axes=(0))) 

    # create batched observation function
    obs_fn_batched = jax.vmap(obs_fn, in_axes=0) 

    # create policy function
    policy_fn_batched = jax.vmap(policy_fn, in_axes=0)

    # set the model dimensions
    nq = mjx_model.nq
    nv = mjx_model.nv
    nu = mjx_model.nu

    # rollout with zero input
    u_zero = jnp.zeros((batch_size, mjx_model.nu)) # (batch, nu)
    def rollout_zero(data_b, T):
        """
        rollout with zero input
        """

        # prealloc logs on device
        q_log = jnp.empty((batch_size, T, nq), dtype=jnp.float32)
        v_log = jnp.empty((batch_size, T, nv), dtype=jnp.float32)
        u_log = jnp.empty((batch_size, T, nu), dtype=jnp.float32)

        def body(t, carry):

            # unpack carry
            data, ql, vl, ul = carry

            # apply control and step
            data = data.replace(ctrl=u_zero)
            data = step_fn_batched(data)             # step all envs

            # log states
            ql = ql.at[:, t, :].set(data.qpos)       # log q
            vl = vl.at[:, t, :].set(data.qvel)       # log v
            ul = ul.at[:, t, :].set(u_zero)          # log u (constant)

            return (data, ql, vl, ul)

        # loop over time steps
        data_b, q_log, v_log, u_log = lax.fori_loop(0, T, 
                                                    body, 
                                                    (data_b, q_log, v_log, u_log))
        
        return data_b, q_log, v_log, u_log
    
    # rollout with random input sequence
    def rollout_random(data_b, T, rng):
        """
        rollout with random input sequence
        """

        # prealloc logs on device
        q_log = jnp.empty((batch_size, T, nq), dtype=jnp.float32) # (batch, time, nq)
        v_log = jnp.empty((batch_size, T, nv), dtype=jnp.float32) # (batch, time, nv)
        u_log = jnp.empty((batch_size, T, nu), dtype=jnp.float32) # (batch, time, nu)

        # sample control sequence
        key, subkey = jax.random.split(rng)

        # sample random control inputs
        u_lb = -jnp.ones((nu,))     # upper bound is  1.0 vector, shape (nu,)
        u_ub =  jnp.ones((nu,))     # lower bound is -1.0 vector, shape (nu,)
        u_seq = jax.random.uniform(subkey, (batch_size, T, nu), minval=u_lb, maxval=u_ub) # shape (batch, time, nu)

        def body(t, carry):

            # unpack the carry
            data, ql, vl, ul = carry

            # get the current control
            u_t = u_seq[:, t, :]          # shape (batch, nu)

            # apply the control and step
            data = data.replace(ctrl=u_t)
            data = step_fn_batched(data)   

            # log states
            ql = ql.at[:, t, :].set(data.qpos)   # log q
            vl = vl.at[:, t, :].set(data.qvel)   # log v
            ul = ul.at[:, t, :].set(u_t)         # log u

            return (data, ql, vl, ul)
        
        # loop over time steps
        data_b, q_log, v_log, u_log = lax.fori_loop(0, T, 
                                                    body, 
                                                    (data_b, q_log, v_log, u_log))
        
        return data_b, q_log, v_log, u_log

    # rollout with policy
    def rollout_with_policy(data_b, T):
        """
        rollout with policy
        """

        # prealloc logs on device
        q_log = jnp.empty((batch_size, T, nq), dtype=jnp.float32)
        v_log = jnp.empty((batch_size, T, nv), dtype=jnp.float32)
        u_log = jnp.empty((batch_size, T, nu), dtype=jnp.float32)

        # initialize input vector
        u_curr = jnp.zeros((batch_size, nu))

        def body(t, carry):
            
            # unpack carry
            data, u_c, ql, vl, ul = carry

            # compute control only on decimated steps
            def compute_control(_):

                obs = obs_fn_batched(data)           # get observations
                act = policy_fn_batched(obs)           # get action
                return act
            
            # if statement to update control at desired rate
            u_next = lax.cond((t % control_decimation) == 0, 
                              compute_control, 
                              lambda _: u_curr, operand=None)
            
            # apply control and step
            data = data.replace(ctrl=u_next)
            data = step_fn_batched(data)             

            # log states
            ql = ql.at[:, t, :].set(data.qpos)       # log q
            vl = vl.at[:, t, :].set(data.qvel)       # log v
            ul = ul.at[:, t, :].set(u_next)          # log u

            return (data, u_next, ql, vl, ul)

        # loop over time steps
        data_b, u_last, q_log, v_log, u_log = lax.fori_loop(0, T, 
                                                            body, 
                                                            (data_b, u_curr, q_log, v_log, u_log))
        
        return data_b, q_log, v_log, u_log


    seed = int(time.time())
    rng = jax.random.PRNGKey(seed)

    # JIT the whole rollout (T is static for best compile; donate big args to reduce copies)
    # fast_rollout = jax.jit(rollout_zero, static_argnames=("T",))
    fast_rollout = jax.jit(rollout_with_policy, static_argnames=("T",))
    # fast_rollout_rand = jax.jit(rollout_random, static_argnames=("T",))

    # Run once to compile, then it’s fast
    t0 = time.time()
    data_b, q_traj, v_traj, u_traj = fast_rollout(mjx_data_batched, num_sim_steps)
    # data_b, q_traj, v_traj, u_traj = fast_rollout_rand(mjx_data_batched, num_sim_steps, rng)
    jax.block_until_ready(q_traj)
    print(f"[first run incl. compile] {(time.time()-t0):.3f}s")

    t0 = time.time()
    data_b, q_traj, v_traj, u_traj = fast_rollout(mjx_data_batched, num_sim_steps)
    # data_b, q_traj, v_traj, u_traj = fast_rollout_rand(mjx_data_batched, num_sim_steps, rng)
    jax.block_until_ready(q_traj)
    print(f"[steady-state] {(time.time()-t0):.3f}s")

    t0 = time.time()
    data_b, q_traj, v_traj, u_traj = fast_rollout(mjx_data_batched, num_sim_steps)
    # data_b, q_traj, v_traj, u_traj = fast_rollout_rand(mjx_data_batched, num_sim_steps, rng)
    jax.block_until_ready(q_traj)
    print(f"[steady-state] {(time.time()-t0):.3f}s")

    t0 = time.time()
    data_b, q_traj, v_traj, u_traj = fast_rollout(mjx_data_batched, num_sim_steps)
    # data_b, q_traj, v_traj, u_traj = fast_rollout_rand(mjx_data_batched, num_sim_steps, rng)
    jax.block_until_ready(q_traj)
    print(f"[steady-state] {(time.time()-t0):.3f}s")

    # convert to numpy for saving
    q_traj = np.array(q_traj)
    v_traj = np.array(v_traj)
    u_traj = np.array(u_traj)

    print(q_traj.shape)
    print(v_traj.shape)
    print(u_traj.shape)

    # save the data
    save_path = "./data/paddle_ball_data.npz"
    np.savez(save_path, q_traj=q_traj, v_traj=v_traj, u_traj=u_traj)
    print(f"Saved data to: {save_path}")
