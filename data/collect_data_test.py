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
import jax.numpy as jnp

# brax imports
from brax import envs

# mujoco imports
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer

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

    # get the enviornment config
    config = env.config

    # import the mujoco model
    model_path = config.model_path
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # sample initial states
    key_name = "default"
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    q0 = mj_model.key_qpos[key_id]
    v0 = mj_model.key_qvel[key_id]

    # simulation parameters
    sim_dt = mj_model.opt.timestep                                  # sim timestep
    control_decimation = env.config.physics_steps_per_control_step  # sim steps per control update

    # simulation parameters
    batch_size = 8
    t_max = 5.0
    num_sim_steps = round(t_max / sim_dt)

    # Create multiple copies of q0 and v0 for parallel simulation
    q0_batch = np.tile(q0, (batch_size, 1))  # shape (batch_size, nq)
    v0_batch = np.tile(v0, (batch_size, 1))  # shape (batch_size, nv)
    q0_batch = jnp.array(q0_batch)           # shape (batch_size, nq)
    v0_batch = jnp.array(v0_batch)           # shape (batch_size, nv)

    # allocate solution arrays
    q_traj = np.zeros((batch_size, num_sim_steps, mj_model.nq)) # (batch, time_size, nq)
    v_traj = np.zeros((batch_size, num_sim_steps, mj_model.nv)) # (batch, time_size, nv)
    u_traj = np.zeros((batch_size, num_sim_steps, mj_model.nu)) # (batch, time_size, nu)

    # convert mujoco model and data to jax
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # # create the policy and observation function here
    # ppo_player = PPO_Play(env, params_path)
    # policy_fn, obs_fn = ppo_player.policy_and_obs_functions()


    # # main loop
    # mjx_data = mjx.make_data(mjx_model, q0_batch, v0_batch)

