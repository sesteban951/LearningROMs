# standard imports
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

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig
from algorithms.ppo_play import PPO_Play

#################################################################

if __name__ == "__main__":

    #----------------------- POLICY SETUP -----------------------#

    # Load the environment
    env = envs.get_environment("cart_pole")
    
    # Path to the trained policy parameters
    params_path = "./rl/policy/cart_pole_policy_2025_08_28_09_45_20.pkl"

    # Create the PPO_Play object
    ppo_player = PPO_Play(env, params_path)

    # get the jitted policy and obs functions
    policy_fn, obs_fn = ppo_player.policy_and_obs_functions()

    #------------------------- SIMULATION -------------------------#

    # import the mujoco model
    config = CartPoleConfig()
    model_path = config.model_path
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # compute some sim and control parameters
    sim_step_dt = mjx_model.opt.timestep
    sim_steps_per_ctrl = config.physics_steps_per_control_step
    sim_step_counter = 0

    # initial state
    mj_data.qpos = np.zeros(mj_model.nq)
    mj_data.qvel = np.zeros(mj_model.nv)
    mj_data.qpos[0] = 0.0     # cart position
    mj_data.qpos[1] = np.pi   # pole angle

    # wall clock timing variables
    t_sim = 0.0
    wall_start = time.time()
    last_render = 0.0

    # start the interactive simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        # Set camera parameters
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])   # look at x, y, z
        viewer.cam.distance = 3.0                           # distance from lookat
        viewer.cam.elevation = -20.0                        # tilt down/up
        viewer.cam.azimuth = 90.0                           # rotate around lookat

        while viewer.is_running():

            # get the current sim time and state
            t_sim = mj_data.time

            # query controller at the desired rate
            if sim_step_counter % sim_steps_per_ctrl == 0:

                print(f"Sim Time: {t_sim:.3f} s")

                # get current state
                qpos = jnp.array(mj_data.qpos)
                qvel = jnp.array(mj_data.qvel)

                # update the mjx_data
                mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

                # compute the observation
                obs = obs_fn(mjx_data)   # obs is a jax array

                # compute the action
                act = policy_fn(obs)  # act is a jax array
            
                # update the controls
                mj_data.ctrl[:] = np.array(act)

            # increment counter
            sim_step_counter += 1

            # step the simulation
            mujoco.mj_step(mj_model, mj_data)

            # sync the viewer
            viewer.sync()

            # sync the sim time with the wall clock time
            wall_elapsed = time.time() - wall_start
            if t_sim > wall_elapsed:
                time.sleep(t_sim - wall_elapsed)
