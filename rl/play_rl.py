##
#
#  Simple Script to Simulate a trained policy in Mujoco
#  
##

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
from envs.cart_pole_env import CartPoleEnv
from envs.acrobot_env import AcrobotEnv
from envs.biped_env import BipedEnv
from envs.biped_basic_env import BipedBasicEnv
from envs.hopper_env import HopperEnv
from envs.paddle_ball_env import PaddleBallEnv
from algorithms.ppo_play import PPO_Play


#################################################################

# function to parse contact information
def parse_contact(data):

    # get the contact information
    num_contacts = data.ncon

    # contact boolean 
    foot_in_contact = False

    # either "torso" or "foot" in contact
    if num_contacts > 0:

        for i in range(num_contacts):

            # get the contact id
            contact_id = i

            # get the geom ids
            geom1_id = data.contact[contact_id].geom1
            geom2_id = data.contact[contact_id].geom2

            # get the geom names
            geom1_name = mujoco.mj_id2name(data.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(data.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            print(f"Contact {i}: {geom1_name} ({geom1_id}) and {geom2_name} ({geom2_id})")

    return foot_in_contact

#################################################################

if __name__ == "__main__":

    #----------------------- POLICY IMPORT -----------------------#

    # Load the environment and policy parameters
    env = envs.get_environment("cart_pole")
    # policy_data_path = "./rl/policy/cart_pole_policy.pkl"
    policy_data_path = "./rl/policy/cart_pole_policy_2025_09_25_18_03_01.pkl"

    # Load the environment and policy parameters
    # env = envs.get_environment("acrobot")
    # policy_data_path = "./rl/policy/acrobot_policy.pkl"

    # Load the environment and policy parameters
    # env = envs.get_environment("paddle_ball")
    # policy_data_path = "./rl/policy/paddle_ball_policy.pkl"

    # Load the environment and policy parameters
    # env = envs.get_environment("hopper")
    # policy_data_path = "./rl/policy/hopper_policy.pkl"

    # Load the environment and policy parameters
    # env = envs.get_environment("biped")
    # # env = envs.get_environment("biped_basic")
    # policy_data_path = "./rl/policy/biped_policy_2025_09_24_15_31_07.pkl"

    #----------------------- POLICY SETUP -----------------------#

    # get the environment config
    config = env.config

    # Create the PPO_Play object
    ppo_player = PPO_Play(env, policy_data_path)

    # get the jitted policy and obs functions
    policy_fn, obs_fn = ppo_player.policy_and_obs_functions()

    #------------------------- SIMULATION -------------------------#

    # import the mujoco model
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
    key_name = "default"
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    mj_data.qpos = mj_model.key_qpos[key_id]
    mj_data.qvel = mj_model.key_qvel[key_id]

    # wall clock timing variables
    t_sim = 0.0
    wall_start = time.time()
    last_render = 0.0

    act = np.zeros(mj_model.nu)

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

            print(f"Sim Time: {t_sim:.3f} s")

            # parse contact information
            contact = parse_contact(mj_data)

            # query controller at the desired rate
            if sim_step_counter % sim_steps_per_ctrl == 0:

                print(f"Sim Time: {t_sim:.3f} s")

                # get current state
                qpos = jnp.array(mj_data.qpos)
                qvel = jnp.array(mj_data.qvel)

                # update the mjx_data
                mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

                # compute the observation
                # obs = obs_fn(mjx_data, act)   # obs is a jax array
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
