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

def pos_act_to_ctrl(dq, data, env):
    """
    Convert Δq action -> absolute q_des -> PD torque -> actuator ctrl.
    Args:
        dq:   Δq action (jax/np array, shape (env.action_size,))
        data: mjx_data or mj_data (must have .qpos and .qvel), pre-step state
        env:  BipedEnv
    Returns:
        ctrl: actuator controls (np array) to write into mj_data.ctrl
    """
    # conver tot jnp array
    dq = jnp.asarray(dq)
    # Δq -> q_des (around standing), clipped to joint limits
    q_des = env._action_to_q_des(dq)
    # PD (uses pre-step state) -> torque -> ctrl (clipped)
    ctrl, _tau = env._q_des_to_torque(data, q_des)

    return np.asarray(ctrl)

#################################################################

if __name__ == "__main__":

    #----------------------------- POLICY IMPORT -----------------------------#

    # Load the environment and policy parameters
    # env = envs.get_environment("cart_pole")
    # # policy_data_path = "./rl/policy/cart_pole_policy.pkl"
    # policy_data_path = "./rl/policy/cart_pole_policy_2025_09_25_18_03_01.pkl"

    # Load the environment and policy parameters
    # env = envs.get_environment("acrobot")
    # # policy_data_path = "./rl/policy/acrobot_policy.pkl"
    # policy_data_path = "./rl/policy/acrobot_policy_2025_09_25_18_26_12.pkl"

    # Load the environment and policy parameters
    # env = envs.get_environment("paddle_ball")
    # # policy_data_path = "./rl/policy/paddle_ball_policy.pkl"
    # policy_data_path = "./rl/policy/paddle_ball_policy_2025_09_25_20_03_56.pkl"

    # Load the environment and policy parameters
    # env = envs.get_environment("hopper")
    # policy_data_path = "./rl/policy/hopper_policy.pkl"

    # Load the environment and policy parameters
    # # env = envs.get_environment("biped_basic")
    env = envs.get_environment("biped")
    policy_data_path = "./rl/policy/biped_policy_2025_09_28_17_34_56.pkl"

    #----------------------------- POLICY SETUP -----------------------------#

    # Create the PPO_Play object
    ppo_player = PPO_Play(env, policy_data_path)

    # get the jitted policy and obs functions
    policy_fn, obs_fn = ppo_player.policy_and_obs_functions()

    #------------------------------- SIMULATION -------------------------------#

    # import the mujoco model
    model_path = env.config.model_path
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # compute some sim and control parameters
    sim_step_dt = mjx_model.opt.timestep
    sim_steps_per_ctrl = env.config.physics_steps_per_control_step
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

    # # initial action and control
    # act = np.zeros(env.action_size, dtype=np.float32)  # initial action
    # ctrl = np.zeros(mj_model.nu, dtype=np.float32)     # initial control

    # # start the interactive simulation
    # with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

    #     # Set camera parameters
    #     viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])   # look at x, y, z
    #     viewer.cam.distance = 3.0                           # distance from lookat
    #     viewer.cam.elevation = -20.0                        # tilt down/up
    #     viewer.cam.azimuth = 90.0                           # rotate around lookat

    #     while viewer.is_running():

    #         # get the current sim time and state
    #         t_sim = mj_data.time

    #         print(f"Sim Time: {t_sim:.3f} s")

    #         # parse contact information
    #         contact = parse_contact(mj_data)
    #         # print(act)

    #         # query controller at the desired rate
    #         if sim_step_counter % sim_steps_per_ctrl == 0:

    #             print(f"Sim Time: {t_sim:.3f} s")

    #             # get current state
    #             qpos = jnp.array(mj_data.qpos)
    #             qvel = jnp.array(mj_data.qvel)

    #             # update the mjx_data
    #             mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

    #             # compute the observation
    #             if env.robot_name == "biped":
    #                 # for biped, pass in previous action as well
    #                 obs = obs_fn(mjx_data, act)   # obs is a jax array
    #             else:
    #                 obs = obs_fn(mjx_data)      # obs is a jax array

    #             # compute the action
    #             act = policy_fn(obs)  # act is a jax array

    #             # biped case the action is position based, convert to control
    #             if env.robot_name == "biped":
    #                 act = np.array(act)      # convert to numpy array
    #                 ctrl = pos_act_to_ctrl(act, mj_data, env)
    #                 ctrl = np.array(ctrl)      # convert to numpy array
    #             else:
                    
    #                 ctrl = np.array(act)      # convert to numpy array
            
    #                 # update the controls
    #                 mj_data.ctrl[:] = ctrl

    #         # increment counter
    #         sim_step_counter += 1

    #         # step the simulation
    #         mujoco.mj_step(mj_model, mj_data)

    #         # sync the viewer
    #         viewer.sync()

    #         # sync the sim time with the wall clock time
    #         wall_elapsed = time.time() - wall_start
    #         if t_sim > wall_elapsed:
    #             time.sleep(t_sim - wall_elapsed)

        # initial action (Δq) and control
    dq_prev = np.zeros(env.action_size, dtype=np.float32)  # previous Δq
    ctrl = np.zeros(mj_model.nu, dtype=np.float32)         # actuator control

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # ... camera setup ...

        while viewer.is_running():
            t_sim = mj_data.time

            # query controller at the desired rate
            if sim_step_counter % sim_steps_per_ctrl == 0:
                # sync mjx_data with current MuJoCo state (PRE-STEP state!)
                qpos = jnp.array(mj_data.qpos)
                qvel = jnp.array(mj_data.qvel)
                mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

                # build observation
                if env.robot_name == "biped":
                    obs = obs_fn(mjx_data, dq_prev)     # pass previous Δq
                else:
                    obs = obs_fn(mjx_data)

                # policy action
                action = policy_fn(obs)                 # jax array

                if env.robot_name == "biped":
                    # action is Δq → convert to actuator control
                    dq = np.array(action)               # Δq for logging/next obs
                    ctrl = pos_act_to_ctrl(dq, mjx_data, env)   # use PRE-STEP mjx_data
                    mj_data.ctrl[:] = ctrl              # send controls
                    dq_prev = dq                        # keep Δq for next obs
                else:
                    # torque-control envs: send directly
                    ctrl = np.array(action)
                    mj_data.ctrl[:] = ctrl

            sim_step_counter += 1
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            # (optional) wall-clock pacing
            wall_elapsed = time.time() - wall_start
            if t_sim > wall_elapsed:
                time.sleep(t_sim - wall_elapsed)