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

# for importing policy
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# custom imports
from envs.cart_pole_env import CartPoleEnv
from envs.cart_pole_tracking_env import CartPoleTrackingEnv
from envs.acrobot_env import AcrobotEnv
from envs.biped_env import BipedEnv
from envs.biped_basic_env import BipedBasicEnv
from envs.hopper_env import HopperEnv
from envs.paddle_ball_env import PaddleBallEnv
from algorithms.ppo_play import PPO_Play
from utils.utils import Joy


#################################################################

# function to parse contact information
def parse_contact(data):
    """
    Parse contact information from the Mujoco simulation data.

    Args:
        data: mjx_data or mj_data (must have .contact, .ncon, .model, etc.)
    Returns:
        contact: bool, True if a geom is in contact, else False
    """

    # get the contact information
    num_contacts = data.ncon

    # contact boolean 
    contact = False

    # if there are contacts, print them
    if num_contacts > 0:

        # set contact boolean
        contact = True

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

    return contact

# function to convert biped position action to mujoco control
def pos_act_to_ctrl(action, data, env):
    """
    Convert Δq action -> absolute q_des -> PD torque -> actuator ctrl.
    Args:
        action:   Δq action (jax/np array, shape (env.action_size,))
        data: mjx_data or mj_data (must have .qpos and .qvel), pre-step state
        env:  BipedEnv
    Returns:
        ctrl: actuator controls (np array) to write into mj_data.ctrl
    """

    # convert to jnp array
    action = jnp.asarray(action)

    # Δq -> q_des (around standing), clipped to joint limits
    q_des = env._action_to_q_des(action)

    # PD (uses pre-step state) -> torque -> ctrl (clipped)
    ctrl, _tau = env._q_des_to_torque(data, q_des)

    # convert to np array
    ctrl = np.asarray(ctrl)

    return ctrl

#################################################################

if __name__ == "__main__":

    #----------------------------- POLICY IMPORT -----------------------------#

    # CART POLE
    # env = envs.get_environment("cart_pole")
    # policy_data_path = "./rl/policy/cart_pole_policy.pkl"
    env = envs.get_environment("cart_pole_tracking")
    policy_data_path = "./rl/policy/cart_pole_tracking_policy_2025_10_15_16_54_25.pkl"

    # ACROBOT
    # env = envs.get_environment("acrobot")
    # policy_data_path = "./rl/policy/acrobot_policy.pkl"

    # PADDLE BALL
    # env = envs.get_environment("paddle_ball")
    # policy_data_path = "./rl/policy/paddle_ball_policy.pkl"

    # HOPPER
    # env = envs.get_environment("hopper")
    # policy_data_path = "./rl/policy/hopper_policy.pkl"

    # BIPED
    # # env = envs.get_environment("biped_basic")
    # env = envs.get_environment("biped")
    # policy_data_path = "./rl/policy/biped_policy.pkl"

    # desired sim dt
    sim_dt = 0.002

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

    # change the simulation dt
    mj_model.opt.timestep = sim_dt
    
    # create the mjx model and data
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

    # check if the env is a tracking environment
    env_name = env.env_name
    is_tracking_env = "tracking" in env_name.lower()
    if is_tracking_env == True:

        # initialize a joystick if available
        joy = Joy()

        # get the command limits
        cmd_lb = env.config.cmd_lb
        cmd_ub = env.config.cmd_ub

        # set the a nominal command (in case no joystick is connected)
        cmd = 0.0
        
    # initial action and control
    action_prev = np.zeros(env.action_size, dtype=np.float32)  # previous Δq
    ctrl = np.zeros(mj_model.nu, dtype=np.float32)             # actuator control

    # start the interactive simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        
        # Set camera parameters
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])   # look at x, y, z
        viewer.cam.distance = 3.0                           # distance from lookat
        viewer.cam.elevation = -20.0                        # tilt down/up
        viewer.cam.azimuth = 90.0                           # rotate around lookat

        # main simulation loop
        while viewer.is_running():

            # get the current sim time and state
            t_sim = mj_data.time

            # print contact information
            contact = parse_contact(mj_data)

            # query controller at the desired rate
            if sim_step_counter % sim_steps_per_ctrl == 0:

                # print sim time
                print(f"Sim Time: {t_sim:.3f} s")

                # update the desired command
                if joy.isConnected:
                    # update the joystick
                    joy.update()
                    
                    # scale using upper and lower limits, such that cmd ∈ [cmd_lb, cmd_ub]
                    cmd_raw = joy.LS_Y
                    cmd = 0.5*(cmd_ub - cmd_lb) * cmd_raw + 0.5 * (cmd_ub + cmd_lb)

                    # print the tracking
                    print(f"cmd: {cmd:.3f}")

                # sync mjx_data with current MuJoCo state (PRE-STEP state!)
                qpos = jnp.array(mj_data.qpos)
                qvel = jnp.array(mj_data.qvel)
                mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

                # build observation
                if env.robot_name == "biped":
                    obs = obs_fn(mjx_data, action_prev)  # pass previous Δq
                elif is_tracking_env:
                    # manually construct tracking obs: [pos, cosθ, sinθ, vel, θdot, pos_cmd]
                    obs = obs_fn(mjx_data, cmd)
                else:
                    # non-tracking, use the env's usual obs function
                    obs = env._compute_obs(mjx_data)

                # policy action
                action = policy_fn(obs)      # jax array

                # compute the control 
                if env.robot_name == "biped":
                    # action is Δq → convert to torque control
                    dq = np.array(action)                       # Δq for logging/next obs
                    ctrl = pos_act_to_ctrl(dq, mjx_data, env)   # use PRE-STEP mjx_data
                    mj_data.ctrl[:] = ctrl                      # send controls
                    action_prev = dq                            # keep Δq for next obs
                else:
                    # torque-control envs: send directly
                    ctrl = np.array(action)
                    mj_data.ctrl[:] = ctrl

            # increment counter
            sim_step_counter += 1

            # step the simulation
            mujoco.mj_step(mj_model, mj_data)

            # sync the viewer
            viewer.sync()

            # wall-clock pacing
            wall_elapsed = time.time() - wall_start
            if t_sim > wall_elapsed:
                time.sleep(t_sim - wall_elapsed)
