# standard imports
import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# mujoco imports
import mujoco
import mujoco.viewer

# brax imports
from brax import envs

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


################################################################################
# UTILS
################################################################################

def load_touch_sensors(mj_model):
    """
    Load all touch sensors from a MuJoCo model.

    Args:
        mj_model: mujoco.MjModel

    Returns:
        touch_sensor_ids: list of int, indices of touch sensors
        touch_sensor_names: list of str, names of touch sensors
        num_touch_sensors: int, number of touch sensors
    """
    touch_sensor_ids = []
    touch_sensor_names = []

    for i, stype in enumerate(mj_model.sensor_type):
        if stype == mujoco.mjtSensor.mjSENS_TOUCH:
            touch_sensor_ids.append(i)
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            touch_sensor_names.append(name)

    num_touch_sensors = len(touch_sensor_ids)

    print(f"Found {num_touch_sensors} touch sensors: {touch_sensor_names}")
    return touch_sensor_ids, touch_sensor_names, num_touch_sensors


################################################################################
# MAIN PLOTTING
################################################################################

if __name__ == "__main__":

    # choose the environment
    # env = envs.get_environment("cart_pole")
    # env = envs.get_environment("acrobot")
    # env = envs.get_environment("paddle_ball")
    env = envs.get_environment("hopper")
    config = env.config
    robot_name = env.robot_name

    # load the mujoco model
    model_path = config.model_path
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # load touch sensors if available
    touch_sensor_ids, touch_sensor_names, num_touch_sensors = load_touch_sensors(mj_model)

    # load the data
    file_name = f"./data/{robot_name}_data.npz"
    data = np.load(file_name)

    # access the arrays
    q_traj = data['q_log']
    v_traj = data['v_log']
    u_traj = data['u_log']
    c_traj = data['c_log']

    # get the shape of the data
    batch_size, N_state, nq = q_traj.shape
    _, _, nv = v_traj.shape
    _, N_input, nu = u_traj.shape
    _, _, nc = c_traj.shape

    # print log shapes
    print(f"Full q_traj shape: {q_traj.shape}")
    print(f"Full v_traj shape: {v_traj.shape}")
    print(f"Full u_traj shape: {u_traj.shape}")
    print(f"Full c_traj shape: {c_traj.shape}")

    # percent segments of the trajectory to use
    traj_segment_percent = (0.0, 1.0)  # (start, end) as percent of trajectory length
    start_idx = int(traj_segment_percent[0] * N_state)
    end_idx = int(traj_segment_percent[1] * N_state)
    q_traj = q_traj[:, start_idx:end_idx, :]
    v_traj = v_traj[:, start_idx:end_idx, :]
    u_traj = u_traj[:, start_idx:end_idx, :]
    c_traj = c_traj[:, start_idx:end_idx, :]

    N_state = q_traj.shape[1]
    N_input = u_traj.shape[1]
    print(f"Using trajectory segment from {start_idx} to {end_idx} (N_state = {N_state})")

    # select one random trajectory to playback
    traj_idx = np.random.randint(batch_size)

    print(f"Playing back trajectory {traj_idx} of {batch_size}")

    sim_dt = env.sys.opt.timestep

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

        # show the initial state
        mj_data.qpos[:] = q_traj[traj_idx, 0]
        mj_data.qvel[:] = v_traj[traj_idx, 0]
        mj_data.ctrl[:] = u_traj[traj_idx, 0]

        mujoco.mj_forward(mj_model, mj_data)  # recompute derived quantities
        viewer.sync()

        while viewer.is_running():

            # get the current sim time and state
            t_sim = mj_data.time

            # print(f"Sim Time: {t_sim:.3f} s")

            # fix the state
            step_idx = int(t_sim / sim_dt)

            if step_idx >= N_state:
                break
            
            # hardcode the trajectory state for playback
            mj_data.qpos = q_traj[traj_idx, step_idx]
            mj_data.qvel = v_traj[traj_idx, step_idx]

            # step the simulation
            mujoco.mj_step(mj_model, mj_data)

            # sync the viewer
            viewer.sync()

            # sync the sim time with the wall clock time
            wall_elapsed = time.time() - wall_start
            if t_sim > wall_elapsed:
                time.sleep(t_sim - wall_elapsed)

    # pull another n_plot random trajectories and plot them 
    n_plot = 20
    n_plot = min(n_plot, batch_size)

    # time vector (N steps at dt)
    t_state = np.arange(N_state) * sim_dt
    t_input = np.arange(N_input) * sim_dt
    t_contact = t_state


    # number of rows = max(nq, nc) so we can fit all signals
    nrows = max(nq, nc)

    # figure and axes: nrows x 4 (q, v, u, c)
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 2.2 * nrows), sharex=True)
    if nrows == 1:
        axes = axes[None, :]  # ensure 2D indexing

    for i in range(nrows):
            ax_q, ax_v, ax_u, ax_c = axes[i, 0], axes[i, 1], axes[i, 2], axes[i, 3]

            # --- Column 0: q[i] ---
            if i < nq:
                for _ in range(n_plot):
                    idx = np.random.randint(batch_size)
                    ax_q.plot(t_state, q_traj[idx, :, i], alpha=0.5)
                ax_q.set_ylabel(f"q[{i}]")
                ax_q.set_xlabel("Time [s]")        
                ax_q.tick_params(labelbottom=True) 
                if i == 0:
                    ax_q.set_title("Positions (q)")
            else:
                ax_q.axis("off")
            ax_q.grid(True, alpha=0.3)

            # --- Column 1: v[i] ---
            if i < nv:
                for _ in range(n_plot):
                    idx = np.random.randint(batch_size)
                    ax_v.plot(t_state, v_traj[idx, :, i], alpha=0.5)
                ax_v.set_ylabel(f"v[{i}]")
                ax_v.set_xlabel("Time [s]")        
                ax_v.tick_params(labelbottom=True) 
                if i == 0:
                    ax_v.set_title("Velocities (v)")
            else:
                ax_v.axis("off")
            ax_v.grid(True, alpha=0.3)

            # --- Column 2: u[i] ---
            if i < nu:
                for _ in range(n_plot):
                    idx = np.random.randint(batch_size)
                    ax_u.plot(t_input, u_traj[idx, :, i], alpha=0.5)
                ax_u.set_ylabel(f"u[{i}]")
                ax_u.set_xlabel("Time [s]")
                ax_u.tick_params(labelbottom=True)  
                if i == 0:
                    ax_u.set_title("Controls (u)")
            else:
                ax_u.axis("off")
            ax_u.grid(True, alpha=0.3)

            # --- Column 3: c[i] ---
            if i < nc:
                for _ in range(n_plot):
                    idx = np.random.randint(batch_size)
                    ax_c.plot(t_contact, c_traj[idx, :, i], alpha=0.5)
                if i < len(touch_sensor_names):
                    ax_c.set_title(touch_sensor_names[i])
                else:
                    ax_c.set_title(f"c[{i}]")
                ax_c.set_ylabel(f"c[{i}]")
                ax_c.set_xlabel("Time [s]")
                ax_c.tick_params(labelbottom=True)   
            else:
                ax_c.axis("off")
            ax_c.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # plot the phase for each variable
    fig, axes = plt.subplots(nrows=nq, ncols=1, figsize=(7, 2.2 * nq), sharex=True)
    if nq == 1:
        axes = axes[None, :]  # ensure 2D indexing when nq == 1 
    for i in range(nq):
        ax = axes[i]
        if i < nv:
            for _ in range(n_plot):
                idx = np.random.randint(batch_size)
                ax.plot(q_traj[idx, :, i], v_traj[idx, :, i], alpha=0.5)
            ax.set_xlabel(f"q[{i}]")
            ax.set_ylabel(f"v[{i}]")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Phase Plot: q[{i}] vs v[{i}]")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()
