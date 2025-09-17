# standard imports
import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt

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


#################################################################

if __name__ == "__main__":

    # load the data
    file_name = "./data/paddle_ball_data.npz"
    data = np.load(file_name)

    # access the arrays
    q_traj = data['q_traj']
    v_traj = data['v_traj']
    u_traj = data['u_traj']

    # get the shape of the data
    batch_size, N_state, nq = q_traj.shape
    _, _, nv = v_traj.shape
    _, N_input, nu = u_traj.shape

    print(f"batch_size: {batch_size}")
    print(f"N_state: {N_state}")
    print(f"N_input: {N_input}")
    print(f"nq: {nq}")
    print(f"nv: {nv}")
    print(f"nu: {nu}")

    # select one random trajectory to playback
    traj_idx = np.random.randint(batch_size)

    print(f"Playing back trajectory {traj_idx} of {batch_size}")

    # load the mujoco model
    env = envs.get_environment("paddle_ball")
    config = env.config

    model_path = config.model_path
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

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
                print("Reached the end of the trajectory")
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

    # figure and axes: nq rows, 3 cols (q, v, u)
    fig, axes = plt.subplots(nrows=nq, ncols=3, figsize=(14, 2.2 * nq), sharex=True)
    if nq == 1:
        axes = axes[None, :]  # ensure 2D indexing when nq == 1

    for i in range(nq):
        ax_q, ax_v, ax_u = axes[i, 0], axes[i, 1], axes[i, 2]

        # --- Column 0: q[i] ---
        for _ in range(n_plot):
            idx = np.random.randint(batch_size)
            ax_q.plot(t_state, q_traj[idx, :, i], alpha=0.5)
        ax_q.set_ylabel(f"q[{i}]")
        ax_q.grid(True, alpha=0.3)
        if i == 0:
            ax_q.set_title("Positions (q)")

        # --- Column 1: v[i] (hide if i >= nv) ---
        if i < nv:
            for _ in range(n_plot):
                idx = np.random.randint(batch_size)
                ax_v.plot(t_state, v_traj[idx, :, i], alpha=0.5)
            ax_v.set_ylabel(f"v[{i}]")
            ax_v.grid(True, alpha=0.3)
            if i == 0:
                ax_v.set_title("Velocities (v)")
        else:
            ax_v.axis("off")

        # --- Column 2: u[i] (hide if i >= nu) ---
        if i < nu:
            for _ in range(n_plot):
                idx = np.random.randint(batch_size)
                ax_u.plot(t_input, u_traj[idx, :, i], alpha=0.5)
            ax_u.set_ylabel(f"u[{i}]")
            ax_u.grid(True, alpha=0.3)
            if i == 0:
                ax_u.set_title("Controls (u)")
        else:
            ax_u.axis("off")

    # x-label only on bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()
