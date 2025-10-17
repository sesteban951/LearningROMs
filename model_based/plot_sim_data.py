# standard imports
import numpy as np
import matplotlib.pyplot as plt
import math

# labels for the biped
def biped_labels():
    q_labels = [
        "x_base", "z_base", "theta_base",
        "q_hip_left", "q_knee_left",
        "q_hip_right", "q_knee_right"
    ]
    v_labels = [
        "vx_base", "vz_base", "omega_base",
        "v_hip_left", "v_knee_left",
        "v_hip_right", "v_knee_right"
    ]
    u_labels = [
        "tau_hip_left", "tau_knee_left",
        "tau_hip_right", "tau_knee_right"
    ]
    c_labels = [
        "contact_left", "contact_right"
    ]
    return q_labels, v_labels, u_labels, c_labels

# labels for the hopper
def hopper_labels():
    q_labels = [
        "x_base", "z_base", "theta_base", "l_leg"
    ]
    v_labels = [
        "vx_base", "vz_base", "omega_base", "v_leg"
    ]
    u_labels = [
        "tau_theta", "tau_leg"
    ]
    c_labels = [
        "contact_foot"
    ]
    return q_labels, v_labels, u_labels, c_labels

def choose_labels(robot):
    if robot == "biped":
        return biped_labels()
    elif robot == "hopper":
        return hopper_labels()
    else:
        # generic fallbacks
        return (
            [f"q[{i}]" for i in range(q_log.shape[1])],
            [f"v[{i}]" for i in range(v_log.shape[1])],
            [f"u[{i}]" for i in range(u_log.shape[1])],
            [f"c[{i}]" for i in range(c_log.shape[1])],
        )

def label_or_idx(labels, i, prefix):
    # robustly fetch labels[i], or a sane default
    return labels[i] if (labels is not None and i < len(labels)) else f"{prefix}[{i}]"


################################################################################
# MAIN PLOTTING
################################################################################

if __name__ == "__main__":

    # data to import 
    robot = "hopper"
    # robot = "biped"

    # import the data file
    data = np.load(f"./model_based/{robot}/{robot}_data.npz")

    # unpck the data 
    t_log = data["t_log"]
    q_log = data["q_log"]
    v_log = data["v_log"]
    u_log = data["u_log"]
    c_log = data["c_log"]
    cmd_log = data["cmd_log"]

    # check sizes
    n_steps = t_log.shape[0]
    nq = q_log.shape[1]
    nv = v_log.shape[1]
    nu = u_log.shape[1]
    nc = c_log.shape[1]
    ncmd = cmd_log.shape[1]

    print(f"t_log shape: {t_log.shape}")
    print(f"q_log shape: {q_log.shape}")
    print(f"v_log shape: {v_log.shape}")
    print(f"u_log shape: {u_log.shape}")
    print(f"c_log shape: {c_log.shape}")
    print(f"cmd_log shape: {cmd_log.shape}")

    # only take a segement of the trajectory
    segment = (0.0, 1.0)  # (start, end) as percent of trajectory length
    start_idx = int(segment[0] * n_steps)
    end_idx = int(segment[1] * n_steps)
    t_log = t_log[start_idx:end_idx]
    q_log = q_log[start_idx:end_idx, :]
    v_log = v_log[start_idx:end_idx, :]
    u_log = u_log[start_idx:end_idx, :]
    c_log = c_log[start_idx:end_idx, :]
    cmd_log = cmd_log[start_idx:end_idx, :]

    # parse the contact log to boolean 
    epsilon = 1e-6
    c_log_bool = np.abs(c_log) > epsilon

    ########################################
    # plot the data
    ########################################
    
    # Robust time bases
    t_state = np.asarray(t_log).squeeze()                     # (T_state,)
    T_state = t_state.shape[0]

    T_q = q_log.shape[0]
    T_v = v_log.shape[0]
    T_u = u_log.shape[0]
    T_c = c_log.shape[0] if 'c_log' in locals() else 0

    # If any series length != T_state, build its own time vector that spans the same duration
    t_end = float(t_state[-1]) if T_state > 0 else 0.0
    def make_time(T, default_dt= (t_state[1]-t_state[0] if T_state>1 else 0.01)):
        if T <= 1:
            return np.array([0.0])
        # evenly span [0, t_end] with T samples
        return np.linspace(0.0, t_end if t_end>0 else default_dt*(T-1), T)

    t_q = t_state if T_q == T_state else make_time(T_q)
    t_v = t_state if T_v == T_state else make_time(T_v)
    t_u = t_state if T_u == T_state else make_time(T_u)
    t_c = t_state if (T_c == T_state and T_c>0) else (make_time(T_c) if T_c>0 else None)

    # Optional: list any angle indices to unwrap
    ANGLE_IDXS = []  # e.g., [1, 3]
    def maybe_unwrap(arr, idx):
        return np.unwrap(arr) if idx in ANGLE_IDXS else arr

    # Row count = max dimension among q, v, u, c (per DOF/component)
    nrows = max(nq, nv, nu, nc)

    # pick labels based on robot
    q_labels, v_labels, u_labels, c_labels = choose_labels(robot)

    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 2.6 * max(1, nrows)), sharex=False)
    if axes.ndim == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("Positions (q)")
    axes[0, 1].set_title("Velocities (v)")
    axes[0, 2].set_title("Controls (u)")
    axes[0, 3].set_title("Contacts (c)")

    alpha_traj = 0.8

    for i in range(nrows):
        ax_q, ax_v, ax_u, ax_c = axes[i, 0], axes[i, 1], axes[i, 2], axes[i, 3]

        # q[i]
        if i < nq:
            ax_q.plot(t_q, maybe_unwrap(q_log[:, i], i), alpha=alpha_traj)
            ax_q.set_ylabel(label_or_idx(q_labels, i, "q"))
            ax_q.grid(True, alpha=0.3)
        else:
            ax_q.axis("off")

        # v[i]
        if i < nv:
            ax_v.plot(t_v, v_log[:, i], alpha=alpha_traj)
            if i == 0:  # vx row
                ax_v.plot(t_v, cmd_log, linestyle="--", linewidth=1.5, label="vx_cmd")
                ax_v.legend(frameon=False, loc="best")
            ax_v.set_ylabel(label_or_idx(v_labels, i, "v"))
            ax_v.grid(True, alpha=0.3)
        else:
            ax_v.axis("off")

        # u[i]
        if i < nu:
            ax_u.plot(t_u, u_log[:, i], alpha=alpha_traj)
            ax_u.set_ylabel(label_or_idx(u_labels, i, "u"))
            ax_u.grid(True, alpha=0.3)
        else:
            ax_u.axis("off")

        # c[i]  (use step style if you like; boolean-friendly)
        if i < nc and t_c is not None:
            ax_c.plot(t_c, c_log[:, i], alpha=alpha_traj, drawstyle="steps-post")
            ax_c.set_ylabel(label_or_idx(c_labels, i, "c"))
            ax_c.grid(True, alpha=0.3)
        else:
            ax_c.axis("off")

    for ax in axes.ravel():
        ax.set_xlabel("Time [s]")
        ax.tick_params(labelbottom=True)

    plt.tight_layout()
    plt.show()
    
    ########################################
    # plot the phase data
    ########################################

    # Build (n_traj, T, dim) arrays from logs (unchanged)
    q_traj = np.expand_dims(q_log.T, axis=0).transpose(0, 2, 1)  # (1, T, nq)
    v_traj = np.expand_dims(v_log.T, axis=0).transpose(0, 2, 1)  # (1, T, nv)

    idxs = [0]
    alpha_traj = 0.5
    show_legends = (len(idxs) <= 6)

    nq_phase = q_traj.shape[2]
    nv_phase = v_traj.shape[2]
    nplots = nq_phase

    ncols = int(math.ceil(math.sqrt(nplots)))
    nrows_phase = int(math.ceil(nplots / ncols))  # avoid clobbering earlier nrows
    fig, axes = plt.subplots(nrows=nrows_phase, ncols=ncols,
                            figsize=(4.8 * ncols, 3.6 * nrows_phase),
                            squeeze=False)

    plot_idx = 0
    for i in range(nq_phase):
        r = plot_idx // ncols
        c = plot_idx % ncols
        ax = axes[r, c]

        if i < nv_phase:
            for idx in idxs:
                ax.plot(q_traj[idx, :, i], v_traj[idx, :, i], alpha=alpha_traj)
            ax.set_xlabel(label_or_idx(q_labels, i, "q"))
            ax.set_ylabel(label_or_idx(v_labels, i, "v"))
            ax.grid(True, alpha=0.3)
            ax.set_aspect('auto')  # 'equal' if you prefer square boxes
            if show_legends and plot_idx == 0 and len(idxs) > 1:
                ax.legend(frameon=False, loc="best")
        else:
            ax.axis("off")
        plot_idx += 1

    # Hide leftovers
    total_axes = nrows_phase * ncols
    for k in range(plot_idx, total_axes):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()
