# standard imports
import numpy as np
import matplotlib.pyplot as plt
import math


################################################################################
# MAIN PLOTTING
################################################################################

if __name__ == "__main__":

    # data to import 
    # robot = "hopper"
    robot = "biped"

    # import the data file
    data = np.load(f"./model_based/{robot}/{robot}_data.npz")

    # unpck the data 
    t_log = data["t_log"]
    q_log = data["q_log"]
    v_log = data["v_log"]
    u_log = data["u_log"]

    # check sizes
    n_steps = t_log.shape[0]
    nq = q_log.shape[1]
    nv = v_log.shape[1]
    nu = u_log.shape[1]

    print(f"t_log shape: {t_log.shape}")
    print(f"q_log shape: {q_log.shape}")
    print(f"v_log shape: {v_log.shape}")
    print(f"u_log shape: {u_log.shape}")

    # only take a segement of the trajectory
    segment = (0.0, 1.0)  # (start, end) as percent of trajectory length
    start_idx = int(segment[0] * n_steps)
    end_idx = int(segment[1] * n_steps)
    t_log = t_log[start_idx:end_idx]
    q_log = q_log[start_idx:end_idx, :]
    v_log = v_log[start_idx:end_idx, :]
    u_log = u_log[start_idx:end_idx, :]

    ########################################
    # plot the data
    ########################################

    # Build (n_traj, T, dim) arrays from logs
    q_traj = np.expand_dims(q_log.T, axis=0).transpose(0, 2, 1)  # (1, T, nq)
    v_traj = np.expand_dims(v_log.T, axis=0).transpose(0, 2, 1)  # (1, T, nv)

    # ---- settings ----
    idxs = [0]           # single trajectory
    alpha_traj = 0.6
    show_legends = (len(idxs) <= 6)

    # If some q's are angles, list their indices here to unwrap:
    ANGLE_IDXS = []  # e.g., [0, 2] if q[0], q[2] are angles

    # ---- utilities ----
    def maybe_unwrap(arr, idx):
        """Return arr if idx not in ANGLE_IDXS; otherwise unwrap along time."""
        if idx in ANGLE_IDXS:
            return np.unwrap(arr)
        return arr

    # Use only valid q-v pairs
    nq_phase = q_traj.shape[2]
    nv_phase = v_traj.shape[2]
    paired_dim = min(nq_phase, nv_phase)   # only plot pairs that exist

    # Choose near-square grid
    nplots = paired_dim
    ncols = int(math.ceil(math.sqrt(nplots)))
    nrows = int(math.ceil(nplots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.8 * ncols, 3.6 * nrows),
                             squeeze=False)

    plot_idx = 0
    for i in range(paired_dim):
        r = plot_idx // ncols
        c = plot_idx % ncols
        ax = axes[r, c]

        for idx in idxs:
            q_i = maybe_unwrap(q_traj[idx, :, i], i)
            v_i = v_traj[idx, :, i]
            ax.plot(q_i, v_i, alpha=alpha_traj)
        ax.set_xlabel(f"q[{i}]")
        ax.set_ylabel(f"v[{i}]")
        ax.grid(True, alpha=0.3)
        # For cleaner geometry, you can use equal aspect (commented by default)
        # ax.set_aspect('equal', adjustable='datalim')
        if show_legends and plot_idx == 0 and len(idxs) > 1:
            ax.legend(frameon=False, loc="best")
        plot_idx += 1

    # Hide any leftover empty axes
    total_axes = nrows * ncols
    for k in range(plot_idx, total_axes):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()

    ########################################
    # plot the phase data
    ########################################

    # Build (n_traj, T, dim) arrays from logs
    q_traj = np.expand_dims(q_log.T, axis=0).transpose(0, 2, 1)  # (1, T, nq)
    v_traj = np.expand_dims(v_log.T, axis=0).transpose(0, 2, 1)  # (1, T, nv)

    idxs = [0]           # single trajectory index
    alpha_traj = 0.5
    show_legends = (len(idxs) <= 6)

    nq_phase = q_traj.shape[2]
    nv_phase = v_traj.shape[2]
    nplots = nq_phase  # one phase plot per q[i] (if no v[i], we skip/hide)

    # Choose near-square grid
    ncols = int(math.ceil(math.sqrt(nplots)))
    nrows = int(math.ceil(nplots / ncols))

    # Size scales with grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.8 * ncols, 3.6 * nrows),
                             squeeze=False)

    plot_idx = 0
    for i in range(nq_phase):
        r = plot_idx // ncols
        c = plot_idx % ncols
        ax = axes[r, c]

        if i < nv_phase:
            for idx in idxs:
                ax.plot(q_traj[idx, :, i], v_traj[idx, :, i],
                        alpha=alpha_traj)
            ax.set_xlabel(f"q[{i}]")
            ax.set_ylabel(f"v[{i}]")
            ax.grid(True, alpha=0.3)
            # Optional: make the phase box roughly square
            ax.set_aspect('auto')  # change to 'equal' if you want perfect squares
            if show_legends and plot_idx == 0:
                ax.legend(frameon=False, loc="best")
        else:
            ax.axis("off")
        plot_idx += 1

    # Hide any leftover empty axes
    total_axes = nrows * ncols
    for k in range(plot_idx, total_axes):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()
