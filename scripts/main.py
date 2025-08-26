##
#
#  Main script to demonstrate the ODE solver with different full order models (FOMs).
#
##


# standard imports 
import numpy as np               # standard numpy
import matplotlib.pyplot as plt  # standard matplotlib
import time                      # standard time

# jax imports
import jax
import jax.numpy as jnp         # standard jax numpy
import jax.random as random     # jax random number generation

# custom imports
from fom import DoubleIntegrator, Pendulum, VanDerPol
from ode_solver import ODESolver          

#############################################################################

# helper function to create subkeys
def create_subkeys(key, num):

    # split the key into num subkeys
    subkeys = random.split(key, num)  # (num, 2), each row is a subkey

    return subkeys 

#############################################################################

# Example usage
if __name__ == "__main__":

    # Choose device
    device_desired = "gpu"  # "cpu" or "gpu"
    if device_desired == "gpu" and jax.devices("gpu"):
        jax.config.update('jax_platform_name', 'gpu')
    else:
        jax.config.update('jax_platform_name', 'cpu')
    backend = jax.default_backend()
    print("JAX is using device:", backend)

    # PRNG key
    seed = 0
    key = random.PRNGKey(seed)
    
    # create an instance of the DoubleIntegrator class
    # rom = DoubleIntegrator()
    # rom = Pendulum()
    rom = VanDerPol()

    # create the ODE solver with the desired dynamics to integrate
    solver = ODESolver(rom)

    ########################### SINGLE TRAJ INTEGRATION ###########################

    # # --- integration parameters ---
    # x0 = jnp.array([1.0, 0.0])  # initial state [position, velocity]
    # dt = 0.01                     # time step
    # N = 500                        # number of steps

    # # --- forward propagate using lax.scan RK4 ---
    # t_traj = solver.create_time_array(dt, N)  # (N+1,)
    # x_traj = solver.forward_propagate_cl(x0, dt, N)

    # print(t_traj.shape, x_traj.shape)  # (N+1,), (N+1, nx)

    # # --- convert to numpy for plotting ---
    # x_traj = np.array(x_traj)

    # print(x_traj.shape)

    # # --- plot results in separate subplots ---
    # fig, axs = plt.subplots(1, 2, figsize=(8, 10))

    # # position vs time
    # axs[0].plot(t_traj, x_traj[:,0], color='tab:blue')
    # axs[0].plot(t_traj, x_traj[:,1], color='tab:orange')
    # axs[0].set_xlabel("Time [s]")
    # axs[0].set_ylabel("States")
    # axs[0].grid(True)

    # # phase portrait (position vs velocity)
    # axs[1].plot(x_traj[:,0], x_traj[:,1], color='tab:green')
    # axs[1].set_xlabel("Position")
    # axs[1].set_ylabel("Velocity")
    # axs[1].set_title("Phase Portrait")
    # axs[1].grid(True)

    # plt.tight_layout()
    # plt.show()

    ########################### BATCH TRAJ INTEGRATION ###########################

    # --- integration parameters ---
    dt = 0.01
    N = 500
    batch_size = 100_000  # number of different initial conditions

    # --- generate random initial conditions ---
    subkeys = create_subkeys(key, 2)  # (batch_size, 2)
    x1 = jax.random.uniform(subkeys[0], shape=(batch_size,), minval=-2.0, maxval=2.0)
    x2 = jax.random.uniform(subkeys[1], shape=(batch_size,), minval=-2.0, maxval=2.0)
    x0_batch = jnp.stack([x1, x2], axis=1)  # shape (batch_size, 2)

    print("x0_batch type:", type(x0_batch))
    print("x0_batch dtype:", x0_batch.dtype)
    print("x0_batch.shape:", x0_batch.shape)

    # --- propagate all trajectories in parallel ---
    t_traj = solver.create_time_array(dt, N)                           # shape (N+1,)
    t0 = time.time()
    x_traj_batch = solver.forward_propagate_cl_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    t1 = time.time()
    x_traj_batch = solver.forward_propagate_cl_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    t2 = time.time()
    x_traj_batch = solver.forward_propagate_cl_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    t3 = time.time()

    print(f"Time for first call (includes compilation): {t1 - t0:.4f} seconds")
    print(f"Time for second call: {t2 - t1:.4f} seconds")
    print(f"Time for third call: {t3 - t2:.4f} seconds")

    print("x_traj_batch type:", type(x_traj_batch))
    print("x_traj_batch dtype:", x_traj_batch.dtype)  
    print("x_traj_batch.shape:", x_traj_batch.shape)  

    # --- convert to numpy for plotting ---
    x_traj_batch = np.array(x_traj_batch)
    print("After conversion to numpy:")
    print("x_traj_batch type:", type(x_traj_batch))
    print("x_traj_batch dtype:", x_traj_batch.dtype)  
    print("x_traj_batch.shape:", x_traj_batch.shape)

    # --- optional: plot a few trajectories ---
    plt.figure(figsize=(8,4))

    # plot up to num_trajs_plot trajectories
    num_trajs_plot = 100
    for i in range(min(num_trajs_plot, batch_size)):  
        plt.plot(x_traj_batch[i,:,1], x_traj_batch[i,:,0], alpha=0.6)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(rom.__class__.__name__)
    plt.grid(True)
    plt.show()
