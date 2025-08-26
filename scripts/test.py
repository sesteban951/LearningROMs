# standard imports 
import numpy as np               # standard numpy
import matplotlib.pyplot as plt  # standard matplotlib
import time                      # standard time

# jax imports
import jax
import jax.numpy as jnp         # standard jax numpy
import jax.scipy as jsp         # standard jax scipy
import jax.random as random     # jax random number generation
from jax import jit, vmap       # jit and vmap for speed and vectorization
from jax.lib import xla_bridge  # for device information
from functools import partial   # for partial functions

# custom imports
from rom import DoubleIntegrator
from ode_solver import ODESolver

#############################################################################

# helper function to create subkeys
def create_subkeys(key, num):
    return random.split(key, num) # (num, 2), each row is a subkey

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
    rom = DoubleIntegrator()

    # create the ODE solver with the desired dynamics to integrate
    solver = ODESolver(rom)

    # --- integration parameters ---
    x0 = jnp.array([1.0, 0.0])  # initial state [position, velocity]
    dt = 0.01                     # time step
    N = 500                        # number of steps

    # --- forward propagate using lax.scan RK4 ---
    t_traj = solver.create_time_array(dt, N)  # (N+1,)
    x_traj = solver.forward_propagate_cl(x0, dt, N)

    print(t_traj.shape, x_traj.shape)  # (N+1,), (N+1, nx)

    # --- convert to numpy for plotting ---
    x_traj = np.array(x_traj)

    print(x_traj.shape)

    # --- plot results in separate subplots ---
    fig, axs = plt.subplots(1, 2, figsize=(8, 10))

    # position vs time
    axs[0].plot(t_traj, x_traj[:,0], color='tab:blue')
    axs[0].plot(t_traj, x_traj[:,1], color='tab:orange')
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("States")
    axs[0].grid(True)

    # phase portrait (position vs velocity)
    axs[1].plot(x_traj[:,0], x_traj[:,1], color='tab:green')
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")
    axs[1].set_title("Phase Portrait")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()




