##
#
#  Parallel ODE Solver
#
##

# jax imports
import jax.numpy as jnp         # standard jax numpy
import jax.lax as lax           # for lax.scan
from jax import jit, vmap       # jit and vmap for speed and vectorization
from functools import partial   # for partial functions

# general ODE solver class
class ODESolver:

    def __init__(self, dynamics):

        # set the dynamics object
        self.nx = dynamics.nx
        self.nu = dynamics.nu

        # dynamics to propagate
        self.dynamics = dynamics

        # create JAX compiled functions
        f_jit = jit(dynamics.f)    # compile dynamics.f with JAX JIT for faster execution
        k_jit = jit(dynamics.k)    # compile dynamics.k with JAX JIT for faster execution
        f_batched = vmap(f_jit, in_axes=(None, 0, 0))   # vectorize f_jit over a batch of states and inputs
        k_batched = vmap(k_jit, in_axes=(None, 0))      # vectorize k_jit over a batch of states

        # store the compiled functions
        self.f_jit = f_jit        # single instance dynamics
        self.k_jit = k_jit        # single instance controller
        self.f_batch = f_batched  # batched dynamics
        self.k_batch = k_batched  # batched controller

        # print message about the initialization
        print(f"ODESolver initialized with dynamics: {dynamics.__class__.__name__}")

    ###################### GENERATE TIME TRAJECTORY ######################

    # create time array
    @staticmethod
    def create_time_array(dt, N):
        # build the time trajectory
        t_traj = jnp.arange(N + 1) * dt

        return t_traj

    ###################### SINGLE TRAJECTORY INTEGRATION ######################

    # RK4 Integration
    @partial(jit, static_argnums=(0,3))
    def forward_propagate_cl(self, x0, dt, N):
        """
        Forward propagate the closed loop (cl) dynamics using RK4 integration.
        
        Args:
            x0: initial state, shape (nx,)
            dt: time step
            N: number of steps
        Returns:
            x_traj: state trajectory, shape (N+1, nx)
        """

        # Use lax.scan over N steps
        t_inputs = jnp.arange(N) * dt  # (N, )

        # RK4 step function
        def rk4_step(x_k, t_k):
            
            t1 = t_k
            x1 = x_k
            u1 = self.k_jit(t1, x1)
            f1 = self.f_jit(t1, x1, u1)

            t2 = t_k + 0.5 * dt
            x2 = x_k + 0.5 * dt * f1
            u2 = self.k_jit(t2, x2)
            f2 = self.f_jit(t2, x2, u2)

            t3 = t_k + 0.5 * dt
            x3 = x_k + 0.5 * dt * f2
            u3 = self.k_jit(t3, x3)
            f3 = self.f_jit(t3, x3, u3)

            t4 = t_k + dt
            x4 = x_k + dt * f3
            u4 = self.k_jit(t4, x4)
            f4 = self.f_jit(t4, x4, u4)

            x_next = x_k + dt/6.0 * (f1 + 2*f2 + 2*f3 + f4)

            return x_next, x_next  # carry, y

        # scan over the time inputs
        x_last, x_hist = lax.scan(rk4_step, x0, t_inputs)

        # prepend initial state
        x_traj = jnp.vstack([x0[None, :], x_hist])

        return x_traj
    
    # RK4 Integration
    @partial(jit, static_argnums=(0,3))
    def forward_propagate_nc(self, x0, dt, N):
        """
        Forward propagate the no control (nc) dynamics using RK4 integration.
        
        Args:
            x0: initial state, shape (nx,)
            dt: time step
            N: number of steps
        Returns:
            x_traj: state trajectory, shape (N+1, nx)
        """

        # time steps for scan
        t_inputs = jnp.arange(N) * dt  # (N,)

        # zero control input
        u_zero = jnp.zeros((self.nu,))

        # RK4 step function
        def rk4_step(x_k, t_k):
        
            t1 = t_k
            x1 = x_k
            f1 = self.f_jit(t1, x1, u_zero)

            t2 = t_k + 0.5 * dt
            x2 = x_k + 0.5 * dt * f1
            f2 = self.f_jit(t2, x2, u_zero)

            t3 = t_k + 0.5 * dt
            x3 = x_k + 0.5 * dt * f2
            f3 = self.f_jit(t3, x3, u_zero)

            t4 = t_k + dt
            x4 = x_k + dt * f3
            f4 = self.f_jit(t4, x4, u_zero)

            x_next = x_k + dt/6.0 * (f1 + 2*f2 + 2*f3 + f4)

            return x_next, x_next

        # scan over the time inputs
        x_last, x_hist = lax.scan(rk4_step, x0, t_inputs)

        # prepend initial state
        x_traj = jnp.vstack([x0[None, :], x_hist])

        return x_traj


    ####################### BATCHED TRAJ INTEGRATION #######################

    # RK4 with fully batched lax.scan
    @partial(jit, static_argnums=(0,3))
    def forward_propagate_cl_batch(self, x0_batch, dt, N):
        """
        Forward propagate a batch of closed loop trajectories using RK4 + lax.scan.

        Args:
            x0_batch: initial states, shape (batch_size, nx)
            dt: time step
            N: number of steps
        Returns:
            x_traj_batch: state trajectories, shape (batch_size, N+1, nx)
        """

        # time steps for scan
        t_inputs = jnp.arange(N) * dt  # (N,)

        # RK4 step function for the whole batch
        def rk4_step(x_k_batch, t_k):
            # x_k_batch: (batch_size, nx)

            t1 = t_k
            x1 = x_k_batch
            u1 = self.k_batch(t1, x1)      # (batch_size, nu)
            f1 = self.f_batch(t1, x1, u1)  # (batch_size, nx)

            t2 = t_k + 0.5 * dt
            x2 = x_k_batch + 0.5 * dt * f1
            u2 = self.k_batch(t2, x2)
            f2 = self.f_batch(t2, x2, u2)

            t3 = t_k + 0.5 * dt
            x3 = x_k_batch + 0.5 * dt * f2
            u3 = self.k_batch(t3, x3)
            f3 = self.f_batch(t3, x3, u3)

            t4 = t_k + dt
            x4 = x_k_batch + dt * f3
            u4 = self.k_batch(t4, x4)
            f4 = self.f_batch(t4, x4, u4)

            x_next = x_k_batch + dt / 6.0 * (f1 + 2*f2 + 2*f3 + f4)

            return x_next, x_next  # carry, y

        # scan over time steps
        x_last, x_hist = lax.scan(rk4_step, x0_batch, t_inputs)

        # prepend initial state, x_traj shape: (N+1, batch_size, nx)
        x_traj = jnp.concatenate([x0_batch[None, :, :], x_hist], axis=0)

        # swap axes to get (batch_size, N+1, nx)
        x_traj_batch = jnp.swapaxes(x_traj, 0, 1)

        return x_traj_batch

    # RK4 with fully batched lax.scan
    @partial(jit, static_argnums=(0,3))
    def forward_propagate_nc_batch(self, x0_batch, dt, N):
        """
        Forward propagate a batch of no control trajectories using RK4 + lax.scan.

        Args:
            x0_batch: initial states, shape (batch_size, nx)
            dt: time step
            N: number of steps
        Returns:
            x_traj_batch: state trajectories, shape (batch_size, N+1, nx)
        """

        # time steps for scan
        t_inputs = jnp.arange(N) * dt  # (N,)

        # zero control input
        batch_size = x0_batch.shape[0]
        u_zero = jnp.zeros((batch_size, self.nu))  # works even if nu=0

        # RK4 step function for the whole batch
        def rk4_step(x_k_batch, t_k):
            # x_k_batch: (batch_size, nx)

            t1 = t_k
            x1 = x_k_batch
            f1 = self.f_batch(t1, x1, u_zero)

            t2 = t_k + 0.5 * dt
            x2 = x_k_batch + 0.5 * dt * f1
            f2 = self.f_batch(t2, x2, u_zero)

            t3 = t_k + 0.5 * dt
            x3 = x_k_batch + 0.5 * dt * f2
            f3 = self.f_batch(t3, x3, u_zero)

            t4 = t_k + dt
            x4 = x_k_batch + dt * f3
            f4 = self.f_batch(t4, x4, u_zero)

            x_next = x_k_batch + dt / 6.0 * (f1 + 2*f2 + 2*f3 + f4)
            
            return x_next, x_next  # carry, y
        
        # scan over time steps
        x_last, x_hist = lax.scan(rk4_step, x0_batch, t_inputs)

        # prepend initial state, x_traj shape: (N+1, batch_size, nx)
        x_traj = jnp.concatenate([x0_batch[None, :, :], x_hist], axis=0)

        # swap axes to get (batch_size, N+1, nx)
        x_traj_batch = jnp.swapaxes(x_traj, 0, 1)

        return x_traj_batch
    

#############################################################################
# TESTING
#############################################################################

# standard imports 
import numpy as np               # standard numpy
import matplotlib.pyplot as plt  # standard matplotlib
import time                      # standard time

# jax imports
import jax
import jax.random as random     # jax random number generation

# custom imports
from fom import *


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
    seed = int(time.time())
    key = random.PRNGKey(seed)
    
    # create an instance of Full Order Model (FOM)
    # fom = DoubleIntegrator()
    # fom = Pendulum()
    # fom = VanDerPol()
    fom = LorenzAttractor()

    # create the ODE solver with the desired dynamics to integrate
    solver = ODESolver(fom)

    ########################### 2D SINGLE TRAJ INTEGRATION ###########################

    # # --- integration parameters ---
    # x0 = jnp.array([1.0, 0.0])  # initial state [position, velocity]
    # dt = 0.01                     # time step
    # N = 5000                        # number of steps

    # # --- forward propagate using lax.scan RK4 ---
    # t_traj = solver.create_time_array(dt, N)  # (N+1,)
    # # x_traj = solver.forward_propagate_cl(x0, dt, N) # (N+1, nx)
    # x_traj = solver.forward_propagate_nc(x0, dt, N) # (N+1, nx)

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
    dt = 0.01            # time step
    N = 500              # number of steps
    batch_size = 1_000   # number of different initial conditions
    num_trajs_plot = 50  # number of trajectories to plot

    # --- generate random initial conditions ---
    key, subkey = random.split(key, 2)  # create subkey
    x_min = fom.x_min
    x_max = fom.x_max
    x0_batch = jax.random.uniform(subkey, shape=(batch_size, fom.nx), minval=x_min, maxval=x_max)

    print("x0_batch type:", type(x0_batch))
    print("x0_batch dtype:", x0_batch.dtype)
    print("x0_batch.shape:", x0_batch.shape)

    # --- propagate all trajectories in parallel ---
    t_traj = solver.create_time_array(dt, N)                           # shape (N+1,)
    t0 = time.time()
    x_traj_batch = solver.forward_propagate_cl_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    # x_traj_batch = solver.forward_propagate_nc_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    x_traj_batch.block_until_ready()  # ensure computation is done
    t1 = time.time()
    x_traj_batch = solver.forward_propagate_cl_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    # x_traj_batch = solver.forward_propagate_nc_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    x_traj_batch.block_until_ready()  # ensure computation is done
    t2 = time.time()
    x_traj_batch = solver.forward_propagate_cl_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    # x_traj_batch = solver.forward_propagate_nc_batch(x0_batch, dt, N)  # shape (batch_size, N+1, 2)
    x_traj_batch.block_until_ready()  # ensure computation is done
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

    # Plot in 3D
    if fom.nx == 3:

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(min(num_trajs_plot, batch_size)):
            ax.plot(
                x_traj_batch[i, :, 0],  # x1(t)
                x_traj_batch[i, :, 1],  # x2(t)
                x_traj_batch[i, :, 2],  # x3(t)
                alpha=0.6
            )

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")

    # Plost 2D
    elif fom.nx == 2:

        fig, ax = plt.subplots(figsize=(8,6))

        for i in range(min(num_trajs_plot, batch_size)):
            ax.plot(
                x_traj_batch[i, :, 0],  # x1(t)
                x_traj_batch[i, :, 1],  # x2(t)
                alpha=0.6
            )

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    ax.set_title(fom.__class__.__name__)
    plt.show()