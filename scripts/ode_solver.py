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

        # time trajectory
        t_traj = jnp.arange(N + 1) * dt

        # Use lax.scan over N steps
        t_inputs = t_traj[:-1]  # (N, )

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

