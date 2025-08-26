##
#
# Assortment of reduced order models (ROMs).
#
##

import jax
import jax.numpy as jnp         # standard jax numpy
import jax.scipy as jsp         # standard jax scipy
import jax.random as random     # jax random number generation
from jax import jit, vmap       # jit and vmap for speed and vectorization
from functools import partial   # for partial functions
from jax.lib import xla_bridge  # for device information

# general ODE solver class
class DoubleIntegrator:
    """
        Double Integrator Dynamics Class
    """
    # initialization
    def __init__(self):

        # state and input dimensions
        self.nx = 2
        self.nu = 1

        # system parameters
        self.A = jnp.array([[0, 1],
                            [0, 0]]) # (2, 2)
        self.B = jnp.array([[0],
                            [1]])    # (2, 1)
        
        # gains to use 
        self.kp = 10.0
        self.kd = 1.0
        self.K = jnp.array([[self.kp, self.kd]]) # (1, 2)

    # dynamics function, continuous time
    def f(self, x, u):
        """
            ẋ = Ax + Bu
        """
        # compute the dynamics
        xdot = jnp.dot(self.A, x) + jnp.dot(self.B, u)

        return xdot
    
    # feedback controller
    def k(self, t, x):
        """
            Simple PD controller
            u = -Kx
        """
        # compute the control input
        u = -jnp.dot(self.K, x)

        return u

################################################

# Example usage
if __name__ == "__main__":

    # set the device to GPU if available, otherwise CPU
    if jax.devices("gpu"):
        jax.config.update('jax_platform_name', 'gpu')
    else:
        jax.config.update('jax_platform_name', 'cpu')
    print("JAX is using device:", xla_bridge.get_backend().platform)

    # PRNG key
    seed = 0
    key = random.PRNGKey(seed)
    key, subkey_x = random.split(key)
    key, subkey_u = random.split(key)
    
    # create an instance of the DoubleIntegrator class
    di = DoubleIntegrator()

    # jit the functions for speed
    f_jit = jit(di.f)
    k_jit = jit(di.k)
    f_batched = vmap(f_jit, in_axes=(0, 0)) # vmap over a bunch of states and inputs
    k_batched = vmap(k_jit, in_axes=(None, 0))

    # vmap over a bunch of states
    batches = 50
    
    # generate sample inside of [1, -1] x [1, -1]
    X = random.uniform(subkey_x, (batches, di.nx), minval=-1.0, maxval=1.0) # (batches, 2)
    U = random.uniform(subkey_u, (batches, di.nu), minval=-1.0, maxval=1.0) # (batches, 1)

    print("X shape:", X.shape)
    print("U shape:", U.shape)

    # feed though control 
    U_ctrl = k_batched(0.0, X) # (batches, 1)
    print("U_ctrl shape:", U_ctrl.shape)

    # feed through dynamics
    Xdot = f_batched(X, U_ctrl) # (batches, 2)
    print("Xdot shape:", Xdot.shape)
