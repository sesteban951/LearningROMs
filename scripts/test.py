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
    subkeys = create_subkeys(key, 2)
    
    # create an instance of the DoubleIntegrator class
    rom = DoubleIntegrator()

    # jit the functions for speed
    f_jit = jit(rom.f)
    k_jit = jit(rom.k)
    f_batched = vmap(f_jit, in_axes=(0, 0)) # vmap over a bunch of states and inputs
    k_batched = vmap(k_jit, in_axes=(None, 0))

    # vmap over a bunch of states
    batches = 50
    
    # generate sample inside of [1, -1] x [1, -1]
    X = random.uniform(subkeys[0], (batches, rom.nx), minval=-1.0, maxval=1.0) # (batches, 2)
    U = random.uniform(subkeys[1], (batches, rom.nu), minval=-1.0, maxval=1.0) # (batches, 1)

    print("X shape:", X.shape)
    print("U shape:", U.shape)

    # feed though control 
    U_ctrl = k_batched(0.0, X) # (batches, 1)
    print("U_ctrl shape:", U_ctrl.shape)

    # feed through dynamics
    Xdot = f_batched(X, U_ctrl) # (batches, 2)
    print("Xdot shape:", Xdot.shape)
