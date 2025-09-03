##
#
#  Main script to demonstrate ROM Learning
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
