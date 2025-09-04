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
from fom import DoubleIntegrator, Pendulum, VanDerPol, LorenzAttractor
from ode_solver import ODESolver
from auto_encoder import AutoEncoder, Trainer     

#############################################################################
# Helper functions
#############################################################################

def generate_batch_data(key, ode_solver, batch_size, dt, N):

    # generate random initial conditions
    subkeys = random.split(key, 3)
    x1 = random.uniform(subkeys[1], minval=-2.0, maxval=2.0, shape=(batch_size,))
    x2 = random.uniform(subkeys[2], minval=-2.0, maxval=2.0, shape=(batch_size,))
    x0_batch = jnp.stack([x1, x2], axis=1)  # shape (batch_size, nx)

    # solve ODE for all initial conditions in parallel
    x_traj_batch = ode_solver.forward_propagate_cl_batch(x0_batch, dt, N)

    # parse for training
    x_t =  x_traj_batch[:, :-1, :]  # shape (batch_size, N, nx)
    x_t1 = x_traj_batch[:, 1:, :]   # shape (batch_size, N, nx)

    # new key
    new_key = subkeys[0]

    return x_t, x_t1, new_key


#############################################################################
# MAIN SCRIPT
#############################################################################


if __name__ == "__main__":

    # Choose device
    device_desired = "gpu"  # "cpu" or "gpu"
    if device_desired == "gpu" and jax.devices("gpu"):
        jax.config.update('jax_platform_name', 'gpu')
    else:
        jax.config.update('jax_platform_name', 'cpu')
    backend = jax.default_backend()
    print("JAX is using device:", backend)

    # create an instance of Full Order Model (FOM)
    # fom = DoubleIntegrator()
    # fom = Pendulum()
    fom = VanDerPol()
    # fom = LorenzAttractor()

    # trajectory parameters
    dt = 0.01             # time step size
    N = 500               # number of time steps

    # create the ODE solver with the desired dynamics to integrate
    ode_solver = ODESolver(fom)

    # create the neural network model
    ae = AutoEncoder(z_dim=2, 
                     f_hidden_dim=64, 
                     E_hidden_dim=64, 
                     D_hidden_dim=64)

    # create the trainer
    seed = 0
    rng = random.PRNGKey(seed)
    trainer = Trainer(ae,
                      rng,
                      fom.nx,
                      learning_rate=1e-3, 
                      lambda_rec=0.8, 
                      lambda_dyn=0.2, 
                      lambda_reg=1e-4)

