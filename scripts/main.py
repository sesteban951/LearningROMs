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

    # split the RNG
    key, key1, key2 = random.split(key, 3)

    # generate random initial conditions
    x1 = random.uniform(key1, minval=-2.0, maxval=2.0, shape=(batch_size,))
    x2 = random.uniform(key2, minval=-2.0, maxval=2.0, shape=(batch_size,))
    x0_batch = jnp.stack([x1, x2], axis=1)  # shape (batch_size, nx)

    # solve ODE for all initial conditions in parallel
    x_traj_batch = ode_solver.forward_propagate_cl_batch(x0_batch, dt, N)
    # x_traj_batch = ode_solver.forward_propagate_nc_batch(x0_batch, dt, N)

    # parse for training
    x_t =  x_traj_batch[:, :-1, :]  # shape (batch_size, N, nx)
    x_t1 = x_traj_batch[:, 1:, :]   # shape (batch_size, N, nx)

    return x_t, x_t1, key


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

    #-----------------------------------------------------------
    # Models
    #-----------------------------------------------------------

    # create an instance of Full Order Model (FOM)
    # fom = DoubleIntegrator()
    # fom = Pendulum()
    fom = VanDerPol()
    # fom = LorenzAttractor()

    # create the ODE solver with the desired dynamics to integrate
    ode_solver = ODESolver(fom)

    #-----------------------------------------------------------
    # Hyper parameters
    #-----------------------------------------------------------

    # trajectory parameters
    dt = 0.01     # time step
    N = 300       # number of time steps to integrate

    # training parameters
    num_steps = 1_000    # number of training steps
    traj_batch_size = 64  # number of trajectories per batch
    mini_batch_size = 16  # number of trajectories per mini-batch
    print_every = 50      # print every n steps

    # random key
    # seed = 0
    seed = int(time.time())  # use current time as seed
    rng = random.PRNGKey(seed)
    rng_train, rng_data = random.split(rng, 2)

    # auto-encoder parameters
    z_dim = 2          # latent space dimension
    f_hidden_dim = 64  # hidden layer size for dynamics model
    E_hidden_dim = 64  # hidden layer size for Encoder
    D_hidden_dim = 64  # hidden layer size for Decoder

    # loss function weights
    learning_rate = 1e-3  # learning rate
    lambda_rec = 0.8      # reconstruction loss weight
    lambda_dyn = 0.5      # latent dynamics loss weight
    lambda_reg = 1e-4     # L2 regularization weight

    #-----------------------------------------------------------
    # Autoencoder + Trainer
    #-----------------------------------------------------------

    # create the neural network model
    ae = AutoEncoder(z_dim=z_dim, 
                     f_hidden_dim=f_hidden_dim, 
                     E_hidden_dim=E_hidden_dim, 
                     D_hidden_dim=D_hidden_dim)

    # create the trainer
    trainer = Trainer(ae,
                      rng_train,
                      fom.nx,
                      learning_rate=learning_rate, 
                      lambda_rec=lambda_rec, 
                      lambda_dyn=lambda_dyn, 
                      lambda_reg=lambda_reg)

    #-----------------------------------------------------------
    # Training loop
    #-----------------------------------------------------------

    # main training loop
    for step in range(num_steps):

        # Generate a fresh batch of trajectories (B_traj, N, nx)
        x_t, x_t1, rng_data = generate_batch_data(rng_data, ode_solver, traj_batch_size, dt, N)

        # Flatten to (num_pairs, nx)
        # Each trajectory contributes N pairs
        nx = x_t.shape[-1]
        X = x_t.reshape(-1, nx)      # (traj_batch_size * N, nx)
        Y = x_t1.reshape(-1, nx)     # (traj_batch_size * N, nx)

        # Sample a mini-batch of pairs
        num_pairs = X.shape[0]

        # mini_batch_size here means "number of PAIRS per step"
        mb = min(mini_batch_size, num_pairs)
        
        # After (on-device):
        rng_data, k_idx = random.split(rng_data)
        perm = random.permutation(k_idx, num_pairs)
        idx  = perm[:mb]
        xb, yb = X[idx], Y[idx]  # still (mb, nx) but stays on device

        # One training step
        trainer.state, metrics = trainer.train_step(trainer.state, xb, yb, step)

        # Log
        if step % print_every == 0:
            print(
                f"step {step:04d} | "
                f"loss={float(metrics.loss):.6f}  "
                f"rec={float(metrics.loss_rec):.6f}  "
                f"dyn={float(metrics.loss_dyn):.6f}  "
                f"reg={float(metrics.loss_reg):.6f}"
            )

    #-----------------------------------------------------------
    # Plotting
    #-----------------------------------------------------------

    # Pick one trajectory from the batch
    x_true = np.array(x_t[0])  # shape (N, nx)

    # Reconstruct
    x_hat, _, _ = ae.apply(trainer.state.params, x_true)

    x_hat = np.array(x_hat)  # convert to numpy for plotting

    plt.figure(figsize=(10,4))
    for i in range(x_true.shape[1]):
        plt.plot(x_true[:, i], label=f"true dim {i}")
        plt.plot(x_hat[:, i], '--', label=f"recon dim {i}")
    plt.xlabel("time step")
    plt.ylabel("state value")
    plt.legend()
    plt.title("True vs reconstructed trajectory")
    plt.show()

    #-----------------------------------------------------------

    # Encode a trajectory
    _, z_t, _ = ae.apply(trainer.state.params, x_true)

    z_t = np.array(z_t)

    plt.figure(figsize=(6,6))
    plt.plot(z_t[:, 0], z_t[:, 1], marker='o', alpha=0.7)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("Latent trajectory")
    plt.grid(True)
    plt.show()

    #-----------------------------------------------------------

    # Encode true z_t and z_t+1
    _, z_t, z_t1_hat = ae.apply(trainer.state.params, x_true[:-1])  # shape (N-1, z_dim)
    _, z_t1_true, _  = ae.apply(trainer.state.params, x_true[1:])

    z_t1_hat = np.array(z_t1_hat)
    z_t1_true = np.array(z_t1_true)

    plt.figure(figsize=(6,6))
    plt.scatter(z_t1_true[:,0], z_t1_true[:,1], label="true z_{t+1}", alpha=0.7)
    plt.scatter(z_t1_hat[:,0], z_t1_hat[:,1], label="pred z_{t+1}", alpha=0.7)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.legend()
    plt.title("Latent dynamics prediction")
    plt.grid(True)
    plt.show()


