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
from fom import *           # full order model examples
from ode_solver import *    # ODE solver for traj generation
from auto_encoder import *  # autoencoder, training, configs

#############################################################################
# Helper functions
#############################################################################

# generate a batch of trajectory data
def generate_batch_data(key, ode_solver, batch_size, dt, N):

    # split the RNG
    key, key1, key2, key3 = random.split(key, 4)

    # generate random initial conditions
    # x1 = random.uniform(key1, minval=-2.0, maxval=2.0, shape=(batch_size,))
    # x2 = random.uniform(key2, minval=-2.0, maxval=2.0, shape=(batch_size,))
    # x0_batch = jnp.stack([x1, x2], axis=1)  # shape (batch_size, nx)

    x1 = random.uniform(key1, minval=-20.0, maxval=20.0, shape=(batch_size,))
    x2 = random.uniform(key2, minval=-20.0, maxval=20.0, shape=(batch_size,))
    x3 = random.uniform(key3, minval=0.0, maxval=50.0, shape=(batch_size,))
    x0_batch = jnp.stack([x1, x2, x3], axis=1)  # shape (batch_size, nx)

    # solve ODE for all initial conditions in parallel
    x_traj_batch = ode_solver.forward_propagate_cl_batch(x0_batch, dt, N)
    # x_traj_batch = ode_solver.forward_propagate_nc_batch(x0_batch, dt, N)

    # parse for training
    x_t =  x_traj_batch[:, :-1, :]  # shape (batch_size, N, nx)
    x_t1 = x_traj_batch[:, 1:, :]   # shape (batch_size, N, nx)

    return x_t, x_t1, key

# rollout a single trajectory using the ode solver
def rollout_true(ode_solver, x0, dt, N):
    
    # ode solver rollout
    x_traj = ode_solver.forward_propagate_cl(x0, dt, N)  # you already have this
    
    return x_traj

# rollout a single trajectory using the auto encoder
def rollout_ae(ae, params, x0, N):

    # z0 from x0
    z0 = ae.apply(params, x0[None,:], method=ae.encode)  # shape (z_dim,)
    z0 = z0[0]  # remove batch dim, shape (z_dim,)

    # step function in latent space
    def step_fn(z_t, _):

        z_t1 = ae.apply(params, z_t[None,:], method=ae.latent_dynamics)[0] # shape (z_dim,)
        z_t1_hat = ae.apply(params, z_t1[None,:], method=ae.decode)[0]     # shape (x_dim,)

        return z_t1, z_t1_hat # carry z_t1, output x_t1_hat
    
    # initial reconstructed state
    x0_hat = ae.apply(params, z0[None,:], method=ae.decode)[0]
    z_last, x_seq = jax.lax.scan(step_fn, z0, xs=None, length=N) # x_seq shape (N, nx)

    x_traj_hat = jnp.concatenate([x0_hat[None,:], x_seq], axis=0)  # shape (N+1, nx)

    return x_traj_hat

# ---------- Compare + basic metrics ----------
def compare_rollouts(ode_solver, ae, params, x0, dt, N):
    x_true = rollout_true(ode_solver, x0, dt, N)  # (N+1, x_dim)
    x_hat  = rollout_ae(ae, params, x0, N)       # (N+1, x_dim)
    mse_dim = jnp.mean((x_hat - x_true) ** 2, axis=0)
    mse_tot = jnp.mean((x_hat - x_true) ** 2)
    return np.array(x_true), np.array(x_hat), np.array(mse_dim), float(mse_tot)



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
    # fom = VanDerPol()
    fom = LorenzAttractor()

    # create the ODE solver with the desired dynamics to integrate
    ode_solver = ODESolver(fom)

    #-----------------------------------------------------------
    # Hyper parameters
    #-----------------------------------------------------------

    # trajectory parameters
    dt = 0.01     # time step
    N = 300       # number of time steps to integrate

    # training parameters
    num_steps = 1_500    # number of training steps
    traj_batch_size = 64  # number of trajectories per batch
    mini_batch_size = 32  # number of trajectories per mini-batch
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
    ae_config = AutoEncoderConfig(x_dim=fom.nx,
                                  z_dim=z_dim, 
                                  f_hidden_dim=f_hidden_dim, 
                                  E_hidden_dim=E_hidden_dim, 
                                  D_hidden_dim=D_hidden_dim)

    # loss function weights
    learning_rate = 1e-4  # learning rate
    lambda_rec = 0.8      # reconstruction loss weight
    lambda_dyn = 0.5      # latent dynamics loss weight
    lambda_reg = 1e-5     # L2 regularization weight
    config = OptimizerConfig(lambda_rec=lambda_rec,
                             lambda_dyn=lambda_dyn,
                             lambda_reg=lambda_reg,
                             learning_rate=learning_rate)

    #-----------------------------------------------------------
    # Autoencoder + Trainer
    #-----------------------------------------------------------

    # create the neural network model
    ae = AutoEncoder(config=ae_config)

    # create the trainer
    trainer = Trainer(ae,
                      rng_train,
                      fom.nx,
                      config=config)

    #-----------------------------------------------------------
    # Training loop
    #-----------------------------------------------------------

    # main training loop
    for step in range(num_steps):

        # Generate a fresh batch of trajectories (B_traj, N, nx)
        x_t, x_t1, rng_data = generate_batch_data(rng_data, ode_solver, traj_batch_size, dt, N)

        # Flatten to (num_pairs, nx)
        # Each trajectory contributes N pairs
        X = x_t.reshape(-1, fom.nx)      # (traj_batch_size * N, nx)
        Y = x_t1.reshape(-1, fom.nx)     # (traj_batch_size * N, nx)

        # Sample a mini-batch of pairs
        num_pairs = X.shape[0]

        # mini_batch_size here means "number of PAIRS per step"
        mb = min(mini_batch_size, num_pairs)
        
        # Instead of training on the full batch, we train on a random mini-batch of pairs
        rng_data, k_idx = random.split(rng_data)
        perm = random.permutation(k_idx, num_pairs)
        idx  = perm[:mb]
        xb, yb = X[idx], Y[idx]  # still (mb, nx) but stays on device

        # One training step
        trainer.state, metrics = trainer.train_step(trainer.state, xb, yb, step)

        # Log
        if step % print_every == 0:
            print(
                f"step {step:05d} | "
                f"loss={float(metrics.loss):.4f}  "
                f"rec={float(metrics.loss_rec):.4f}  "
                f"dyn={float(metrics.loss_dyn):.4f}  "
                f"reg={float(metrics.loss_reg):.4f}  "
                f"grad_norm={float(metrics.grad_norm):.4f}  "
                f"update_norm={float(metrics.update_norm):.4f}"
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
    # plt.plot(x_true[:, 0], x_true[:, 1], label="true x")
    # plt.plot(x_hat[:, 0], x_hat[:, 1], '--', label="recon x")
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

    #-----------------------------------------------------------

    # Sample a fresh x0 (match your system dimensionality & range)
    key_eval = jax.random.PRNGKey(123)
    x0 = jnp.array([10.0, -8.0, 25.0])  # e.g., Lorenz; adjust bounds as you prefer

    x_true, x_hat, mse_dim, mse_tot = compare_rollouts(
        ode_solver, ae, trainer.state.params, x0, dt, N
    )

    print("MSE per dim:", mse_dim)
    print("Total MSE:", mse_tot)

    # Time-series plot
    plt.figure(figsize=(10,4))
    for i in range(x_true.shape[1]):
        plt.plot(x_true[:, i], label=f"true x[{i}]")
        plt.plot(x_hat[:,  i], '--', label=f"AE x̂[{i}]")
    plt.xlabel("time step"); plt.ylabel("state"); plt.title("RK4 vs AE rollout")
    plt.legend(ncol=min(4, x_true.shape[1])); plt.tight_layout(); plt.show()

    # Phase/3D plot (auto-handles 2D vs 3D)
    if x_true.shape[1] == 2:
        plt.figure(figsize=(5,5))
        plt.plot(x_true[:,0], x_true[:,1], label="true", alpha=0.9)
        plt.plot(x_hat[:,0],  x_hat[:,1],  '--', label="AE", alpha=0.9)
        plt.xlabel("x[0]"); plt.ylabel("x[1]"); plt.title("Phase portrait")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    elif x_true.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_true[:,0], x_true[:,1], x_true[:,2], label="true", alpha=0.85)
        ax.plot(x_hat[:,0],  x_hat[:,1],  x_hat[:,2],  '--', label="AE", alpha=0.85)
        ax.set_xlabel("x[0]"); ax.set_ylabel("x[1]"); ax.set_zlabel("x[2]")
        ax.set_title("3D trajectory: true vs AE"); ax.legend(); plt.tight_layout(); plt.show()

