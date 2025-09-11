##
#
#  Main script to demonstrate ROM Learning
#
##

# standard imports 
import numpy as np                       # standard numpy
import matplotlib.pyplot as plt          # standard matplotlib
import time                              # standard time

# jax imports
import jax
import jax.numpy as jnp         # standard jax numpy

# custom imports
from fom import *           # full order model examples
from ode_solver import *    # ODE solver for traj generation
from auto_encoder import *  # autoencoder, training, configs

#############################################################################
# Helper functions
#############################################################################

# rollout a single trajectory using the ode solver
def rollout_true(ode_solver, x0, dt, N):
    
    # ode solver rollout
    x_traj = ode_solver.forward_propagate_cl(x0, dt, N)  # you already have this
    
    return x_traj

# rollout a single trajectory using the auto encoder
def rollout_ae(ae, params, normalizer, x0, N):

    # normalize the intial condition
    x0_normalized = normalizer.normalize(x0)

    # z0 from x0
    z0 = ae.apply(params, x0_normalized[None,:], method=ae.encode)  # shape (1, z_dim)
    z0 = z0[0]  # remove batch dim, shape (z_dim,)

    # step function in latent space
    def step_fn(z_t, _):

        # z_t1 = ae.apply(params, z_t[None,:], method=ae.latent_dynamics)[0] # shape (z_dim,)
        z_t1 = ae.apply(params, z_t[None,:], method=ae.latent_dynamics_residual)[0] # shape (z_dim,)
        x_t1_hat_normalized = ae.apply(params, z_t1[None,:], method=ae.decode)[0]   # shape (x_dim,)
        x_t1_hat = normalizer.denormalize(x_t1_hat_normalized)                

        return z_t1, x_t1_hat # carry z_t1, output x_t1_hat

    # initial reconstructed state
    x0_hat_normalized = ae.apply(params, z0[None,:], method=ae.decode)[0]
    x0_hat = normalizer.denormalize(x0_hat_normalized)                       

    # rollout in latent space
    z_last, x_seq = jax.lax.scan(step_fn, z0, xs=None, length=N) # x_seq shape (N, nx)

    x_traj_hat = jnp.concatenate([x0_hat[None,:], x_seq], axis=0)  # shape (N+1, nx)

    return x_traj_hat

# compare rollouts of true dynamics vs AE dynamics from same x0
def compare_rollouts(ode_solver, ae, params, normalizer, x0, dt, N):

    # rollout true dynamics and AE dynamics
    x_true = rollout_true(ode_solver, x0, dt, N)        # (N+1, nx)
    x_hat  = rollout_ae(ae, params, normalizer, x0, N)  # (N+1, nx)

    # compute MSE
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
    # fom = CartPole()

    # create the ODE solver with the desired dynamics to integrate
    ode_solver = ODESolver(fom)

    #-----------------------------------------------------------
    # Hyper parameters
    #-----------------------------------------------------------

    # trajectory parameters
    dt = 0.01     # time step
    N = 50       # number of time steps to integrate
    sim_config = SimulationConfig(dt=dt, 
                                  N=N)

    # training parameters
    num_steps = 2_000      # number of training steps
    traj_batch_size = 128  # number of trajectories per batch
    mini_batch_size = 64  # number of trajectories per mini-batch
    print_every = 50       # print every n steps
    training_config = TrainingConfig(num_steps=num_steps,
                                     batch_size=traj_batch_size,
                                     mini_batch_size=mini_batch_size,
                                     print_every=print_every)

    # loss function weights
    learning_rate = 5e-4  # learning rate
    lambda_rec = 0.5      # reconstruction loss weight
    lambda_dyn = 0.5      # latent dynamics loss weight
    lambda_roll = 0.2     # rollout loss weight
    lambda_reg = 1e-4     # L2 regularization weight
    opt_config = OptimizerConfig(lambda_rec=lambda_rec,
                                 lambda_dyn=lambda_dyn,
                                 lambda_roll=lambda_roll,
                                 lambda_reg=lambda_reg,
                                 learning_rate=learning_rate)
    
    # autoencoder parameters
    z_dim = 2          # latent space dimension
    f_hidden_dim = 32  # hidden layer size for dynamics model
    E_hidden_dim = 32  # hidden layer size for Encoder
    D_hidden_dim = 32  # hidden layer size for Decoder
    ae_config = AutoEncoderConfig(x_dim=fom.nx,
                                  z_dim=z_dim, 
                                  f_hidden_dim=f_hidden_dim, 
                                  E_hidden_dim=E_hidden_dim, 
                                  D_hidden_dim=D_hidden_dim)
    
    # random key
    # seed = 0
    seed = int(time.time())  # use current time as seed
    rng = random.PRNGKey(seed)
    
    #-----------------------------------------------------------
    # Autoencoder + Trainer
    #-----------------------------------------------------------

    # create the neural network model
    ae = AutoEncoder(config=ae_config)

    # create the trainer
    trainer = Trainer(ae,
                      ode_solver,
                      sim_config,
                      training_config,
                      opt_config,
                      rng)

    #-----------------------------------------------------------
    # Training loop
    #-----------------------------------------------------------

    # train the model
    t_start = time.time()
    params = trainer.train()  
    t_end = time.time()

    print(f"Training time: {t_end - t_start:.2f} seconds")

    #-----------------------------------------------------------
    # Plotting
    #-----------------------------------------------------------

    # Pick one trajectory from the batch
    key = jax.random.PRNGKey(42)
    x_t, _ = trainer.generate_batch_data(key)  # (batch_size, N+1, nx)
    x_true = np.array(x_t[0])  # unnormalized true traj

    # Normalize before AE
    x_true_norm = trainer.normalizer.normalize(x_true)
    x_hat_norm, _, _ = ae.apply(params, x_true_norm)
    x_hat = trainer.normalizer.denormalize(x_hat_norm)

    # plot reconstructions
    plt.figure(figsize=(10,4))
    for i in range(x_true.shape[1]):
        (true_line,) = plt.plot(x_true[:, i], label=f"true dim {i}")
        color = true_line.get_color()
        plt.plot(x_hat[:, i], ls='--', color=color, label=f"recon dim {i}")
    plt.xlabel("time step"); plt.ylabel("state value"); plt.legend()
    plt.title("True vs reconstructed trajectory"); plt.show()

    #-----------------------------------------------------------

    # Encode a trajectory (latent space)
    _, z_t, _ = ae.apply(params, x_true_norm)
    z_t = np.array(z_t)

    plt.figure(figsize=(6,6))
    for i in range(z_t.shape[1]):
        plt.plot(z_t[:, i], label=f"z dim {i}", marker='o', alpha=0.7)
    plt.xlabel("time step"); plt.ylabel("latent value"); plt.legend()
    plt.title("Encoded latent trajectory")
    plt.grid(True); plt.show()

    #-----------------------------------------------------------

    # Encode true z_t and predict z_{t+1}
    _, z_t, z_t1_hat = ae.apply(params, x_true_norm[:-1])  
    _, z_t1_true, _  = ae.apply(params, x_true_norm[1:])

    z_t1_hat = np.array(z_t1_hat)
    z_t1_true = np.array(z_t1_true)

    plt.figure(figsize=(6,6))
    for i in range(z_t1_true.shape[1]):
        (true_line,) = plt.plot(z_t1_true[:, i], label=f"true z[{i}]")
        color = true_line.get_color()
        plt.plot(z_t1_hat[:, i], ls='--', color=color, label=f"pred ẑ[{i}]")
    plt.xlabel("time step"); plt.ylabel("latent value"); plt.legend()
    plt.title("Latent dynamics prediction"); plt.grid(True); plt.show()

    #-----------------------------------------------------------

    # Rollout comparison
    x_min = trainer.ode_solver.dynamics.x_min
    x_max = trainer.ode_solver.dynamics.x_max
    x0 = jax.random.uniform(key, shape=(fom.nx,), minval=x_min, maxval=x_max)
    x_true, x_hat, mse_dim, mse_tot = compare_rollouts(
        ode_solver, ae, params, trainer.normalizer, x0, dt, N
    )

    print("MSE per dim:", mse_dim)
    print("Total MSE:", mse_tot)

    # Time-series plot
    plt.figure(figsize=(10,4))
    for i in range(x_true.shape[1]):
        (true_line,) = plt.plot(x_true[:, i], label=f"true x[{i}]")
        color = true_line.get_color()
        plt.plot(x_hat[:,  i], ls='--', color=color, label=f"AE x̂[{i}]")
    plt.xlabel("time step"); plt.ylabel("state"); plt.title("RK4 vs AE rollout")
    plt.legend(ncol=min(4, x_true.shape[1])); plt.tight_layout(); plt.show()

    # 3D trajectory plot
    if x_true.shape[1] == 3:
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_true[:,0], x_true[:,1], x_true[:,2], label="true", alpha=0.85)
        ax.plot(x_hat[:,0],  x_hat[:,1],  x_hat[:,2],  '--', label="AE", alpha=0.85)
        ax.set_xlabel("x[0]"); ax.set_ylabel("x[1]"); ax.set_zlabel("x[2]")
        ax.set_title("3D trajectory: true vs AE"); ax.legend()
        plt.tight_layout(); plt.show()