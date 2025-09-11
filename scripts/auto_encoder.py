##
#
# Auto Encoder Model for ROM Learning
#
##

# jax imports
import jax                      
import jax.numpy as jnp         # standard jax numpy
import jax.random as random     # jax random number generation
from functools import partial   # for partial function application

# flax imports
from flax import linen as nn           # neural network library
from flax import struct                # immutable dataclass
from flax.training import train_state  # simple train state for optimization

# optax imports
import optax         # gradient processing and optimization library

# custom imports
from ode_solver import ODESolver  # ODE solver class

############################################################################
# AUTOENCODER
############################################################################

# config for the AutoEncoder model
@struct.dataclass
class AutoEncoderConfig:

    x_dim: int         # Dimension of the FOM state
    z_dim: int         # Dimension of the ROM space
    f_hidden_dim: int  # Hidden layer size for dynamics model
    E_hidden_dim: int  # Hidden layer size for encoder
    D_hidden_dim: int  # Hidden layer size for decoder


# AutoEncoder for ROM learning
class AutoEncoder(nn.Module):

    # hyperparameters config
    config: AutoEncoderConfig

    # setup the layers
    def setup(self):

        # Encoder layers
        self.enc_fc1  = nn.Dense(self.config.E_hidden_dim, name="enc_fc1")
        self.enc_fc2  = nn.Dense(self.config.E_hidden_dim, name="enc_fc2")
        self.enc_out  = nn.Dense(self.config.z_dim,        name="enc_out")

        # Decoder layers
        self.dec_fc1  = nn.Dense(self.config.D_hidden_dim, name="dec_fc1")
        self.dec_fc2  = nn.Dense(self.config.D_hidden_dim, name="dec_fc2")
        self.dec_out  = nn.Dense(self.config.x_dim,        name="dec_out")

        # Latent dynamics layers
        self.dyn_fc1  = nn.Dense(self.config.f_hidden_dim, name="dyn_fc1")
        self.dyn_fc2  = nn.Dense(self.config.f_hidden_dim, name="dyn_fc2")
        self.dyn_out  = nn.Dense(self.config.z_dim,        name="dyn_out")

    # Encoder (x_t -> z_t)
    def encode(self, x_t):
        """
        Encoder network to map FOM state to latent space:
            zₜ = E(xₜ)
        """
        x = x_t
        x = nn.relu(self.enc_fc1(x))
        x = nn.relu(self.enc_fc2(x))
        z_t = self.enc_out(x)
        return z_t

    # Decoder (z_t -> x_t_hat)
    def decode(self, z_t):
        """
        Decoder network to map latent space to reconstructed FOM state:
            x̂ₜ = D(zₜ)
        """
        z = z_t
        z = nn.relu(self.dec_fc1(z))
        z = nn.relu(self.dec_fc2(z))
        x_t_hat = self.dec_out(z)
        return x_t_hat
    
    # Latent dynamics, residual form (z_t -> z_{t+1})
    def latent_dynamics(self, z_t):
        """
        Simple feedforward network to model latent dynamics:
            zₜ₊₁ = f_θ(zₜ)
        """
        z = z_t
        z = nn.relu(self.dyn_fc1(z))
        z = nn.relu(self.dyn_fc2(z))
        z_t1 = self.dyn_out(z)  # direct mapping
        return z_t1

    # Latent dynamics (z_t -> z_{t+1})
    def latent_dynamics_residual(self, z_t):
        """
        Simple feedforward network to model latent dynamics:
            zₜ₊₁ = zₜ + r_θ(zₜ)
        """
        z = z_t
        z = nn.relu(self.dyn_fc1(z))
        z = nn.relu(self.dyn_fc2(z))
        dz = self.dyn_out(z)
        z_t1 = z_t + dz          # residual increment
        return z_t1

    # Main forward pass (for convenience)
    def __call__(self, x_t):
        """
        Forward pass through the autoencoder and dynamics model 
        
        Args:
            x_t:  FOM Input state at time t, shape (batch_size, nx)
        Returns:
            x_t_hat: Reconstructed FOM state at time t, shape (batch_size, nx)
            z_t:     Latent representation at time t, shape (batch_size, z_dim)
            z_t1:    Predicted latent representation at time t+1, shape (batch_size, z_dim)
        """
        z_t      = self.encode(x_t)
        x_t_hat  = self.decode(z_t)
        # z_t1 = self.latent_dynamics(z_t)
        z_t1 = self.latent_dynamics_residual(z_t)
        return x_t_hat, z_t, z_t1


############################################################################
# LOSS FUNCTION
############################################################################

# loss function with full rollout penalities
def loss_rollout_fn(params, model, x_traj):
    """
    Compute the total loss for the autoencoder model using full trajectory rollouts
    
    Args:
        params:     Model parameters
        model:      AutoEncoder model instance
        x_traj:     FOM state trajecotry,   shape (mini_batch_size, N+1, nx)
    Returns:
        mse:        Mean Squared Error over the full trajectory, (excludes initial condition b/c redundant)
    """

    # initial conditions
    x0 = x_traj[:, 0, :]                               # shape (mini_batch_size, nx)
    z0 = model.apply(params, x0, method=model.encode)  # shape (mini_batch_size, z_dim)

    # function to step in latent space
    def step_fn(z_t, _):

        # estimate next latent state
        z_t1 = model.apply(params, z_t, method=model.latent_dynamics_residual) # shape (mini_batch_size, z_dim)

        # decode to reconstructed FOM state
        x_t1_hat = model.apply(params, z_t1, method=model.decode)    # shape (mini_batch_size, nx)

        return z_t1, x_t1_hat  # carry z_t1, output x_t1_hat
    
    # loop to rollout the trajectory in latent space
    z_last, x_hat_traj = jax.lax.scan(step_fn, z0, xs=None, length=x_traj.shape[1]-1)

    # swap axes to get shape (mini_batch_size, N, nx)
    x_hat_traj = jnp.swapaxes(x_hat_traj, 0, 1)  # shape (mini_batch_size, N, nx)

    # compare the true trajectory (excludes initial condition)
    x_true_traj = x_traj[:, 1:, :]  # shape (mini_batch_size, N, nx)

    # MSE 
    mse = jnp.mean((x_hat_traj - x_true_traj)**2)

    return mse

# main loss function with all the loss components
def loss_fn(params, model, x_traj, opt_config):
    """
    Compute the total loss for the autoencoder model
    
    Args:
        params:      Model parameters
        model:       AutoEncoder model instance
        x_traj:      FOM state full trajectories,   shape (mini_batch_size, N+1, nx)
        opt_config:  OptimizerConfig instance with options
    Returns:
        total_loss:  Combined loss value
        (loss_rec, loss_dyn, loss_reg): Tuple of individual loss components
    """

    # shape of the data
    nx = x_traj.shape[2]

    # parse the trajectory into pairs
    x_t  = x_traj[:, :-1, :].reshape(-1, nx)   # shape (mini_batch_size * N, nx)
    x_t1 = x_traj[:, 1:, :].reshape(-1, nx)    # shape (mini_batch_size * N, nx)
    
    # Forward pass the AutoEncoder
    x_t_hat, z_t, z_t1_hat = model.apply(params, x_t)

    # Encode true next state to get the target latent z_{t+1} = E(x_{t+1})
    z_t1_true = model.apply(params, x_t1, method=model.encode)
    z_t1_true = jax.lax.stop_gradient(z_t1_true)  # stop gradient flow to encoder
                                                  # decouples dynamics loss from encoder update

    # Reconstruction loss, λ_rec * (1/B) * Σ  ‖x̂ₜ − xₜ‖²
    loss_rec = opt_config.lambda_rec * jnp.mean((x_t_hat - x_t)**2)

    # Latent dynamics loss, λ_dyn * (1/B) * Σ ‖ẑₜ₊₁ − zₜ₊₁‖²
    loss_dyn = opt_config.lambda_dyn * jnp.mean((z_t1_hat - z_t1_true)**2)

    # Rollout loss, λ_roll * (1/B) * Σ ‖x̂ₜ₊ₖ − xₜ₊ₖ‖²
    loss_roll = opt_config.lambda_roll * loss_rollout_fn(params, model, x_traj)

    # L2 model param regularization (no reg on biases), λ_reg * Σ ‖θ‖²
    l2 = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim > 1)
    loss_reg = opt_config.lambda_reg * l2

    # Total loss
    total_loss = loss_rec + loss_dyn + loss_roll + loss_reg

    return total_loss, (loss_rec, loss_dyn, loss_roll, loss_reg)


############################################################################
# Data Normalizer
############################################################################

# normalize data to zero mean and unit variance
@struct.dataclass
class Normalizer:

    mean: jnp.ndarray     # mean of the data
    std: jnp.ndarray      # standard deviation of the data
    epsilon: float = 1e-6 # small number to avoid division by zero

    # normalize the data
    def normalize(self, x):
        return (x - self.mean) / (self.std + self.epsilon)
    
    # denormalize the data
    def denormalize(self, x_norm):
        return x_norm * (self.std + self.epsilon) + self.mean

############################################################################
# TRAINER
############################################################################

# struct to hold simulation parameters
@struct.dataclass
class SimulationConfig:

    # simulation time parameters
    dt: float       # total simulation time
    N: int         # simulation time step


# struct to hold training parameters
@struct.dataclass
class TrainingConfig:

    # number of total steps
    num_steps: int       # total number of training steps

    # batch size
    batch_size: int       # mini-batch size for training
    mini_batch_size: int  # size of mini-batches for gradient computation

    # updating frequency
    print_every: int      # print metrics every n steps


# struct to hold optimization parameters
@struct.dataclass
class OptimizerConfig:
    
    # Loss weights
    lambda_rec:  float   # reconstruction loss weight
    lambda_dyn:  float   # dynamics loss weight
    lambda_roll: float   # rollout loss weight
    lambda_reg:  float   # regularization weight

    # Learning rate
    learning_rate: float  # descending step size


# struct to hold metrics
@struct.dataclass
class Metrics:

    # progress metrics
    step: int        # current training step
    loss: float      # total loss               
    loss_rec: float  # reconstruction loss      L_rec = λ_rec * (1/B) * Σ  ‖x̂ₜ − xₜ‖²
    loss_dyn: float  # dynamics loss            L_dyn = λ_dyn * (1/B) * Σ  ‖ẑₜ₊₁ − zₜ₊₁‖²
    loss_roll: float   # rollout loss             L_ro = λ_ro * (1/B) * Σ  ‖x̂ₜ₊ₖ − xₜ₊ₖ‖²
    loss_reg: float  # regularization loss      L_reg = λ_reg * ‖θ‖²

    # step information
    grad_norm: float   # gradient norm          ‖g‖₂ = ‖∇_θ L‖₂
    update_norm: float # parameter update norm  ‖Δθ‖₂ = ‖θₖ₊₁ − θₖ‖₂


# simple trainer class to handle training
class Trainer:

    # initialize the trainer
    def __init__(self, model: nn.Module, 
                       ode_solver: ODESolver,
                       sim_config: SimulationConfig,
                       training_config: TrainingConfig,
                       opt_config: OptimizerConfig,
                       rng: jax.random.PRNGKey):

        # initialize model and parameters
        self.model = model

        # initialize the ODE solver
        self.ode_solver = ode_solver

        # load all the configs
        self.opt_config = opt_config
        self.sim_config = sim_config
        self.training_config = training_config

        # split the RNG
        rng_data, rng_init, rng_norm, rng_tab = jax.random.split(rng, 4)
        self.rng_data = rng_data

        # initialize model parameters
        dummy_input = jnp.ones((1, self.ode_solver.dynamics.nx))  # shape (batch_size=1, nx)
        params = self.model.init(rng_init, dummy_input)

        # setup normalizer
        x_traj_sample, _ = self.generate_batch_data(rng_norm) # shape (batch_size, N+1, nx)
        x_mean = jnp.mean(x_traj_sample, axis=(0,1))          # mean over batch and time, shape (nx,)
        x_std  = jnp.std(x_traj_sample, axis=(0,1))           # std over batch and time, shape (nx,)
        self.normalizer = Normalizer(mean=x_mean, std=x_std)

        # setup the optimizer
        tx = optax.adam(self.opt_config.learning_rate)

        # create the training state (this is a snapshot of the model + optimizer)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, # model function
                                                   params=params,             # model parameters
                                                   tx=tx)                     # optimizer
        
        # print model summary
        print(self.model.tabulate(rng_tab, dummy_input))

    
    # function that generates data
    @partial(jax.jit, static_argnums=(0,))  # static self
    def generate_batch_data(self, rng):
        """
        Generate training data by simulating the FOM from random initial conditions

        Args:
            rng:      JAX random key
        Returns:
            x_traj:   FOM state trajectory, shape (batch_size, N+1, nx)
            rng:      Updated JAX random key
        """

        # split the RNG
        rng, subkey = jax.random.split(rng)

        # generate random intial conditions
        x_min = self.ode_solver.dynamics.x_min  # shape (nx,)
        x_max = self.ode_solver.dynamics.x_max  # shape (nx,)
        x0_batch = jax.random.uniform(subkey,
                                      shape=(self.training_config.batch_size, self.ode_solver.dynamics.nx),
                                      minval=x_min,
                                      maxval=x_max)  # shape (batch_size, nx)
        
        # solve ODE for all initial conditions in parallel, shape (batch_size, N+1, nx)
        x_traj_batch = self.ode_solver.forward_propagate_cl_batch(x0_batch,
                                                                  self.sim_config.dt,
                                                                  self.sim_config.N)

        return x_traj_batch, rng


    # single training step function
    @partial(jax.jit, static_argnums=0, donate_argnums=(1,))
    def train_step(self, state, x_traj, step):
        """
        Single training step

        Args:
            state:  Current TrainState (model + optimizer)
            x_traj: FOM Input state trajectory, shape (mini_batch_size, N, nx)
            step:   Current training step (for logging)
        Returns:
            new_state: Updated TrainState after applying gradients
            metrics:   Metrics dataclass with training info
        """

        # define a loss function wrapper to compute gradients
        def loss_fn_wrap(params):
            return loss_fn(params,          # model parameters to optimize
                           self.model,      # model instance
                           x_traj,          # input data
                           self.opt_config) # optimizer config
            
        # make a function that computes the loss and its gradients
        grad_fn = jax.value_and_grad(loss_fn_wrap,  # function that only takes the model parameters as input
                                     has_aux=True)  # tells JAX that loss_fn returns auxiliary data in addition to the loss value
        
        # compute the loss and gradients w.r.t. the model parameters
        (loss, (loss_rec, loss_dyn, loss_roll, loss_reg)), grads = grad_fn(state.params)

        # apply the gradients to update the model parameters
        new_state = state.apply_gradients(grads=grads)

        # compute the gradient norm and update norm for logging
        grad_norm = optax.global_norm(grads)              
        delta_params = jax.tree_util.tree_map(lambda new, old: new - old,
                                                     new_state.params, state.params)
        update_norm = optax.global_norm(delta_params)

        # update the metrics
        metrics = Metrics(step=step, 
                          loss=loss, 
                          loss_rec=loss_rec, 
                          loss_dyn=loss_dyn, 
                          loss_roll=loss_roll,
                          loss_reg=loss_reg,
                          grad_norm=grad_norm,
                          update_norm=update_norm)

        return new_state, metrics
    

    # main training loop
    def train(self):
        """
        Main training loop
        """

        # parameters that are reused
        mb_size = self.training_config.mini_batch_size
        print_every = self.training_config.print_every

        # loop over training steps
        for step in range(self.training_config.num_steps):

            # generate fresh training data
            x_traj, self.rng_data = self.generate_batch_data(self.rng_data)  # shape (batch_size, N+1, nx)
            
            # sample a mini-batch
            self.rng_data, rng_idx = jax.random.split(self.rng_data)
            mb_idx = jax.random.choice(rng_idx,
                                       a=self.training_config.batch_size,
                                       shape=(mb_size,),
                                       replace=False)  # shape (mini_batch_size,)
            xb = x_traj[mb_idx, :, :]                  # shape (mini_batch_size, N+1, nx)

            # normalize the data before sending to the model
            xb = self.normalizer.normalize(xb)

            # perform a single training step
            self.state, metrics = self.train_step(self.state, xb, step)

            # print Logging info
            if step % print_every == 0:
                print(
                    f"Step {step:05d} | "
                    f"L_tot = {metrics.loss:.4f}, "
                    f"L_rec = {metrics.loss_rec:.4f}, "
                    f"L_dyn = {metrics.loss_dyn:.4f}, "
                    f"L_roll = {metrics.loss_roll:.4f}, "
                    f"L_reg = {metrics.loss_reg:.4f}, "
                    f"‖g‖ = {metrics.grad_norm:.4f}, "
                    f"‖Δθ‖ = {metrics.update_norm:.4f}"
                )

        return self.state.params  # return the trained model parameters
