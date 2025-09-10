##
#
# Auto Encoder Model for ROM Learning
#
##

# jax imports
import jax                      
import jax.numpy as jnp         # standard jax numpy
from functools import partial   # for partial function application

# flax imports
from flax import linen as nn           # neural network library
from flax import struct                # immutable dataclass
from flax.training import train_state  # simple train state for optimization

# optax imports
import optax         # gradient processing and optimization library


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

    # Latent dynamics (z_t -> z_{t+1})
    def latent_dynamics(self, z_t):
        """
        Simple feedforward network to model latent dynamics:
            zₜ₊₁ = f_θ(zₜ)
        """
        z = z_t
        z = nn.relu(self.dyn_fc1(z))
        z = nn.relu(self.dyn_fc2(z))
        z_t1_hat = self.dyn_out(z)
        return z_t1_hat

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
        z_t1_hat = self.latent_dynamics(z_t)
        return x_t_hat, z_t, z_t1_hat


############################################################################
# LOSS FUNCTION
############################################################################

# main loss function with all the loss components
def loss_fn(params, model, x_t, x_t1, opt_config):
    """
    Compute the total loss for the autoencoder model
    
    Args:
        params:      Model parameters
        model:       AutoEncoder model instance
        x_t:         FOM Input state at time t,   shape (mini_batch_size, nx)
        x_t1:        FOM Input state at time t+1, shape (mini_batch_size, nx)
        opt_config:  OptimizerConfig instance with options
    Returns:
        total_loss:  Combined loss value
        (loss_rec, loss_dyn, loss_reg): Tuple of individual loss components
    """

    # Forward pass the AutoEncoder
    x_t_hat, z_t, z_t1_hat = model.apply(params, x_t)

    # Encode true next state to get the target latent z_{t+1} = E(x_{t+1})
    # (call model again and take only the latent from the first block)
    z_t1_true = model.apply(params, x_t1, method=model.encode)

    # Reconstruction loss, λ_rec * (1/B) * Σ  ‖x̂ₜ − xₜ‖²
    loss_rec = opt_config.lambda_rec * jnp.mean((x_t_hat - x_t)**2)

    # Latent dynamics loss, λ_dyn * (1/B) * Σ ‖ẑₜ₊₁ − zₜ₊₁‖²
    loss_dyn = opt_config.lambda_dyn * jnp.mean((z_t1_hat - z_t1_true)**2)

    # L2 regularization on model parameters (no reg on the biases), λ_reg * Σ ‖θ‖²
    l2 = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim > 1)
    loss_reg = opt_config.lambda_reg * l2

    # Total loss
    total_loss = loss_rec + loss_dyn + loss_reg

    return total_loss, (loss_rec, loss_dyn, loss_reg)


############################################################################
# TRAINER
############################################################################

# struct to hold optimization parameters
@struct.dataclass
class OptimizerConfig:
    
    # Loss weights
    lambda_rec: float     # reconstruction loss weight
    lambda_dyn: float     # dynamics loss weight
    lambda_reg: float     # regularization weight

    # Learning rate
    learning_rate: float  # descending step size


# struct to hold metrics
@struct.dataclass
class Metrics:

    # progress metrics
    step: int        # current training step
    loss: float      # total loss               
    loss_rec: float  # reconstruction loss      L_rec = λ_r * ‖x̂ₜ − xₜ‖²
    loss_dyn: float  # dynamics loss            L_dyn = λ_d * ‖ẑₜ₊₁ − zₜ₊₁‖²
    loss_reg: float  # regularization loss      L_reg = λₗ * ‖θ‖²

    # step information
    grad_norm: float   # gradient norm          ‖g‖₂ = ‖∇_θ L‖₂
    update_norm: float # parameter update norm  ‖Δθ‖₂ = ‖θₖ₊₁ − θₖ‖₂


# simple trainer class to handle training
class Trainer:

    def __init__(self, model, 
                       rng,
                       input_size, 
                       config):

        # initialize model and parameters
        self.model = model

        # load the config
        self.config = config

        # split the RNG
        rng, rng_init, rng_tab = jax.random.split(rng, 3)
        self.rng = rng

        # initialize model parameters
        dummy_input = jnp.ones((1, input_size))  # shape (batch_size=1, nx)
        params = self.model.init(rng_init, dummy_input)

        # setup the optimizer
        tx = optax.adam(self.config.learning_rate)

        # create the training state (this is a snapshot of the model + optimizer)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, # model function
                                                   params=params,             # model parameters
                                                   tx=tx)                     # optimizer
        
        # print model summary
        print(self.model.tabulate(rng_tab, dummy_input))

    # single training step function
    @partial(jax.jit, static_argnums=0)
    def train_step(self, state, x_t, x_t1, step):
        """
        Single training step

        Args:
            state:  Current TrainState (model + optimizer)
            x_t:    FOM Input state at time t,   shape (mini_batch_size, nx)
            x_t1:   FOM Input state at time t+1, shape (mini_batch_size, nx)
            step:   Current training step (for logging)
        Returns:
            new_state: Updated TrainState after applying gradients
            metrics:   Metrics dataclass with training info
        """

        # define a loss function wrapper to compute gradients
        def loss_fn_wrap(params):
            tot_loss, (loss_rec, loss_dyn, loss_reg) = loss_fn(params,      # model parameters to optimize
                                                               self.model,  # model instance
                                                               x_t, x_t1,   # input data
                                                               self.config) # optimizer config
            return tot_loss, (loss_rec, loss_dyn, loss_reg)

        # make a function that computes the loss and its gradients
        grad_fn = jax.value_and_grad(loss_fn_wrap,  # function that only takes the model parameters as input
                                     has_aux=True)  # tells JAX that loss_fn returns auxiliary data in addition to the loss value
        
        # compute the loss and gradients w.r.t. the model parameters
        (loss, (loss_rec, loss_dyn, loss_reg)), grads = grad_fn(state.params)

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
                          loss_reg=loss_reg,
                          grad_norm=grad_norm,
                          update_norm=update_norm)

        return new_state, metrics
