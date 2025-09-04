##
#
# Auto Encoder Model for ROM Learning
#
##

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct  # TODO: eventually store hyperparameters in a dataclass
from flax.training import train_state
import optax

from functools import partial

############################################################################
# AUTOENCODER
############################################################################

# AutoEncoder for learning latent representations
class AutoEncoder(nn.Module):

    # Hyperparameters
    z_dim: int         # Dimension of the latent space
    f_hidden_dim: int  # Hidden layer size for dynamics model
    E_hidden_dim: int  # Hidden layer size for encoder
    D_hidden_dim: int  # Hidden layer size for decoder

    # TODO: for better readability, separate Encoder, Decoder, and Dynamics into separate classes

    # main forward pass
    @nn.compact
    def __call__(self, x_t):
        """
        Forward pass through the autoencoder and dynamics model 
        
        Args:
            x_t:  FOM Input state at time t, shape (batch_size, data_dim)
        Returns:
            x_t_hat: Reconstructed FOM state at time t, shape (batch_size, data_dim)
            z_t:     Latent representation at time t, shape (batch_size, z_dim)
            z_t1:    Predicted latent representation at time t+1, shape (batch_size, z_dim)
        """

        # Encoder (x_t -> z_t)
        x = x_t
        x = nn.Dense(self.E_hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.E_hidden_dim)(x)
        x = nn.relu(x)
        z_t = nn.Dense(self.z_dim)(x)          # z_t = E(x_t)

        # Decoder (z_t -> x_t_hat)
        z = z_t
        z = nn.Dense(self.D_hidden_dim)(z)
        z = nn.relu(z)
        z = nn.Dense(self.D_hidden_dim)(z)
        z = nn.relu(z)
        x_t_hat = nn.Dense(x_t.shape[-1])(z)   # x_t_hat = D(z_t)

        # Latent dynamics model (z_t -> z_t+1)
        z = z_t
        z = nn.Dense(self.f_hidden_dim)(z)
        z = nn.relu(z)
        z = nn.Dense(self.f_hidden_dim)(z)
        z = nn.relu(z)
        z_t1_hat = nn.Dense(self.z_dim)(z)     # z_t1 = f(z_t)
        # TODO: consider adding figuring out how to bound this output

        return x_t_hat, z_t, z_t1_hat

############################################################################
# LOSS FUNCTION
############################################################################

# main loss function with all the loss components
def loss_fn(params, model, x_t, x_t1, 
            lambda_rec=0.8, lambda_dyn=0.2, lambda_reg=1e-4):
    """
    Compute the total loss for the autoencoder model
    
    Args:
        params:      Model parameters
        model:       AutoEncoder model instance
        x_t:         FOM Input state at time t, shape (batch_size, data_dim)
        x_t1:        FOM Input state at time t+1, shape (batch_size, data_dim)
        lambda_rec:  Weight for reconstruction loss
        lambda_dyn:  Weight for latent dynamics loss
        lambda_reg:  Weight for L2 regularization
    Returns:
        total_loss:  Combined loss value
        (loss_rec, loss_dyn, loss_reg): Tuple of individual loss components
    """

    # Forward pass the AutoEncoder
    x_t_hat, z_t, z_t1_hat = model.apply(params, x_t)

    # Encode true next state to get the target latent z_{t+1} = E(x_{t+1})
    # (call model again and take only the latent from the first block)
    _, z_t1_true, _ = model.apply(params, x_t1)

    # Reconstruction loss, λᵣ ‖x̂ₜ − xₜ‖²
    loss_rec = lambda_rec * jnp.mean((x_t_hat - x_t)**2)

    # Latent dynamics loss, λ_d‖ẑₜ₊₁ − zₜ₊₁‖²
    loss_dyn = lambda_dyn * jnp.mean((z_t1_hat - z_t1_true)**2)

    # L2 regularization on model parameters, λₗ * ‖θ‖²
    l2 = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    loss_reg = lambda_reg * l2

    # Total loss
    total_loss = loss_rec + loss_dyn + loss_reg

    return total_loss, (loss_rec, loss_dyn, loss_reg)

############################################################################
# TRAINER
############################################################################

# struct to hold metrics
@struct.dataclass
class Metrics:
    step: int        # training step
    loss: float      # total loss
    loss_rec: float  # reconstruction loss
    loss_dyn: float  # dynamics loss
    loss_reg: float  # regularization loss

# simple trainer class to handle training
class Trainer:

    def __init__(self, model, 
                       rng,
                       input_size, 
                       learning_rate, 
                       lambda_rec, 
                       lambda_dyn, 
                       lambda_reg):

        # initialize model and parameters
        self.model = model
        self.lambda_rec = lambda_rec
        self.lambda_dyn = lambda_dyn
        self.lambda_reg = lambda_reg

        # initialize model parameters
        dummy_input = jnp.ones((1, input_size))  # shape (batch_size=1, data_dim)
        params = self.model.init(rng, dummy_input)

        # setup the optimizer
        tx = optax.adam(learning_rate)

        # create the training state (this is a snapshot of the model + optimizer)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, # model function
                                                   params=params,             # model parameters
                                                   tx=tx)                     # optimizer
        
        # print model summary
        print(self.model.tabulate(rng, dummy_input))

    # single training step function
    @partial(jax.jit, static_argnums=0)
    def train_step(self, state, x_t, x_t1, step):
        """
        Single training step
        """

        # define a loss function wrapper to compute gradients
        def loss_fn_wrap(params):
            tot_loss, (loss_rec, loss_dyn, loss_reg) = loss_fn(params,      # model parameters to optimize
                                                               self.model,  # model instance
                                                               x_t, x_t1,   # input data
                                                               self.lambda_rec, self.lambda_dyn, self.lambda_reg) # hyperparameters

            return tot_loss, (loss_rec, loss_dyn, loss_reg)

        # make a function that computes the loss and its gradients
        grad_fn = jax.value_and_grad(loss_fn_wrap,  # function that only takes the model parameters as input
                                     has_aux=True)  # tells JAX that loss_fn returns auxiliary data in addition to the loss value
        
        # compute the loss and gradients w.r.t. the model parameters
        (loss, (loss_rec, loss_dyn, loss_reg)), grads = grad_fn(state.params)

        # apply the gradients to update the model parameters
        new_state = state.apply_gradients(grads=grads)

        # update the metrics
        metrics = Metrics(step=step, 
                          loss=loss, 
                          loss_rec=loss_rec, 
                          loss_dyn=loss_dyn, 
                          loss_reg=loss_reg)

        return new_state, metrics
