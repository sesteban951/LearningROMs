##
#
# Custom assortment of nerual networks.
#
##

# python imports
from typing import Sequence

# flax impports
import flax.linen as nn

# jax imports
import jax
import jax.numpy as jnp

# basic MLP
class MLP(nn.Module):
    """
    Simple multi-layer perceptron (MLP) network.
    """

    # MLP configuration
    layer_sizes: Sequence[int]               # sizes of each hidden layer
    bias: bool = True                        # whether to use bias in dense layers
    activate_final: bool = False             # whether to activate the final layer
    activate_final_fn: callable = nn.tanh    # activation function to use at final layer

    # main forward pass
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through this custom MLP network.
        """
        for i, layer_size in enumerate(self.layer_sizes):            
            
            # apply dense layer
            x = nn.Dense(
                features=layer_size,
                use_bias=self.bias,
                kernel_init=nn.initializers.lecun_uniform(),
                name=f"dense_{i}",
            )(x)

            # apply the activation function
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activate_final_fn(x)

        return x
            


