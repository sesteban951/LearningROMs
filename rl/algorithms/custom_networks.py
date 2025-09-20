##
#
# Custom assortment of nerual networks and wrappers to be used with Brax PPO training.
#
##

# python imports
from typing import Sequence

# flax impports
import flax.linen as nn
from flax import struct

# jax imports
import jax
import jax.numpy as jnp

# brax imports
from brax.envs.base import PipelineEnv
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo.networks import PPONetworks, make_inference_fn
from brax.training.types import Params


##################################### NETWORKS #########################################

# MLP config
@struct.dataclass
class MLPConfig:
    """
    # Example: (if activation_fn=nn.tanh, layer_sizes=[32, 16, 4], activate_final=True)
        - Input → Dense(32) → tanh → Dense(16) → tanh → Dense(4) → tanh → Output

    # Example initialization
        - Activation: tanh, sigmoid -> LeCun uniform/normal
        - Activation: relu, leaky_relu, elu, gelu -> Kaiming/He uniform/normal
        - softmax (for final layer) -> Xavier/Glorot uniform/normal
    """

    layer_sizes: Sequence[int]          # sizes of each hidden layer
    bias: bool = True                   # whether to use bias vector in dense layers
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_uniform()  # kernel initializer
    activate_final: bool = False        # whether to activate the final layer
    activation_fn: nn.Module = nn.tanh  # activation function to use


# basic MLP
class MLP(nn.Module):
    """
    Simple multi-layer perceptron (MLP) network.
    """

    # MLP configuration
    config: MLPConfig

    # main forward pass
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through this custom MLP network.
        """
        for i, layer_size in enumerate(self.config.layer_sizes):            
            
            # apply dense layer
            x = nn.Dense(
                features=layer_size,
                use_bias=self.config.bias,
                kernel_init=nn.initializers.lecun_uniform(),
                name=f"dense_{i}",
            )(x)

            # apply the activation function at every layer and optionally at the final layer
            if i != len(self.config.layer_sizes) - 1 or self.config.activate_final:
                x = self.config.activation_fn(x)

        return x


##################################### WRAPPER #########################################

@struct.dataclass
class BraxPPONetworksWrapper:
    """
    Thin wrapper to hold custom networks for PPO training in Brax.
    """

    policy_network: nn.Module # the policy network
    value_network: nn.Module  # the value network
    action_distribution: distribution.ParametricDistribution # distribution for actions


    def make_ppo_networks(
        self,
        obs_size: int,   # observations size
        act_size: int,   # actions size
        preprocess_observations_fn: types.PreprocessObservationFn # function to preprocess observations
    ) -> PPONetworks:
        """
        Create the PPO networks using the custom policy and value networks.

        Args:
            obs_size: Size of the observations.
            act_size: Size of the actions.
            preprocess_observations_fn: Function to preprocess observations.
        Returns:
            An instance of PPONetworks containing the policy and value networks.
        """

        # create action distribution. The policy network should output the parameters of this distribution
        action_dist = self.action_distribution(event_size=act_size)

        # create dummy observation for initialization
        dummy_obs = jnp.zeros((1, obs_size))

        # create a random key for initialization
        dummy_rng = jax.random.PRNGKey(0)

        # check that the size of the policy network matches the size of the action distribution
        dummy_params = self.policy_network.init(dummy_rng, dummy_obs)
        dummy_policy_output = self.policy_network.apply(dummy_params, dummy_obs)
        dummy_value_output = self.value_network.apply(dummy_params, dummy_obs)

        # shapes to make sure should match
        action_dist_params_shape = action_dist.param_size
        policy_output_shape = dummy_policy_output.shape[-1]
        value_output_shape = dummy_value_output.shape[-1]

        # assert the networks output shape matches the action distribution parameter size
        assert policy_output_shape == action_dist_params_shape, (
            f"Policy network output shape ({policy_output_shape}) does not match "
            f"action distribution parameter size ({action_dist_params_shape})."
        )

        # assert the value network output shape is correct
        assert value_output_shape == 1, (
            f"Value network output shape ({value_output_shape}) is not 1."
        )


        # create the Policy Network functions
        # init funcition
        def policy_init(key):
            return self.policy_network.init(key, dummy_obs)
        
        # apply function
        def policy_apply(processor_params, policy_params, obs):
            processed_obs = preprocess_observations_fn(obs, processor_params)
            return self.policy_network.apply(policy_params, processed_obs)
        
        # feedforward policy network
        policy_network = networks.FeedForwardNetwork(
            init=policy_init,
            apply=policy_apply
        )


        # create the Value Network functions
        # init function
        def value_init(key):
            return self.value_network.init(key, dummy_obs)
        
        # apply function
        def value_apply(processor_params, value_params, obs):
            processed_obs = preprocess_observations_fn(obs, processor_params)
            return jnp.squeeze(self.value_network.apply(value_params, processed_obs), axis=-1)

        # feedforward value network
        value_network = networks.FeedForwardNetwork(
            init=value_init,
            apply=value_apply
        )


        # bulid the PPONetworks dataclass
        ppo_networks = PPONetworks(
            policy_network=policy_network,
            value_network=value_network,
            action_distribution=action_dist,
        )

        return ppo_networks


##################################### UTILS #########################################

# util to print some details about a flax model
def print_model_summary(module: nn.Module, input_shape: Sequence[int]):
    """
    Print a readable summary of a flax neural network module.

    Args:
        module: The flax module to summarize.
        input_shape: The shape of the input to the module.
    """

    # Create a dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones(input_shape)
    print(module.tabulate(rng, dummy_input, depth=1))


# create a policy function
def make_policy_function(
        network_wrapper: BraxPPONetworksWrapper, # the networks wrapper
        params: Params,                          # the model parameters
        obs_size: int,                           # the observation size
        act_size: int,                           # the action size
        normalize_observations: bool = True,     # whether to normalize observations
        deterministic: bool = True               # whether to use deterministic actions
    ):
    """
    Create from a trained model a function that takes observations and returns actions.

    Args:
        network_wrapper: The BraxPPONetworksWrapper containing the policy and value networks.
        params: The trained model parameters.
        obs_size: The size of the observations.
        act_size: The size of the actions.
        normalize_observations: Whether to normalize observations using running statistics.
        deterministic: Whether to use deterministic actions (e.g., mean of the distribution).
    Returns:

    """

    # preprocessing of observatiosn functions
    if normalize_observations:
        preprocess_observations_fn = running_statistics.normalize
    else:
        preprocess_observations_fn = types.identity_observation_preprocessor
    
    # create the PPO networks
    ppo_networks = network_wrapper.make_ppo_networks(
        obs_size=obs_size,
        act_size=act_size,
        preprocess_observations_fn=preprocess_observations_fn
    )

    # make the inference function
    inference_fn = make_inference_fn(ppo_networks=ppo_networks)
    policy = inference_fn(params=params, deterministic=deterministic)

    return policy
