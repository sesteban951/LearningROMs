from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct
from mujoco import mjx
import mujoco

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig

##############################################################

if __name__ == "__main__":

    # get the current directory
    current_dir = Path(__file__).parent

    # import the environment
    env = CartPoleEnv(CartPoleConfig()) 