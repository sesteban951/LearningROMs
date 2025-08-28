# jax imports
import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct
from brax import envs

# mujoco imports
import mujoco


# struct to hold the configuration parameters
@struct.dataclass
class CartPoleConfig:
    """Config dataclass for cart-pole."""

    # model path (NOTE: relative the script that calls this class)
    model_path: str = "./models/cart_pole.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 1

    # Reward function coefficients
    reward_cart_pos: float = 0.1
    reward_pole_pos: float = 1.0
    reward_cart_vel: float = 0.01
    reward_pole_vel: float = 0.01
    reward_control: float = 0.001

    # Ranges for sampling initial conditions
    lb_pos: float = -1.5
    ub_pos: float =  1.5
    lb_theta: float = -jnp.pi
    ub_theta: float =  jnp.pi
    lb_vel: float = -10.0
    ub_vel: float =  10.0
    lb_theta_dot: float = -10.0
    ub_theta_dot: float =  10.0


# environment class
class CartPoleEnv(PipelineEnv):
    """
    Environment for training a cart-pole swingup task.

    States: x = (pos, theta, vel, dtheta), shape=(4,)
    Observations: o = (pos, cos(theta), sin(theta), vel, dtheta), shape=(5,)
    Actions: a = tau, the force on the cart, shape=(1,)
    """

    # initialize the environment
    def __init__(self, config: CartPoleConfig = CartPoleConfig()):

        # load the config
        self.config = config

        # create the brax system
        # TODO: eventually refactor to use MJX fully instead since BRAX is now moving away from MJCF
        # see https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
        mj_model = mujoco.MjModel.from_xml_path(self.config.model_path)
        sys = mjcf.load_model(mj_model)

        # insantiate the parent class
        super().__init__(
            sys=sys,                                             # brax system defining the kinematic tree and other properties
            backend="mjx",                                       # defining the physics pipeline
            n_frames=self.config.physics_steps_per_control_step  # number of times to step the physics pipeline per control step
                                                                 # for each environment step
        )
        # n_frames: number of sim steps per control step, dt = n_frames * xml_dt

        # print message
        print(f"Initialized CartPoleEnv with model [{self.config.model_path}].")

    # reset function
    def reset(self, rng):
        """
        Resets the environment to an initial state.

        Args:
            rng: jax random number generator (jax.Array)

        Returns:
            State: brax.envs.base.State object
                   Environment state for training and inference.
        """
        
        # split the rng to sample unique initial conditions
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # set the state bounds for sampling initial conditions
        qpos_lb = jnp.array([self.config.lb_pos, self.config.lb_theta])
        qpos_ub = jnp.array([self.config.ub_pos, self.config.ub_theta])
        qvel_lb = jnp.array([self.config.lb_vel, self.config.lb_theta_dot])
        qvel_ub = jnp.array([self.config.ub_vel, self.config.ub_theta_dot])

        # sample the initial state
        qpos = jax.random.uniform(rng1, (2,), minval=qpos_lb, maxval=qpos_ub)
        qvel = jax.random.uniform(rng2, (2,), minval=qvel_lb, maxval=qvel_ub)
        
        # reset the physics state
        data = self.pipeline_init(qpos, qvel)

        # reset the observation
        obs = self._compute_obs(data)

        # reset reward
        reward, done = jnp.zeros(2)

        # reset the metrics
        metrics = {"reward_cart_pos": jnp.array(0.0),
                   "reward_pole_pos": jnp.array(0.0),
                   "reward_cart_vel": jnp.array(0.0),
                   "reward_pole_vel": jnp.array(0.0),
                   "reward_control": jnp.array(0.0)}

        # state info
        info = {"rng": rng,
                "step": 0}
        
        return State(pipeline_state=data,
                     obs=obs,
                     reward=reward,
                     done=done,
                     metrics=metrics,
                     info=info)

    # physics step function
    def step(self, state, action):
        """
        Step the environment by one timestep.

        Args:
            state: brax.envs.base.State object
                   The current state of the environment.
            action: jax.Array
                    The action to be applied to the environment.
        """

        # step the physics
        data = self.pipeline_step(state.pipeline_state, action)

        # update the observations
        obs = self._compute_obs(data)

        # data
        pos = data.qpos[0]
        theta = data.qpos[1]
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        vel = data.qvel[0]
        theta_dot = data.qvel[1]
        tau = data.ctrl

        # special angle error
        theta_angle_vec = jnp.array([cos_theta - 1.0, sin_theta]) # want (0, 0)

        # compute error terms
        cart_pos_err = jnp.square(pos).sum()
        pole_pos_err = jnp.square(theta_angle_vec).sum()
        cart_vel_err = jnp.square(vel).sum()
        theta_dot_err = jnp.square(theta_dot).sum()
        control_err = jnp.square(tau).sum()

        # compute the rewards
        reward_cart_pos = -self.config.reward_cart_pos * cart_pos_err
        reward_pole_pos = -self.config.reward_pole_pos * pole_pos_err
        reward_cart_vel = -self.config.reward_cart_vel * cart_vel_err
        reward_pole_vel = -self.config.reward_pole_vel * theta_dot_err
        reward_control  = -self.config.reward_control * control_err

        # compute the rewards with exp kernels
        # std_cart_pos = 0.3    # tuning parameter
        # std_pole_pos = 0.1
        # std_cart_vel = 10.0
        # std_pole_vel = 10.0
        # std_control = 0.01
        # reward_cart_pos = jnp.exp(-cart_pos_err / std_cart_pos) * self.config.reward_cart_pos
        # reward_pole_pos = jnp.exp(-pole_pos_err / std_pole_pos) * self.config.reward_pole_pos
        # reward_cart_vel = jnp.exp(cart_vel_err / std_cart_vel) * self.config.reward_cart_vel
        # reward_pole_vel = jnp.exp(theta_dot_err / std_pole_vel) * self.config.reward_pole_vel
        # reward_control = jnp.exp( control_err / std_control) * self.config.reward_control

        # compute the total reward
        reward = (reward_cart_pos + reward_pole_pos + 
                  reward_cart_vel + reward_pole_vel + 
                  reward_control)
        # reward = reward / jnp.sum(jnp.array([self.config.reward_cart_pos,
        #                                     self.config.reward_pole_pos,
        #                                     self.config.reward_cart_vel,
        #                                     self.config.reward_pole_vel,
        #                                     self.config.reward_control]))
        
        # update the metrics and info dictionaries
        state.metrics["reward_cart_pos"] = reward_cart_pos
        state.metrics["reward_pole_pos"] = reward_pole_pos
        state.metrics["reward_cart_vel"] = reward_cart_vel
        state.metrics["reward_pole_vel"] = reward_pole_vel
        state.metrics["reward_control"] = reward_control
        state.info["step"] += 1

        return state.replace(pipeline_state=data,
                             obs=obs,
                             reward=reward)

    # internal function to compute the observation
    def _compute_obs(self, data):
        """
        Compute the observation from the physics state.

        Args:
            data: brax.physics.base.State object
                  The physics state of the environment.
        """

        # extract the relevant information from the data
        pos = data.qpos[0]
        vel = data.qvel[0]
        theta = data.qpos[1]
        theta_dot = data.qvel[1]

        # compute the observation
        obs = jnp.array([pos,            # cart position
                         jnp.cos(theta), # normalized angle
                         jnp.sin(theta), # normalized angle
                         vel,            # cart velocity
                         theta_dot])     # pole angular velocity
        
        return obs
    
    @property
    def observation_size(self):
        """Returns the size of the observation space."""
        return 5
    
    @property
    def action_size(self):
        """Returns the size of the action space."""
        return 1


# register the environment
envs.register_environment("cart_pole", CartPoleEnv)