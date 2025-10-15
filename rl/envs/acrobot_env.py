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
class AcrobotConfig:
    """Config dataclass for acrobot."""

    # model path (NOTE: relative the script that calls this class)
    model_path: str = "./models/acrobot.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 2

    # reward function coefficients
    reward_waist_height: float = 10.0
    reward_tip_height: float = 0.1
    reward_angle_vel: float = 0.01
    reward_control: float = 1e-4

    # Ranges for sampling initial conditions
    lb_angle_pos: float = -jnp.pi
    ub_angle_pos: float =  jnp.pi 
    lb_angle_vel: float = -6.0
    ub_angle_vel: float =  6.0


# environment class
class AcrobotEnv(PipelineEnv):
    """
    Environment for training a acrobot swingup task.

    States: x = (theta1, theta2, theta1_dot, theta2_dot), shape=(4,)
    Observations: o = (cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot), shape=(6,)
    Actions: a = tau, the torques applied to the joints, shape=(1,)
    """

    # initialize the environment
    def __init__(self, config: AcrobotConfig = AcrobotConfig()):

        # robot name
        self.robot_name = "acrobot"

        # environment name
        self.env_name = "acrobot"

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
        print(f"Initialized AcrobotEnv with model [{self.config.model_path}].")

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
        qpos_lb = jnp.array([self.config.lb_angle_pos, self.config.lb_angle_pos])
        qpos_ub = jnp.array([self.config.ub_angle_pos, self.config.ub_angle_pos])
        qvel_lb = jnp.array([self.config.lb_angle_vel, self.config.lb_angle_vel])
        qvel_ub = jnp.array([self.config.ub_angle_vel, self.config.ub_angle_vel])

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
        metrics = {"reward_angle_pos": jnp.array(0.0),
                   "reward_angle_vel": jnp.array(0.0),
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
        theta1 = data.qpos[0]         # angle at the wrist
        theta2 = data.qpos[1]         # angle at the hip
        cos_theta1 = jnp.cos(theta1)           # normalized height contribution from first angle
        cos_thetas = jnp.cos(theta1 + theta2)  # normalized height contribution from second angle
        theta1_dot = data.qvel[0]     # angular velocity at the wrist
        theta2_dot = data.qvel[1]     # angular velocity at the hip
        tau = data.ctrl               # waist torque

        # compute error terms
        y_waist = cos_theta1          # normalized height of the first link
        y_tip = cos_thetas            # normalized height of the tip
        angle_vel_err = jnp.square(theta1_dot).sum() + jnp.square(theta2_dot).sum()
        control_err = jnp.square(tau).sum()

        # rewrad for upright
        reward_pos_waist = self.config.reward_waist_height * y_waist
        reward_pos_tip = self.config.reward_tip_height * y_tip

        # compute the rewards
        reward_angle_pos = reward_pos_waist + reward_pos_tip
        reward_angle_vel = -self.config.reward_angle_vel * angle_vel_err
        reward_control  = -self.config.reward_control * control_err

        # compute the total reward
        reward = reward_angle_pos + reward_angle_vel + reward_control
        
        # update the metrics and info dictionaries
        state.metrics["reward_angle_pos"] = reward_angle_pos
        state.metrics["reward_angle_vel"] = reward_angle_vel
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
        theta1 = data.qpos[0]
        theta2 = data.qpos[1]
        theta1_dot = data.qvel[0]
        theta2_dot = data.qvel[1]

        # compute the observation
        obs = jnp.array([jnp.cos(theta1), 
                         jnp.sin(theta1), 
                         jnp.cos(theta2), 
                         jnp.sin(theta2),
                         theta1_dot, 
                         theta2_dot])

        return obs
    
    @property
    def observation_size(self):
        """Returns the size of the observation space."""
        return 6

    @property
    def action_size(self):
        """Returns the size of the action space."""
        return 1


# register the environment
envs.register_environment("acrobot", AcrobotEnv)