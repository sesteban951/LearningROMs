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
class BipedConfig:
    """Config dataclass for biped."""

    # model path (NOTE: relative the script that calls this class)
    model_path: str = "./models/biped.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 2

    # Reward function coefficients
    reward_base_height: float = 2.0
    reward_base_orient: float = 2.0
    reward_joint_pos: float = 2.0
    reward_base_vel: float = 0.1
    reward_joint_vel: float = 0.1
    reward_control: float = 1e-4

    # Ranges for sampling initial conditions
    lb_hip_joint_pos: float = -jnp.pi  # hip joint pos limits
    ub_hip_joint_pos: float =  jnp.pi
    lb_knee_joint_pos: float = -2.4    # knee joint pos limits
    ub_knee_joint_pos: float =  0.0
    lb_base_theta_pos: float = -jnp.pi / 2.0  # base theta pos limits
    ub_base_theta_pos: float =  jnp.pi / 2.0

    lb_joint_vel: float = -3.0         # joint vel limits
    ub_joint_vel: float =  3.0
    lb_base_cart_vel: float = -2.5     # base cart vel limits
    ub_base_cart_vel: float =  2.5
    lb_base_theta_vel: float = -3.0    # base theta vel limits
    ub_base_theta_vel: float =  3.0


# environment class
class BipedEnv(PipelineEnv):
    """
    Environment for training a planar biped walking task.
    Close to: https://gymnasium.farama.org/environments/mujoco/walker2d/

    States: x = (base_pos, base_theta, join_pos, base_vel, base_theta_dot, joint_vel), shape=(14,)
    Actions: a = (hip_left, knee_left, hip_right, knee_right), the torques applied to the joints, shape=(4,)
    """

    # initialize the environment
    def __init__(self, config: BipedConfig = BipedConfig()):

        # robot name
        self.robot_name = "biped"

        # load the config
        self.config = config

        # create the brax system
        # TODO: eventually refactor to use MJX fully instead since BRAX is now moving away from MJCF
        # see https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
        mj_model = mujoco.MjModel.from_xml_path(self.config.model_path)
        sys = mjcf.load_model(mj_model)

        # get default keyframe
        self.qpos_default = jnp.array(mj_model.key_qpos[0])
        self.qpos_stand = jnp.array(mj_model.key_qpos[1])

        # insantiate the parent class
        super().__init__(
            sys=sys,                                             # brax system defining the kinematic tree and other properties
            backend="mjx",                                       # defining the physics pipeline
            n_frames=self.config.physics_steps_per_control_step  # number of times to step the physics pipeline per control step
                                                                 # for each environment step
        )
        # n_frames: number of sim steps per control step, dt = n_frames * xml_dt

        # print message
        print(f"Initialized BipedEnv with model [{self.config.model_path}].")

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

        # sample the generalized position
        q_base_theta_lb = jnp.array([self.config.lb_base_theta_pos])
        q_base_theta_ub = jnp.array([self.config.ub_base_theta_pos])
        q_joint_lb = jnp.array([self.config.lb_hip_joint_pos, self.config.lb_knee_joint_pos,
                                self.config.lb_hip_joint_pos, self.config.lb_knee_joint_pos])
        q_joint_ub = jnp.array([self.config.ub_hip_joint_pos, self.config.ub_knee_joint_pos,
                                self.config.ub_hip_joint_pos, self.config.ub_knee_joint_pos])
        qpos_no_base_cart_lb = jnp.concatenate([q_base_theta_lb, q_joint_lb])
        qpos_no_base_cart_ub = jnp.concatenate([q_base_theta_ub, q_joint_ub])
        
        # sample the generalized velocity
        v_base_lb = jnp.array([self.config.lb_base_cart_vel, self.config.lb_base_cart_vel, self.config.lb_base_theta_vel])
        v_base_ub = jnp.array([self.config.ub_base_cart_vel, self.config.ub_base_cart_vel, self.config.ub_base_theta_vel])
        v_joint_lb = jnp.array([self.config.lb_joint_vel, self.config.lb_joint_vel,
                                self.config.lb_joint_vel, self.config.lb_joint_vel])
        v_joint_ub = jnp.array([self.config.ub_joint_vel, self.config.ub_joint_vel,
                                self.config.ub_joint_vel, self.config.ub_joint_vel])
        qvel_lb = jnp.concatenate([v_base_lb, v_joint_lb])
        qvel_ub = jnp.concatenate([v_base_ub, v_joint_ub])

        # sample the initial state
        qpos_base_cart = self.qpos_default[0:2]  # base x position fixed at default
        qpos_no_base_cart = jax.random.uniform(rng1, (5,), minval=qpos_no_base_cart_lb, maxval=qpos_no_base_cart_ub)
        qvel = jax.random.uniform(rng2, (7,), minval=qvel_lb, maxval=qvel_ub)
        qpos = jnp.concatenate([qpos_base_cart, qpos_no_base_cart])

        # reset the physics state
        data = self.pipeline_init(qpos, qvel)

        # reset the observation
        obs = self._compute_obs(data)

        # reset reward
        reward, done = jnp.zeros(2)

        # reset the metrics
        metrics = {"reward_base_height": jnp.array(0.0),
                   "reward_base_orient": jnp.array(0.0),
                   "reward_joint_pos": jnp.array(0.0),
                   "reward_base_vel": jnp.array(0.0),
                   "reward_joint_vel": jnp.array(0.0),
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
        base_height = data.qpos[1]
        base_theta = data.qpos[2]
        joint_pos = data.qpos[3:]
        base_vel = data.qvel[0:3]
        joint_vel = data.qvel[3:]
        tau = data.ctrl

        base_cos_theta = jnp.cos(base_theta)
        base_sin_theta = jnp.sin(base_theta)
        base_theta_vec = jnp.array([base_cos_theta - 1.0, base_sin_theta]) # want (0, 0)

        # compute error terms
        height_err = jnp.square(base_height - self.qpos_stand[1]).sum()
        orient_err = jnp.square(base_theta_vec).sum()
        joint_pos_err = jnp.square(joint_pos - self.qpos_stand[3:]).sum()
        base_vel_err = jnp.square(base_vel).sum()
        joint_vel_err = jnp.square(joint_vel).sum()
        control_err = jnp.square(tau).sum()

        # compute the reward terms
        reward_base_height = -self.config.reward_base_height * height_err
        reward_base_orient = -self.config.reward_base_orient * orient_err
        reward_joint_pos = -self.config.reward_joint_pos * joint_pos_err
        reward_base_vel = -self.config.reward_base_vel * base_vel_err
        reward_joint_vel = -self.config.reward_joint_vel * joint_vel_err
        reward_control  = -self.config.reward_control * control_err

        # compute the total reward
        reward = (reward_base_height + reward_base_orient + reward_joint_pos +
                  reward_base_vel + reward_joint_vel + reward_control)

        # update the metrics and info dictionaries
        state.metrics["reward_base_height"] = reward_base_height
        state.metrics["reward_base_orient"] = reward_base_orient
        state.metrics["reward_joint_pos"] = reward_joint_pos
        state.metrics["reward_base_vel"] = reward_base_vel
        state.metrics["reward_joint_vel"] = reward_joint_vel
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

        # base height
        base_height = data.qpos[1]
        base_cos_theta = jnp.cos(data.qpos[2])
        base_sin_theta = jnp.sin(data.qpos[2])
        base_pos = jnp.array([base_height, base_cos_theta, base_sin_theta])

        # joint state
        joint_pos = data.qpos[3:]

        # full velocity state
        qvel = data.qvel

        # compute the observation
        obs = jnp.concatenate([base_pos,
                               joint_pos,
                               qvel])

        return obs
    
    @property
    def observation_size(self):
        """Returns the size of the observation space."""
        return 14

    @property
    def action_size(self):
        """Returns the size of the action space."""
        return 4


# register the environment
envs.register_environment("biped", BipedEnv)