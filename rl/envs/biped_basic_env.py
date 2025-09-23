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
class BipedBasicConfig:
    """Config dataclass for biped."""

    # model path (NOTE: relative the script that calls this class)
    model_path: str = "./models/biped.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 4

    # Reward function coefficients
    reward_com_height: float = 2.0   # center of mass height target
    reward_orientation: float = 0.5  # torso orientation target
    reward_joint_pos: float = 0.05    # joint position target
    # reward_joint_vel: float = 0.01    # joint position target
    reward_forward: float = 2.0      # forward velocity target
    reward_control: float = 1e-4     # control cost
    reward_alive: float = 1.0        # alive reward bonus (if not terminated)

    # TODO: rewrads to add:
    # - foot slip (lateral and longitudinal)
    # - foot alternating via phase
    # - should I go back to base height?
    # - no slip on stance foot. Use complementary constraint-like reward


    # desired values
    com_des: float = 0.8    # desired center of mass height
    vel_des: float = 1.0     # desired forward velocity
    theta_des: float = -0.1  # desired torso lean angle

    # termination conditions
    min_com_height: float = 0.5      # terminate if falls below this height
    max_base_pitch: float = 1.5      # terminate if base pitch exceeds this value
    min_steps_before_done: int = 5   # tiny grace period after reset

    # Ranges for sampling initial conditions
    lb_base_theta_pos: float = -jnp.pi / 3.0  # base theta pos limits
    ub_base_theta_pos: float =  jnp.pi / 3.0
    lb_base_cart_vel: float = -1.0     # base cart vel limits
    ub_base_cart_vel: float =  1.0
    lb_base_theta_vel: float = -1.0    # base theta vel limits
    ub_base_theta_vel: float =  1.0
    lb_hip_joint_pos: float =  (-jnp.pi / 2.0) * 0.8  # hip joint pos limits
    ub_hip_joint_pos: float =  ( jnp.pi / 2.0) * 0.8
    lb_knee_joint_pos: float = (-2.4) * 0.8    # knee joint pos limits
    ub_knee_joint_pos: float =  0.0
    lb_joint_vel: float = -1.0         # joint vel limits
    ub_joint_vel: float =  1.0

# environment class
class BipedBasicEnv(PipelineEnv):
    """
    Environment for training a planar biped walking task.
    Close to: https://gymnasium.farama.org/environments/mujoco/walker2d/

    States: x = (base_pos, base_theta, joint_pos, base_vel, base_theta_dot, joint_vel), shape=(14,)
    Actions: a = (hip_left, knee_left, hip_right, knee_right), the torques applied to the joints, shape=(4,)
    """

    # initialize the environment
    def __init__(self, config: BipedBasicConfig = BipedBasicConfig()):

        # robot name
        self.robot_name = "biped"

        # load the config
        self.config = config

        # create the brax system
        # TODO: eventually refactor to use MJX fully instead since BRAX is now moving away from MJCF
        # see https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
        mj_model = mujoco.MjModel.from_xml_path(self.config.model_path)
        sys = mjcf.load_model(mj_model)
        self.sys = sys

        # get default keyframe
        key_name = "standing"
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        self.qpos_stand = jnp.array(mj_model.key_qpos[key_id])
        self.q_joints_stand = self.qpos_stand[3:]  # joint positions only

        # foot touch sensors
        left_foot_touch_sensor_name = "left_foot_touch"
        right_foot_touch_sensor_name = "right_foot_touch"
        left_foot_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, left_foot_touch_sensor_name)
        right_foot_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, right_foot_touch_sensor_name)
        self.left_adr = mj_model.sensor_adr[left_foot_sensor_id]
        self.right_adr = mj_model.sensor_adr[right_foot_sensor_id]
        self.left_dim = mj_model.sensor_dim[left_foot_sensor_id]
        self.right_dim = mj_model.sensor_dim[right_foot_sensor_id]

        # instantiate the parent class
        super().__init__(
            sys=sys,                                             # brax system defining the kinematic tree and other properties
            backend="mjx",                                       # defining the physics pipeline
            n_frames=self.config.physics_steps_per_control_step  # number of times to step the physics pipeline per control step
                                                                 # for each environment step
        )
        # n_frames: number of sim steps per control step, dt = n_frames * xml_dt

        # print message
        print(f"Initialized BipedBasicEnv with model [{self.config.model_path}].")

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

        # sample generalized position
        q_pos_lb = jnp.array([-0.01, 0.7, 
                              self.config.lb_base_theta_pos,
                              self.config.lb_hip_joint_pos,
                              self.config.lb_knee_joint_pos,
                              self.config.lb_hip_joint_pos,
                              self.config.lb_knee_joint_pos])
        q_pos_ub = jnp.array([0.01, 0.9,
                              self.config.ub_base_theta_pos,
                              self.config.ub_hip_joint_pos,
                              self.config.ub_knee_joint_pos,
                              self.config.ub_hip_joint_pos,
                              self.config.ub_knee_joint_pos])
        qpos = jax.random.uniform(rng1, (7,), minval=q_pos_lb, maxval=q_pos_ub)

        # sample the generalized velocity
        q_vel_lb = jnp.array([self.config.lb_base_cart_vel, self.config.lb_base_cart_vel, 
                              self.config.lb_base_theta_vel,
                              self.config.lb_joint_vel,
                              self.config.lb_joint_vel,
                              self.config.lb_joint_vel,
                              self.config.lb_joint_vel])
        q_vel_ub = jnp.array([self.config.ub_base_cart_vel, self.config.ub_base_cart_vel, 
                              self.config.ub_base_theta_vel,
                              self.config.ub_joint_vel,
                              self.config.ub_joint_vel,
                              self.config.ub_joint_vel,
                              self.config.ub_joint_vel])
        qvel = jax.random.uniform(rng2, (7,), minval=q_vel_lb, maxval=q_vel_ub)

        # reset the physics state
        data = self.pipeline_init(qpos, qvel)

        # reset the observation
        # action  = jnp.zeros(self.action_size)  # assume zero initial action
        obs = self._compute_obs(data)

        # reset reward
        reward, done = jnp.zeros(2)

        # reset the metrics
        metrics = {"reward_com_height": 0.0,
                   "reward_forward": 0.0,
                   "reward_orientation": 0.0,
                   "reward_joint_pos": 0.0,
                #    "reward_joint_vel": 0.0,
                   "reward_control": 0.0,
                   "reward_alive": 0.0
                   }

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

        # initial and final pipline state
        data0 = state.pipeline_state
        data  = self.pipeline_step(data0, action)

        # update the observations
        obs = self._compute_obs(data)

        # extract data
        com_pos0 = data0.subtree_com[1]
        com_pos  = data.subtree_com[1]
        com_vel = (com_pos[0] - com_pos0[0]) / self.dt
        base_theta = data.qpos[2]
        cos_theta = jnp.cos(base_theta)
        sin_theta = jnp.sin(base_theta)
        joint_pos = data.qpos[3:]
        # joint_vel = data.qvel[3:]

        # special angle error
        cos_theta_des = jnp.cos(self.config.theta_des)
        sin_theta_des = jnp.sin(self.config.theta_des)
        base_theta_vec = jnp.array([cos_theta - cos_theta_des,
                                    sin_theta - sin_theta_des])

        # compute errors
        com_pos_err  = jnp.abs(com_pos[2] - self.config.com_des)
        com_vel_err = jnp.square(com_vel - self.config.vel_des)
        base_theta_err = jnp.square(base_theta_vec).sum()
        joint_pos_err = jnp.square(joint_pos - self.q_joints_stand).sum()
        # joint_vel_err = jnp.square(joint_vel).sum()
        tau_err = jnp.square(data.ctrl).sum()

        # compute the reward terms
        reward_com_height = -self.config.reward_com_height * com_pos_err
        reward_forward = -self.config.reward_forward * com_vel_err
        reward_orientation = -self.config.reward_orientation * base_theta_err
        reward_joint_pos = -self.config.reward_joint_pos * joint_pos_err
        # reward_joint_vel = -self.config.reward_joint_vel * joint_vel_err
        reward_control = -self.config.reward_control * tau_err

        # termination conditions
        below_height = com_pos[2] < self.config.min_com_height
        tilted_over = jnp.abs(base_theta) > self.config.max_base_pitch

        # small grace window to avoid instant termination on the first few frames after reset
        after_grace_period = state.info["step"] >= self.config.min_steps_before_done

        # determine if should terminate
        done = jnp.where((below_height | tilted_over) & after_grace_period, 1.0, 0.0)

        # if not terminated, give a small alive bonus
        reward_alive = self.config.reward_alive * (1.0 - done)

        # compute the total reward
        reward = (reward_com_height + reward_forward   + reward_orientation +
                  reward_joint_pos  + 
                #   reward_joint_vel +
                  reward_control    + reward_alive)

        # update the metrics and info dictionaries
        state.metrics["reward_com_height"] = reward_com_height
        state.metrics["reward_forward"] = reward_forward
        state.metrics["reward_orientation"] = reward_orientation
        state.metrics["reward_joint_pos"] = reward_joint_pos
        # state.metrics["reward_joint_vel"] = reward_joint_vel
        state.metrics["reward_control"] = reward_control
        state.metrics["reward_alive"] = reward_alive
        state.info["step"] += 1

        return state.replace(pipeline_state=data,
                             obs=obs,
                             reward=reward,
                             done=done)

    # internal function to compute the observation
    def _compute_obs(self, data):
        """
        Compute the observation from the physics state.

        Args:
            data: brax.physics.base.State object
                  The physics state of the environment.
        Returns:
            obs: jax.Array
                 The observation of the environment.
        """

        # positions
        base_height = data.qpos[1]
        base_cos_theta = jnp.cos(data.qpos[2])
        base_sin_theta = jnp.sin(data.qpos[2])
        base_pos = jnp.array([base_height, base_cos_theta, base_sin_theta]) # shape (3,)
        joint_pos = data.qpos[3:]                                           # shape (4,)
        position = jnp.concatenate([base_pos, joint_pos])                   # shape (7,)

        # full velocity state
        velocity = data.qvel   # shape (7,)

        # # com inertia and velocity (skip the first entry which is the world)
        # cinert = data.cinert[1:].ravel()  # shape (nbody x 10)
        # cvel = data.cvel[1:].ravel()      # shape (nbody x 6)

        # generalized forces on the full system 
        # qfrc = data.qfrc_actuator  # shape (nv x 1)

        # TODO: add a phase variable? Should help with feet alternation

        # contact at the foot
        sd = data.sensordata  # mjx exposes sensordata
        l_touch = sd[self.left_adr : self.left_adr + self.left_dim][0]
        r_touch = sd[self.right_adr: self.right_adr + self.right_dim][0]

        # convert to {0,1}; clamp to be JAX-friendly
        contact_eps = 1e-3
        left_in_contact  = jnp.where(l_touch > contact_eps, 1.0, 0.0)
        right_in_contact = jnp.where(r_touch > contact_eps, 1.0, 0.0)
        foot_contacts = jnp.array([left_in_contact, right_in_contact]) # shape (2,)

        # TODO: investigate which observations actually help

        # compute the observation
        obs = jnp.concatenate([position,       # base pos + joint pos
                               velocity,       # full gen velocity
                            #    cinert,         # com inertia
                            #    cvel,           # com velocity
                            #    qfrc,           # generalized forces
                               foot_contacts]) # foot contact flags

        return obs


    @property
    def observation_size(self):
        """Returns the size of the observation space."""
        # return 103
        return 16

    @property
    def action_size(self):
        """Returns the size of the action space."""
        return 4


# register the environment
envs.register_environment("biped_basic", BipedBasicEnv)