# jax imports
from xml.parsers.expat import model
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
    physics_steps_per_control_step: int = 10

    # Reward function coefficients
    reward_base_height: float = 1.0  # base height target
    reward_base_orient: float = 0.25  # base orientation target
    reward_base_vx: float = 0.8      # forward velocity
    reward_base_vz: float = 0.1      # vertical velocity
    reward_base_omega: float = 0.1   # angular velocity
    reward_joint_pos: float = 0.01    # joint position target
    reward_joint_vel: float = 0.01    # joint velocity target
    reward_control: float = 1e-6    # control cost

    # alive reward bonus (if not terminated)
    reward_alive: float = 0.3

    # Ranges for sampling initial conditions
    lb_base_theta_pos: float = -jnp.pi / 4.0  # base theta pos limits
    ub_base_theta_pos: float =  jnp.pi / 4.0
    lb_base_cart_vel: float = -3.0     # base cart vel limits
    ub_base_cart_vel: float =  3.0
    lb_base_theta_vel: float = -3.0    # base theta vel limits
    ub_base_theta_vel: float =  3.0
    
    lb_hip_joint_pos: float = -jnp.pi * 0.8  # hip joint pos limits
    ub_hip_joint_pos: float =  jnp.pi * 0.8
    lb_knee_joint_pos: float = -2.4 * 0.8    # knee joint pos limits
    ub_knee_joint_pos: float =  0.0
    lb_joint_vel: float = -3.0         # joint vel limits
    ub_joint_vel: float =  3.0

    # termination conditions
    min_base_height: float = 0.5      # terminate if falls below this height
    max_abs_pitch_rad: float = 1.5    # terminate if base pitch exceeds this value
    min_steps_before_done: int = 5    # tiny grace period after reset

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
        key_name = "standing"
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        self.qpos_stand = jnp.array(mj_model.key_qpos[key_id])

        # foot touch sensors
        left_foot_touch_snesor_name = "left_foot_touch"
        right_foot_touch_snesor_name = "right_foot_touch"
        left_foot_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, left_foot_touch_snesor_name)
        right_foot_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, right_foot_touch_snesor_name)
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
        obs = self._compute_obs(data)

        # reset reward
        reward, done = jnp.zeros(2)

        # reset the metrics
        metrics = {"reward_base_height": 0.0,
                   "reward_base_orient": 0.0,
                   "reward_base_vx": 0.0,
                   "reward_base_vz": 0.0,
                   "reward_base_omega": 0.0,
                   "reward_joint_pos": 0.0,
                   "reward_joint_vel": 0.0,
                   "reward_control": 0.0
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

        # step the physics
        data = self.pipeline_step(state.pipeline_state, action)

        # update the observations
        obs = self._compute_obs(data)

        # data
        base_height = data.qpos[1]
        base_theta = data.qpos[2]
        joint_pos = data.qpos[3:]
        base_vx = data.qvel[0]
        base_vz = data.qvel[1]
        base_omega = data.qvel[2]
        joint_vel = data.qvel[3:]
        tau = data.ctrl

        # desired values
        base_height_des = 0.83
        base_theta_des = -0.05
        cos_theta_des = jnp.cos(base_theta_des)
        sin_theta_des = jnp.sin(base_theta_des)
        base_vx_des = 0.5
        joint_pos_des = self.qpos_stand[3:]

        # special angle error
        cos_theta = jnp.cos(base_theta)
        sin_theta = jnp.sin(base_theta)
        base_theta_err_vec = jnp.array([cos_theta - cos_theta_des, 
                                        sin_theta - sin_theta_des]) # want (0, 0)

        # compute error terms
        height_err = jnp.square(base_height - base_height_des).sum()
        orient_err = jnp.square(base_theta_err_vec).sum()
        joint_pos_err = jnp.square(joint_pos - joint_pos_des).sum()
        base_vx_err = jnp.square(base_vx - base_vx_des).sum()
        base_vz_err = jnp.square(base_vz).sum()
        base_omega_err = jnp.square(base_omega).sum()
        joint_vel_err = jnp.square(joint_vel).sum()
        control_err = jnp.square(tau).sum()

        # compute the reward terms
        reward_base_height = -self.config.reward_base_height * height_err
        reward_base_orient = -self.config.reward_base_orient * orient_err
        reward_joint_pos = -self.config.reward_joint_pos * joint_pos_err
        reward_base_vx = -self.config.reward_base_vx * base_vx_err
        reward_base_vz = -self.config.reward_base_vz * base_vz_err
        reward_base_omega = -self.config.reward_base_omega * base_omega_err
        reward_joint_vel = -self.config.reward_joint_vel * joint_vel_err
        reward_control  = -self.config.reward_control * control_err

        # termination conditions
        below_height = base_height < self.config.min_base_height
        tilted_over = jnp.abs(base_theta) > self.config.max_abs_pitch_rad

        # small grace window to avoid instant termination on the first few frames after reset
        after_grace_period = state.info["step"] >= self.config.min_steps_before_done

        # determine if should terminate
        done = jnp.where((below_height | tilted_over) & after_grace_period, 1.0, 0.0)

        # if not terminated, give a small alive bonus
        reward_alive = self.config.reward_alive * (1.0 - done)

        # compute the total reward
        reward = (reward_base_height + reward_base_orient + reward_joint_pos +
                  reward_base_vx + reward_base_vz + reward_base_omega + reward_joint_vel + 
                  reward_control + reward_alive)

        # update the metrics and info dictionaries
        state.metrics["reward_base_height"] = reward_base_height
        state.metrics["reward_base_orient"] = reward_base_orient
        state.metrics["reward_base_vx"] = reward_base_vx
        state.metrics["reward_base_vz"] = reward_base_vz
        state.metrics["reward_base_omega"] = reward_base_omega
        state.metrics["reward_joint_pos"] = reward_joint_pos
        state.metrics["reward_joint_vel"] = reward_joint_vel
        state.metrics["reward_control"] = reward_control
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

        # --- foot contact flags from sensors ---
        sd = data.sensordata  # mjx exposes sensordata
        l_touch = sd[self.left_adr : self.left_adr + self.left_dim][0]
        r_touch = sd[self.right_adr: self.right_adr + self.right_dim][0]

        # convert to {0,1}; clamp to be JAX-friendly
        contact_eps = 1e-3
        left_in_contact  = jnp.where(l_touch > contact_eps, 1.0, 0.0)
        right_in_contact = jnp.where(r_touch > contact_eps, 1.0, 0.0)
        foot_contacts = jnp.array([left_in_contact, right_in_contact])

        # compute the observation
        obs = jnp.concatenate([base_pos,       # height, cos(theta), sin(theta)
                               joint_pos,      # joint positions
                               qvel,           # full velocity state
                               foot_contacts]) # foot contact flags    

        return obs
    
    @property
    def observation_size(self):
        """Returns the size of the observation space."""
        return 16

    @property
    def action_size(self):
        """Returns the size of the action space."""
        return 4


# register the environment
envs.register_environment("biped", BipedEnv)