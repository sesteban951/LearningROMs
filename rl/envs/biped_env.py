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
    physics_steps_per_control_step: int = 4

    # Reward function coefficients
    reward_base_pos_z: float = 1.0    # target base height
    reward_base_vel_x: float = 1.0     # forward velocity target
    reward_base_vel_z: float = 0.01    # target base height
    reward_base_ang_pos: float = 2.0   # torso orientation target
    reward_base_ang_vel: float = 0.15  # torso angular velocity target
    reward_joint_pos: float = 0.1      # joint position 
    reward_joint_vel: float = 1e-3     # joint velocity 
    reward_joint_acc: float = 2.5e-7   # joint acceleration
    reward_action_rate: float = 0.005   # control cost
    reward_foot_contact: float = 0.1       # foot contact reward
    reward_foot_slip: float = 0.1          # foot slip penalty
    reward_alive: float = 1.0         # alive reward bonus (if not terminated)
    cost_termination: float = 100.0      # cost at termination (if falls)

    # desired values
    base_pos_z_des: float = 0.82  # desired center of mass height
    base_vel_x_des: float = 0.5   # desired forward velocity
    theta_des: float = -0.1   # desired torso lean angle
    hip_des: float = 0.22     # desired hip joint angle

    # phase parameters (inspired by unitree_rl_gym)
    T_phase = 0.8           # total period of the gait cycle
    phase_offset = 0.5      # percent offset between left and right legs
    phase_threshold = 0.55  # percent threshold that defines stance. leg_phase < 0.55 means

    # termination conditions
    min_base_height: float = 0.25     # terminate if falls below this height
    max_base_pitch: float = 1.5      # terminate if base pitch exceeds this value
    min_steps_before_done: int = 5   # tiny grace period after reset

    # Ranges for sampling initial conditions
    lb_base_height: float = 0.75  # base height pos limits
    ub_base_height: float = 0.85
    lb_base_theta_pos: float = -jnp.pi  # base theta pos limits
    ub_base_theta_pos: float =  jnp.pi
    lb_hip_joint_pos: float =  -jnp.pi  # hip joint pos limits
    ub_hip_joint_pos: float =   jnp.pi
    lb_knee_joint_pos: float = -2.4   # knee joint pos limits
    ub_knee_joint_pos: float =  0.0
    
    lb_base_cart_vel_x: float = -0.5     # base cart vel limits
    ub_base_cart_vel_x: float =  0.5
    lb_base_cart_vel_z: float = -0.1     # base cart vel z limits
    ub_base_cart_vel_z: float =  0.1
    lb_base_theta_vel: float = -3.14    # base theta vel limits
    ub_base_theta_vel: float =  3.14
    lb_joint_vel: float = -3.14     # joint vel limits
    ub_joint_vel: float =  3.14

    # position action scale, action = (q_des - q_nom) / pos_action_scale
    pos_action_scale: float = 0.25 

    # PD gains
    kp_hip: float = 200.0
    kd_hip: float = 5.0
    kp_knee: float = 200.0
    kd_knee: float = 5.0


# environment class
class BipedEnv(PipelineEnv):
    """
    Environment for training a planar biped walking task.
    NOTE: Try to follow some stuff from Mujoco Playground g1/joystick.py example
          https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/g1/joystick.py

    States: x = (base_pos, base_theta, joint_pos, base_vel, base_theta_dot, joint_vel), shape=(14,)
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
        self.sys = sys

        # get default keyframe
        key_name = "standing"
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        self.qpos_stand = jnp.array(mj_model.key_qpos[key_id])

        # build the limits vectors
        self.q_pos_lb = jnp.array([-0.01, self.config.lb_base_height,
                                   self.config.lb_base_theta_pos,
                                   self.config.lb_hip_joint_pos, self.config.lb_knee_joint_pos,
                                   self.config.lb_hip_joint_pos, self.config.lb_knee_joint_pos])
        self.q_pos_ub = jnp.array([ 0.01, self.config.ub_base_height,
                                   self.config.ub_base_theta_pos,
                                   self.config.ub_hip_joint_pos, self.config.ub_knee_joint_pos,
                                   self.config.ub_hip_joint_pos, self.config.ub_knee_joint_pos])
        self.q_vel_lb = jnp.array([self.config.lb_base_cart_vel_x, self.config.lb_base_cart_vel_z,
                                   self.config.lb_base_theta_vel,
                                   self.config.lb_joint_vel, self.config.lb_joint_vel,
                                   self.config.lb_joint_vel, self.config.lb_joint_vel])
        self.q_vel_ub = jnp.array([self.config.ub_base_cart_vel_x, self.config.ub_base_cart_vel_z,
                                   self.config.ub_base_theta_vel,
                                   self.config.ub_joint_vel, self.config.ub_joint_vel,
                                   self.config.ub_joint_vel, self.config.ub_joint_vel])
        
        # nominal joint positions (standing pose)
        self.q_joints_nom = self.qpos_stand[3:]  # joint positions only

        # joint position bounds
        self.q_joints_pos_lb = self.q_pos_lb[3:]
        self.q_joints_pos_ub = self.q_pos_ub[3:]

        # build the PD gains vectors
        self.kp = jnp.array([self.config.kp_hip, self.config.kp_knee, 
                             self.config.kp_hip, self.config.kp_knee])
        self.kd = jnp.array([self.config.kd_hip, self.config.kd_knee, 
                             self.config.kd_hip, self.config.kd_knee])
        
        # get the contorl limits
        self.ctrl_min = jnp.array(mj_model.actuator_ctrlrange[:, 0])  # shape (4,)
        self.ctrl_max = jnp.array(mj_model.actuator_ctrlrange[:, 1])  # shape (4,)

        # build gear ratio vector
        self.gear = jnp.array(mj_model.actuator_gear[:, 0])  # shape (4,)

        # site ids
        self.left_site_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_site")
        self.right_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_site")

        # foot touch sensors
        left_foot_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_touch")
        right_foot_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_touch")
        self.left_adr = mj_model.sensor_adr[left_foot_sensor_id]    # dim should be 1
        self.right_adr = mj_model.sensor_adr[right_foot_sensor_id]  # dim should be 1

        # foot linear velocity sensors
        lvel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_linvel")
        rvel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_linvel")
        self.lvel_adr = mj_model.sensor_adr[lvel_id]    # dim should be 3
        self.rvel_adr = mj_model.sensor_adr[rvel_id]    # dim should be 3

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

    ######################################### RESET ###############################################

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

        # uniform sample within the limits
        # qpos = jax.random.uniform(rng1, (7,), minval=self.q_pos_lb, maxval=self.q_pos_ub)
        # qvel = jax.random.uniform(rng2, (7,), minval=self.q_vel_lb, maxval=self.q_vel_ub)

        # sample around the standing pose
        qpos = self.qpos_stand + 0.25*jax.random.normal(rng1, self.qpos_stand.shape)
        qvel = 0.05*jax.random.normal(rng2, self.qpos_stand.shape)

        # clip to be within the limits
        qpos = jnp.clip(qpos, self.q_pos_lb, self.q_pos_ub)
        qvel = jnp.clip(qvel, self.q_vel_lb, self.q_vel_ub)

        # reset the physics state
        data = self.pipeline_init(qpos, qvel)

        # MODIFIED: initialize with zero previous action
        prev_action = jnp.zeros(self.action_size)

        # reset the observation (now includes previous action)
        obs = self._compute_obs(data, prev_action)

        # reset reward
        reward, done = jnp.zeros(2)

        # reset the metrics
        metrics = {"reward_base_pos_z": 0.0,
                   "reward_base_vel_x": 0.0,
                   "reward_base_vel_z": 0.0,
                   "reward_base_ang_pos": 0.0,
                   "reward_base_ang_vel": 0.0,
                   "reward_joint_pos": 0.0,
                   "reward_joint_vel": 0.0,
                   "reward_joint_acc": 0.0,
                   "reward_action_rate": 0.0,
                   "reward_foot_contact": 0.0,
                   "reward_foot_slip": 0.0,
                   "reward_alive": 0.0,
                   "cost_termination": 0.0
                   }

        # state info - MODIFIED: store previous action
        info = {"rng": rng,
                "step": 0,
                "prev_action": prev_action,
                }
        
        return State(pipeline_state=data,
                     obs=obs,
                     reward=reward,
                     done=done,
                     metrics=metrics,
                     info=info)
    
    ######################################### STEP ###############################################

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

        # MODIFIED: get previous action from state
        prev_action = state.info["prev_action"]

        # Δq -> q_des (absolute target around standing)
        q_des = self._action_to_q_des(action)

        # PD torque control using pre-state before step
        ctrl, _ = self._q_des_to_torque(data0, q_des)

        # take a step in the physics
        data = self.pipeline_step(data0, ctrl)

        # update the observations (now includes current action)
        obs = self._compute_obs(data, prev_action)

        # extract data
        base_pos_z = data.qpos[1]
        base_vel_x = data.qvel[0]
        base_vel_z = data.qvel[1]
        base_ang_pos = data.qpos[2]
        base_ang_vel = data.qvel[2]
        joint_pos = data.qpos[3:]
        joint_vel = data.qvel[3:]
        joint_acc = data.qacc[3:]

        # special angle error
        cos_theta = jnp.cos(base_ang_pos)
        sin_theta = jnp.sin(base_ang_pos)
        cos_theta_des = jnp.cos(self.config.theta_des)
        sin_theta_des = jnp.sin(self.config.theta_des)
        base_ang_pos_vec = jnp.array([cos_theta - cos_theta_des,
                                      sin_theta - sin_theta_des])
        
        # special foot contact error
        _, _, left_phase, right_phase = self._compute_phase(data.time)
        left_should_be_in_stance = left_phase < self.config.phase_threshold
        right_should_be_in_stance = right_phase < self.config.phase_threshold
        left_in_contact, right_in_contact = self._compute_foot_contact(data)
        left_match  = jnp.logical_not(jnp.logical_xor(left_in_contact,  left_should_be_in_stance))
        right_match = jnp.logical_not(jnp.logical_xor(right_in_contact, right_should_be_in_stance))

        # foot slip penalty
        foot_slip = self._compute_feet_slip(data)

        # compute errors
        base_pos_z_err = jnp.square(base_pos_z - self.config.base_pos_z_des).sum()
        base_vel_x_err = jnp.square(base_vel_x - self.config.base_vel_x_des).sum()
        base_vel_z_err = jnp.square(base_vel_z).sum()
        base_ang_pos_err = jnp.square(base_ang_pos_vec).sum()
        base_ang_vel_err = jnp.square(base_ang_vel).sum()
        joint_pos_err = jnp.square(joint_pos - self.q_joints_nom).sum()
        joint_vel_err = jnp.square(joint_vel).sum()
        joint_acc_err = jnp.square(joint_acc).sum()

        # smooth Δq, action rate error (difference from previous action)
        action_rate_err = jnp.square(action - prev_action).sum()

        # compute the reward terms
        reward_base_pos_z = -self.config.reward_base_pos_z * base_pos_z_err
        reward_base_vel_x = -self.config.reward_base_vel_x * base_vel_x_err
        reward_base_vel_z = -self.config.reward_base_vel_z * base_vel_z_err
        reward_base_ang_pos = -self.config.reward_base_ang_pos * base_ang_pos_err
        reward_base_ang_vel = -self.config.reward_base_ang_vel * base_ang_vel_err
        reward_joint_pos = -self.config.reward_joint_pos * joint_pos_err
        reward_joint_vel = -self.config.reward_joint_vel * joint_vel_err
        reward_joint_acc = -self.config.reward_joint_acc * joint_acc_err
        reward_action_rate = -self.config.reward_action_rate * action_rate_err
        reward_foot_contact = (left_match.astype(jnp.float32) + 
                               right_match.astype(jnp.float32)) * self.config.reward_foot_contact / 2.0
        reward_foot_slip = -self.config.reward_foot_slip * foot_slip

        # termination conditions
        below_height = base_pos_z < self.config.min_base_height
        tilted_over = jnp.abs(base_ang_pos) > self.config.max_base_pitch

        # small grace window to avoid instant termination on the first few frames after reset
        after_grace_period = state.info["step"] >= self.config.min_steps_before_done

        # determine if should terminate
        done = jnp.where((below_height | tilted_over) & after_grace_period, 1.0, 0.0)
        cost_termination = -self.config.cost_termination * done

        # if not terminated, give a small alive bonus
        reward_alive = self.config.reward_alive * (1.0 - done)

        # compute the total reward
        reward = (reward_base_pos_z +
                  reward_base_vel_x   + reward_base_vel_z +
                  reward_base_ang_pos + reward_base_ang_vel +
                  reward_joint_pos + reward_joint_vel + reward_joint_acc +
                  reward_action_rate  +  
                  reward_foot_contact + reward_foot_slip +
                  reward_alive + cost_termination)

        # update the metrics and info dictionaries
        state.metrics["reward_base_pos_z"]   = reward_base_pos_z
        state.metrics["reward_base_vel_x"]   = reward_base_vel_x
        state.metrics["reward_base_vel_z"]   = reward_base_vel_z
        state.metrics["reward_base_ang_pos"] = reward_base_ang_pos
        state.metrics["reward_base_ang_vel"] = reward_base_ang_vel
        state.metrics["reward_joint_pos"]    = reward_joint_pos
        state.metrics["reward_joint_vel"]    = reward_joint_vel
        state.metrics["reward_joint_acc"]    = reward_joint_acc
        state.metrics["reward_action_rate"]  = reward_action_rate
        state.metrics["reward_foot_contact"] = reward_foot_contact
        state.metrics["reward_foot_slip"]    = reward_foot_slip
        state.metrics["reward_alive"]        = reward_alive
        state.metrics["cost_termination"]    = cost_termination
        
        # update the state info
        state.info["step"] += 1
        state.info["prev_action"] = action  # Store current action for next step

        return state.replace(pipeline_state=data,
                             obs=obs,
                             reward=reward,
                             done=done)
    
    ####################################### OBSERVATION #############################################

    # internal function to compute the observation
    def _compute_obs(self, data, prev_action):
        """
        Compute the observation from the physics state.

        Args:
            data: brax.physics.base.State object
                  The physics state of the environment.
            action: jax.Array
                    The action taken by the agent.
        Returns:
            obs: jax.Array
                 The observation of the environment.
        """

        # base positions
        base_height = data.qpos[1]
        base_theta = data.qpos[2]
        base_cos_theta = jnp.cos(base_theta)
        base_sin_theta = jnp.sin(base_theta)
        base_pos = jnp.array([base_height, base_cos_theta, base_sin_theta]) # shape (3,)
        
        # joint positions
        joint_pos = data.qpos[3:]                        # shape (4,)
        joint_pos_rel = (joint_pos - self.q_joints_nom)  # shape (4,)

        # full velocity state
        position = jnp.concatenate([base_pos, joint_pos_rel])                   # shape (7,)
        velocity = data.qvel   # shape (7,)

        # phase variable
        sin_phase, cos_phase, _, _ = self._compute_phase(data.time)

        # compute the observation
        obs = jnp.concatenate([position,        # base pos + joint pos
                               velocity,        # full gen velocity
                               prev_action,     # previous action
                               sin_phase,       # phase variable
                               cos_phase])      # phase variable

        return obs
    
    ########################################## UTILS ################################################

    # helper to compute if the feet are in contact with the ground
    def _compute_foot_contact(self, data):
        """
        Compute the contact state of the left and right feet.

        Args:
            data: brax.physics.base.State object
                  The physics state of the environment.
        Returns:
            left_in_contact: bool
                             Whether the left foot is in contact with the ground.
            right_in_contact: bool
                              Whether the right foot is in contact with the ground.
        """
        # foot contact flags
        sd = data.sensordata  # mjx exposes sensordata

        # return left_in_contact, right_in_contact
        l_val = sd[self.left_adr ]   # should be dim 1
        r_val = sd[self.right_adr]   # should be dim 1
        left_in_contact  = l_val > 1e-3
        right_in_contact = r_val > 1e-3

        return left_in_contact, right_in_contact
    
    # compute if there is foot slip 
    def _compute_feet_slip(self, data):
        """
        Compute foot slipping while in contact
        
        Args:
            data: brax.physics.base.State object, The physics state of the environment before the step.
        Returns:
            slip_cost: jax.Array, The computed slip cost, shape ()
        """

        # contact booleans from your touch sensors
        left_c, right_c = self._compute_foot_contact(data)  # 0/1 scalars

        sd = data.sensordata
        vL = sd[self.lvel_adr : self.lvel_adr + 3]  # (3,)
        vR = sd[self.rvel_adr : self.rvel_adr + 3]  # (3,)

        # only consider x direction slip
        vL_x = jnp.abs(vL[0])
        vR_x = jnp.abs(vR[0])

        # penalize slip only when in contact
        slip_cost = vL_x * left_c + vR_x * right_c

        return slip_cost
               

    # helper function to compute the phase
    def _compute_phase(self, t):
        """
        Compute the phase for the gait cycle.
        Args:
            t: float
               The current simulation time.
        Returns:
            sin_phase: jax.Array
                       The sine of the phase variable, shape (1,)
            cos_phase: jax.Array
                       The cosine of the phase variable, shape (1,)
            left_phase: jax.Array
                        The phase of the left leg, shape (1,)
            right_phase: jax.Array
                         The phase of the right leg, shape (1,)
        """

        # compute phase variables
        phase = jnp.mod(t, self.config.T_phase) / self.config.T_phase
        sin_phase = jnp.sin(2.0 * jnp.pi * phase)
        cos_phase = jnp.cos(2.0 * jnp.pi * phase)
        sin_phase = sin_phase[..., None]  # add last axis → (1, 1)
        cos_phase = cos_phase[..., None]  # add last axis → (1, 1)

        # compute the phase for each leg
        left_phase = phase
        right_phase = jnp.mod(phase + self.config.phase_offset, 1.0)

        return sin_phase, cos_phase, left_phase, right_phase
    
    # convert normalized position action to joint position target
    def _action_to_q_des(self, action):
        """
        Map normalized Δq action to absolute joint target:
            q_des = clip(q_nom + pos_action_scale * clip(action, -1, 1), [q_low, q_high])

        Args:
            action: jax.Array, 
        Returns:
            q_des: jax.Array, desired joint positions, shape (4,)
        """
        # clip the action to be within [-1, 1]
        a = jnp.clip(action, -1.0, 1.0)

        # compute the desired joint position
        q_des = self.q_joints_nom + self.config.pos_action_scale * a

        # clip to the feasible joint limits
        q_des = jnp.clip(q_des, self.q_joints_pos_lb, self.q_joints_pos_ub)

        return q_des
    
    # compute torque from desired position (not used)
    def _q_des_to_torque(self, data0, q_des):
        """
        Convert a joint position target to a torque using PD control.

        Args:
            data0: brax.physics.base.State object, The physics state of the environment before the step.
            q_des: jax.Array, The desired joint positions, shape (4,).
        Returns:
            u: jax.Array, The actuator controls to be applied, shape (4,).
            tau: jax.Array, The computed torques before mapping to actuator controls, shape (4,).
        """
        # get current joint state
        q_pos = data0.qpos[3:]
        q_vel = data0.qvel[3:]

        # PD control
        tau = self.kp * (q_des - q_pos) + self.kd * (0.0 - q_vel)
        
        # Map torque -> actuator control via gear and clip to ctrlrange
        # (gear == 75 in your XML; guard against zeros just in case)
        u = tau / self.gear                            # normalize the torque
        u = jnp.clip(u, self.ctrl_min, self.ctrl_max)  # clip to control range to [-1, 1]

        return u, tau
    
    ######################################## PROPERTIES ##############################################

    @property
    def observation_size(self):
        """Returns the size of the observation space."""
        return 20

    @property
    def action_size(self):
        """Returns the size of the action space."""
        return 4


# register the environment
envs.register_environment("biped", BipedEnv)