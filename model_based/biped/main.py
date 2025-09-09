# standard includes
import numpy as np
import scipy as sp
import time          

# pacakge includes
import mujoco 
import glfw

# custom includes 
from indeces import Biped_IDX
from model_based.biped.utils import InverseKinematics, bezier_curve, Joy

##################################################################################

class Controller:

    def __init__(self, model_file):

        # load the model file
        model = mujoco.MjModel.from_xml_path(model_file)
        self.model = model

        # create indexing object
        self.idx = Biped_IDX()

        # create joystick object
        self.joystick = Joy()

        # create IK object
        self.ik = InverseKinematics(model_file)

        # gains (NOTE: these does not take gear reduction into account)
        self.kp = np.array([250.0, 250.0, 250.0, 250.0])
        self.kd = np.array([5.0, 5.0, 5.0, 5.0])

        # get foot locations
        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_site")
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_site")

        # get the gear ratios
        LH_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_left_motor")
        LK_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_left_motor")
        RH_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_right_motor")
        RK_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_right_motor")
        self.gear_ratio_LH = model.actuator_gear[LH_actuator_id, 0]
        self.gear_ratio_LK = model.actuator_gear[LK_actuator_id, 0]
        self.gear_ratio_RH = model.actuator_gear[RH_actuator_id, 0]
        self.gear_ratio_RK = model.actuator_gear[RK_actuator_id, 0]

        # internal state and time
        self.time = 0.0
        self.q = np.zeros(7)
        self.v = np.zeros(7)
        self.q_base = np.zeros(3)
        self.v_base = np.zeros(3)
        self.q_joints = np.zeros(4)
        self.v_joints = np.zeros(4)

        # Timing variables
        self.t_phase = 0.0
        self.num_steps = 0
        self.first_step = True
        self.T_SSP = 0.45
        self.T_DSP = 0.0
        self.T = self.T_SSP + self.T_DSP

        # frame variables
        self.stance_foot = None
        self.stance_foot_pos_W = None
        self.swing_foot = None
        self.swing_init_pos_W = None

        # ROM variables
        self.p_com_W = np.zeros(3)
        self.v_com_W = np.zeros(3)

        # foot variables
        self.z_apex = 0.04           # maximum foot height achieved during swing
        self.z_base = 0.75           # nominal height of the base from the ground
        self.z_foot_offset = 0.045   # offset of the foot from the ground for foot geom

        # compute LIP parameters
        self.lam = np.sqrt(9.81 / self.z_base)  # natural frequency
        self.A = np.array([[0,           1],    # LIP drift matrix
                           [self.lam**2, 0]])
        
        # create lambda function for hyperbolic trig
        self.coth = lambda x: (np.exp(2 * x) + 1) / (np.exp(2 * x) - 1)
        self.tanh = lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        self.sech = lambda x: 1 / np.cosh(x)

        # Period-1 orbit parameters
        self.sigma_P1 = self.lam * self.coth(0.5 * self.lam * self.T_SSP)

        # LIP foot placement gains (deadbeat, but can use anything else, e.g., LQR or Pole placement)
        self.Kp_db = 1.0
        self.Kd_db = self.T_DSP + (1/self.lam) * self.coth(self.lam * self.T_SSP)  # deadbeat gains

        # Raibert foot placement gains
        self.Kp_foot = 1.8
        self.Kd_foot = 0.05

        # which foot stepping controller to use
        self.foot_placement_ctrl = "LIP"   # "Raibert" or "LIP"
        # self.foot_placement_ctrl = "Raibert"   # "Raibert" or "LIP"

        # velocity command parameters
        self.vx_cmd_scale = 0.5    # m/s per unit joystick command
        self.vx_cmd = 0.0          # desired forward velocity (used with joystick if connected)
        self.vx_cmd_prev = 0.0    
        self.vx_cmd_curr = 0.0    
        self.vx_cmd_alpha = 0.01   # low-pass filter, (very low b/c fast control loop)

    # update internal state and time
    def update_state(self, data):

        # update time
        self.time = data.time

        # update joint positions and velocities
        self.q = data.qpos.copy()
        self.v = data.qvel.copy()

        # update base state
        self.q_base = self.q[self.idx.POS.POS_X : self.idx.POS.POS_X + 3].copy()
        self.v_base = self.v[self.idx.VEL.VEL_X : self.idx.VEL.VEL_X + 3].copy()

        # update the joint state
        self.q_joints = self.q[self.idx.POS.POS_LH : self.idx.POS.POS_LH + 4].copy()
        self.v_joints = self.v[self.idx.VEL.VEL_LH : self.idx.VEL.VEL_LH + 4].copy()

        # left and right legs
        self.q_left_joints = self.q_joints[[self.idx.JOINT.LH, self.idx.JOINT.LK]].copy()
        self.v_left_joints = self.v_joints[[self.idx.JOINT.LH, self.idx.JOINT.LK]].copy()
        self.q_right_joints = self.q_joints[[self.idx.JOINT.RH, self.idx.JOINT.RK]].copy()
        self.v_right_joints = self.v_joints[[self.idx.JOINT.RH, self.idx.JOINT.RK]].copy()

    # compute the center of mass positions and velocity
    def update_com_state(self, data):

        # total mass
        total_mass = np.sum(self.model.body_mass)

        # position of center of mass in world frame
        com_pos_W = np.zeros(3)
        for i in range(self.model.nbody):
            body_mass = self.model.body_mass[i]
            body_pos = data.xipos[i]      # body position in world frame
            com_pos_W += body_mass * body_pos
        com_pos_W /= total_mass

        # velocity of center of mass in world frame
        com_vel_W = np.zeros(3)
        for i in range(self.model.nbody):
            body_mass = self.model.body_mass[i]
            body_vel = data.cvel[i, 3:]   # linear velocity from cvel (6D: ang[0:3], lin[3:6])
            com_vel_W += body_mass * body_vel
        com_vel_W /= total_mass

        # update the internal state
        self.p_com_W = np.array([com_pos_W[0], com_pos_W[2]])  # only x, z components
        self.v_com_W = np.array([com_vel_W[0], com_vel_W[2]])  # only x, z components

    # update the timing variables
    def update_timing(self, data):

        # compute the current phase value
        self.t_phase = self.time % self.T_SSP
    
        # compute the number of steps taken
        num_steps_old = self.num_steps
        self.num_steps = int(self.time // self.T_SSP)

        # check if there has been a step increment
        new_step = (num_steps_old != self.num_steps)
        if (new_step == True) or (self.first_step == True):

            # update the swing/stance foot info
            self.update_frames(data)

            # set the first step flag to false
            self.first_step = False

    # update the swing and stance foot frames
    def update_frames(self, data):

        # left is swing, right is stance
        if (self.num_steps % 2) == 0:

            # update labels
            self.swing_foot = "left"

            # update the stance and intial swing foot position in world frame
            _stance_pos_W = data.site_xpos[self.right_foot_id, [0, 2]]
            _swing_init_pos_W = data.site_xpos[self.left_foot_id, [0, 2]]

            # reshape and store
            self.stance_foot_pos_W = _stance_pos_W.reshape(2,)
            self.swing_init_pos_W = _swing_init_pos_W.reshape(2,)
        
        # right is swing, left is stance
        else:

            # update labels
            self.swing_foot = "right"

            # update the stance and intial swing foot position in world frame
            _stance_pos_W = data.site_xpos[self.left_foot_id, [0, 2]]
            _swing_init_pos_W = data.site_xpos[self.right_foot_id, [0, 2]]

            # reshape and store
            self.stance_foot_pos_W = _stance_pos_W.reshape(2,)
            self.swing_init_pos_W = _swing_init_pos_W.reshape(2,)

    # update velocity command from joystick
    def update_joystick_command(self):

        # if joystick is connected
        if self.joystick.isConnected:

            # update the joystick inputs
            self.joystick.update()

            # get input
            vx_cmd_raw = self.joystick.LS_Y
            self.vx_cmd_curr = self.vx_cmd_scale * vx_cmd_raw

            # low-pass filter
            self.vx_cmd = (  self.vx_cmd_alpha * self.vx_cmd_curr 
                           + (1.0 - self.vx_cmd_alpha) * self.vx_cmd_prev)
            self.vx_cmd_prev = self.vx_cmd

            # deadband
            if abs(self.vx_cmd) < 0.05:
                self.vx_cmd = 0.0

        # no joystick
        else:
            # will just use the defualt in the init
            pass

    # compute the foot targets
    def compute_foot_targets(self):

        # parse the swing foot initial position and stance foot position
        stance_pos_W = np.array([self.stance_foot_pos_W[0], 
                                 self.z_foot_offset])
        swing_init_pos_W = np.array([self.swing_init_pos_W[0], 
                                     self.z_foot_offset])

        # compute the COM state in STANCE foot frame (world aligned)
        p_com_ST = self.p_com_W - stance_pos_W
        v_com_ST = self.v_com_W

        # extract the x, z components
        p_com = p_com_ST[0]
        v_com = v_com_ST[0]

        # use LIP controller
        if self.foot_placement_ctrl == "LIP":

            # compute the Robot preimpact estimate using LIP dynamics
            x_com = np.array([[p_com], [v_com]])
            x_com_pre = sp.linalg.expm(self.A * (self.T_SSP - self.t_phase)) @ x_com
            p_com_pre = x_com_pre[0][0]
            v_com_pre = x_com_pre[1][0]

            # compute the LIP preimpact state
            p_com_LIP_pre = (self.vx_cmd * self.T) / (2.0 + self.T_DSP * self.sigma_P1)
            v_com_LIP_pre =  self.sigma_P1 * (self.vx_cmd * self.T) / (2.0 + self.T_DSP * self.sigma_P1)

            # LIP foot placement (deadbeat control)
            u_ff = self.vx_cmd * self.T_SSP
            u_fb = self.Kp_db * (p_com_pre - p_com_LIP_pre) + self.Kd_db * (v_com_pre - v_com_LIP_pre)
            u = u_ff + u_fb

        # use Raibert controller
        elif self.foot_placement_ctrl == "Raibert":

            # Raibert foot placement controller
            u = (  self.Kp_foot * p_com 
                 + self.Kd_foot * (v_com - self.vx_cmd) 
                 + 0.5 * self.vx_cmd * self.T_SSP)
        
        print(f"vx_cmd: {self.vx_cmd:.2f} m/s, vx_com: {v_com:.2f} m/s")

        # compute the foot placement (apex, 5th order use 8/3, 7th order use 16/5)
        swing_init_W = swing_init_pos_W
        swing_target_W = stance_pos_W + np.array([u, 0.0])
        swing_middle_W = np.array([(swing_init_W[0] + swing_target_W[0]) / 2.0, 
                                   (16.0/5.0) * (self.z_apex + self.z_foot_offset)])
        
        # compute the swing foot trajectory using a Bezier curve
        ctrl_pts_swing = np.array([swing_init_W,
                                   swing_init_W,
                                   swing_init_W,
                                   swing_middle_W,
                                   swing_target_W,
                                   swing_target_W,
                                   swing_target_W])
        t_eval = np.array([self.t_phase / self.T_SSP])
        p_swing_des, v_swing_des = bezier_curve(t_eval, ctrl_pts_swing)
    
        # left foot is swing
        if self.swing_foot == "left":
            
            # right foot is stationary
            p_right_des = stance_pos_W
            v_right_des = np.array([0.0, 0.0])
            
            # left foot is swing
            p_left_des = p_swing_des[0].flatten()
            v_left_des = v_swing_des[0].flatten()

        # right foot is swing
        elif self.swing_foot == "right":
            
            # left foot is stationary
            p_left_des = stance_pos_W
            v_left_des = np.array([0.0, 0.0])
            
            # right foot is swing
            p_right_des = p_swing_des[0].flatten()
            v_right_des = v_swing_des[0].flatten()

        return p_left_des, p_right_des, v_left_des, v_right_des

    # compute the control input
    def compute_input(self, data):

        # update the internal state
        self.update_state(data)

        # update COM state
        self.update_com_state(data)

        # update the timing variables
        self.update_timing(data)

        # update the joystick command
        self.update_joystick_command()

        # compute desired foot positions and velocities
        p_left_des_W, p_right_des_W, v_left_des_W, v_right_des_W = self.compute_foot_targets()
        p_left_des_W = p_left_des_W.reshape(2,1)
        p_right_des_W = p_right_des_W.reshape(2,1)
        v_left_des_W = v_left_des_W.reshape(2,1)
        v_right_des_W = v_right_des_W.reshape(2,1)

        # compute desired base position and velocity
        p_base_des_W = np.array([self.q_base[0], self.z_base + self.z_foot_offset]).reshape(2,1)
        o_base_des_W = -0.2
        v_base_des_W = np.array([self.v_base[0], 0.0]).reshape(2,1)
        w_base_des_W = 0.0

        # do IK
        q_des, _ = self.ik.ik_world(p_base_des_W, o_base_des_W, p_left_des_W, p_right_des_W,
                                        v_base_des_W, w_base_des_W, v_left_des_W, v_right_des_W)

        # extract the desired joint positions and velocities
        q_joints_des = q_des[3:7].flatten()
        v_joints_des = np.zeros(4)

        # compute the control
        tau = (  self.kp * (q_joints_des - self.q_joints) 
               + self.kd * (v_joints_des - self.v_joints))

        # account for the gear ratio
        tau[0] /= self.gear_ratio_LH
        tau[1] /= self.gear_ratio_LK
        tau[2] /= self.gear_ratio_RH
        tau[3] /= self.gear_ratio_RK

        # clip the torques
        tau = np.clip(tau, -1.0, 1.0)

        return tau


##################################################################################

# main function
if __name__ == "__main__":

    # model path
    model_file = "./models/biped.xml"

    # load the file
    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)

    # setup the glfw window
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    window = glfw.create_window(1920, 1080, "Robot", None, None)
    glfw.make_context_current(window)

    # set the window to be resizable
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)

    # create camera to render the scene
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    cam.distance = 2.5
    cam.elevation = -15
    cam.azimuth = 90
    cam.lookat[1] = 0.0
    cam.lookat[2] = 0.6
    
    # create the scene and context
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_200)

    # create an object for indexing
    idx = Biped_IDX()

    # create controller object
    control = Controller(model_file)

    # set the initial state
    key_name = "standing"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    data.qpos = model.key_qpos[key_id]
    data.qvel = model.key_qvel[key_id]

    # simulation setup
    hz_render = 50.0
    t_max = 60.0

    # compute the decimation
    sim_dt = model.opt.timestep

    # wall clock timing variables
    t_sim = 0.0
    wall_start = time.time()
    last_render = 0.0

    # do one update of the scene
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

    while (not glfw.window_should_close(window)) and (t_sim < t_max):

        # get the current sim time and state
        t_sim = data.time

        # compute the inputs
        tau = control.compute_input(data)

        data.ctrl[:] = tau

        # hold the base in the air
        # data.qpos[idx.POS.POS_X] = 0.0
        # data.qpos[idx.POS.POS_Z] = 1.0
        # data.qpos[idx.POS.EUL_Y] = 0.0
        # data.qvel[idx.VEL.VEL_X] = 0.0
        # data.qvel[idx.VEL.VEL_Z] = 0.0
        # data.qvel[idx.VEL.ANG_Y] = 0.0

        # set the torques
        data.ctrl[:] = tau

        # step the simulation
        mujoco.mj_step(model, data)

        # render at desired rate
        if t_sim - last_render > 1.0 / hz_render:

            # move the camer to point at the robot
            cam.lookat[0] = data.qpos[idx.POS.POS_X]

            # update the scene, render
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)

            # display the simulation time overlay
            label_text = f"Sim Time: {t_sim:.2f} sec \nWall Time: {wall_elapsed:.2f} sec"
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_200,   # font scale
                mujoco.mjtGridPos.mjGRID_TOPLEFT,      # position on screen
                viewport,                              # this must be the MjrRect, not context
                label_text,                            # main overlay text (string, not bytes)
                "",                                    # optional secondary text
                context                                # render context
            )

            # swap the OpenGL buffers and poll for GUI events
            glfw.swap_buffers(window)
            glfw.poll_events()

            # record the last render time
            last_render = t_sim

        # sync the sim time with the wall clock time
        wall_elapsed = time.time() - wall_start
        if t_sim > wall_elapsed:
            time.sleep(t_sim - wall_elapsed)

        # exit if the sim time exceeds max time
        if t_sim >= t_max:
            glfw.set_window_should_close(window, True)
            break
