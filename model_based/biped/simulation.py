# standard includes
import numpy as np
import scipy as sp
import time        
from dataclasses import dataclass

# pacakge includes
import mujoco 
import glfw

# custom includes 
from indeces import Biped_IDX
from utils import InverseKinematics, bezier_curve, Joy

##################################################################################
# CONTROLLER
##################################################################################

class Controller:

    def __init__(self, model_file):

        # load the model file
        model = mujoco.MjModel.from_xml_path(model_file)
        self.model = model

        # create indexing object
        self.idx = Biped_IDX()

        # create IK object
        self.ik = InverseKinematics(model_file, verbose=False)

        # gains (NOTE: these does not take gear reduction into account)
        self.kp = np.array([200.0, 200.0, 200.0, 200.0])
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

        # cache sensor IDs
        self.sid_left_foot  = model.sensor("left_foot_touch").id
        self.sid_right_foot = model.sensor("right_foot_touch").id

        # internal state and time
        self.time = 0.0
        self.q_base = np.zeros(3)
        self.v_base = np.zeros(3)
        self.q_joints = np.zeros(4)
        self.v_joints = np.zeros(4)

        # Timing variables
        self.t_phase = 0.0
        self.num_steps = 0
        self.first_step = True
        self.T_SSP = 0.4
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
        self.bez_order = 4           # order of the Bezier curve for swing foot trajectory (4 or 6)  

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
        self.Kp_raibert = 1.4
        self.Kd_raibert = 0.2

        # max step increment
        self.u_max = 0.4

        # feedforward bias input for stepping
        self.u_bias = -0.025

        # desired base orientation
        self.base_orient_des = -0.2

        # which foot stepping controller to use
        self.foot_placement_ctrl = "LIP"   # "Raibert" or "LIP"
        # self.foot_placement_ctrl = "Raibert"   # "Raibert" or "LIP"


    # update internal state and time
    def update_state(self, data):

        # update time
        self.time = data.time

        # update base state
        self.q_base = data.qpos[self.idx.POS.POS_X : self.idx.POS.POS_X + 3].copy()
        self.v_base = data.qvel[self.idx.VEL.VEL_X : self.idx.VEL.VEL_X + 3].copy()

        # update the joint state
        self.q_joints = data.qpos[self.idx.POS.POS_LH : self.idx.POS.POS_LH + 4].copy()
        self.v_joints = data.qvel[self.idx.VEL.VEL_LH : self.idx.VEL.VEL_LH + 4].copy()

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

    # compute the foot targets
    def compute_foot_targets(self, vx_cmd):

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
            p_com_LIP_pre = (vx_cmd * self.T) / (2.0 + self.T_DSP * self.sigma_P1)
            v_com_LIP_pre =  self.sigma_P1 * (vx_cmd * self.T) / (2.0 + self.T_DSP * self.sigma_P1)

            # LIP foot placement (deadbeat control)
            u_ff = vx_cmd * self.T_SSP
            u_fb = self.Kp_db * (p_com_pre - p_com_LIP_pre) + self.Kd_db * (v_com_pre - v_com_LIP_pre)
            u = u_ff + u_fb + self.u_bias

        # use Raibert style controller (added the Kp term for better regulation)
        elif self.foot_placement_ctrl == "Raibert":

            # Raibert foot placement controller
            u_ff = 0.5 * v_com * self.T_SSP
            u_fb = self.Kd_raibert * (v_com - vx_cmd) + self.Kp_raibert * (p_com - 0.0)
            u = u_ff + u_fb + self.u_bias

        # invalid foot placement controller
        else:
            raise ValueError("Invalid foot placement controller selected")
        
        # saturate the step increment (in base frame)
        px_base_W = self.q_base[0]
        u_W = stance_pos_W[0] + u
        u_base = u_W - px_base_W
        u_base = np.clip(u_base, -self.u_max, self.u_max)
        u_W = u_base + px_base_W
        u = u_W - stance_pos_W[0]

        # compute the swing foot landing target
        swing_init_W = swing_init_pos_W
        swing_target_W = stance_pos_W + np.array([u, 0.0])

        # 4th order Bezier curve
        if self.bez_order == 4:

            # compute the middle control point (highest point in swing)
            swing_middle_W = np.array([(swing_init_W[0] + swing_target_W[0]) / 2.0, 
                                       (8.0/3.0) * (self.z_apex + self.z_foot_offset)])
            
            # compute the swing foot trajectory using a Bezier curve
            ctrl_pts_swing = np.array([swing_init_W,
                                       swing_init_W,
                                       swing_middle_W,
                                       swing_target_W,
                                       swing_target_W])
        
        # 6th order Bezier curve
        elif self.bez_order == 6:

            # compute the middle control point (highest point in swing)
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
            
        # invalid Bezier order
        else:
            raise ValueError("Invalid Bezier curve order, must be 4 or 6")

        # evaluate the Bezier curve at the current phase
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
    def compute_input(self, data, vel_cmd):

        # update the internal state
        self.update_state(data)

        # update COM state
        self.update_com_state(data)

        # update the timing variables
        self.update_timing(data)

        # print contact info
        self.parse_contact(data)

        # compute desired foot positions and velocities
        p_left_des_W, p_right_des_W, v_left_des_W, v_right_des_W = self.compute_foot_targets(vel_cmd)
        p_left_des_W = p_left_des_W.reshape(2,1)
        p_right_des_W = p_right_des_W.reshape(2,1)
        v_left_des_W = v_left_des_W.reshape(2,1)
        v_right_des_W = v_right_des_W.reshape(2,1)

        # compute desired base position and velocity
        p_base_des_W = np.array([self.q_base[0], self.z_base + self.z_foot_offset]).reshape(2,1)
        o_base_des_W = self.base_orient_des
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

    # parse contact information
    def parse_contact(self, data):

        # read the sensor values
        left_foot_force  = data.sensordata[self.sid_left_foot]
        right_foot_force = data.sensordata[self.sid_right_foot]

        return left_foot_force, right_foot_force


##################################################################################
# UTILS
##################################################################################

# update joystick command
def update_joystick_command(joystick, scaling):

    # update the joystick state
    joystick.update()

    # get the axis value
    raw_val = joystick.LS_Y

    # deadband
    if abs(raw_val) < 0.1:
        raw_val = 0.0

    # scale the value
    cmd = scaling * raw_val

    return cmd

##################################################################################
# SIMULATION
##################################################################################


# simulation config
@dataclass
class SimulationConfig:

    # visualization parameters
    visualization: bool = True

    # time parameters
    sim_dt: float = 0.002
    sim_time: float = 10.0

    # default command
    cmd_scaling: float = 1.0
    cmd_default: float = 0.0

# main simulation class
class Simulation:
    
    # initializer
    def __init__(self, config: SimulationConfig):

        # store config
        self.config = config
        
    # simulate function
    def simulate(self):
        
        # model path
        model_file = "./models/biped.xml"

        # load the file
        model = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model) 

        # change the sim timestep
        model.opt.timestep = self.config.sim_dt

        # compute the decimation
        if self.config.visualization == True:

            # setup the glfw window
            if not glfw.init():
                raise Exception("Could not initialize GLFW")
            window = glfw.create_window(1080, 740, "Robot", None, None)
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

            # rendering frequency
            hz_render = 50.0

            # wall clock timing variables
            wall_start = time.time()
            last_render = 0.0

            # do one update of the scene
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()
        
        # create an object for indexing
        idx = Biped_IDX()

        # create controller object
        controller = Controller(model_file)

        # create joystick object
        joystick = Joy()

        # set the initial state
        key_name = "default"
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        data.qpos = model.key_qpos[key_id]
        data.qvel = model.key_qvel[key_id]

        # simulation setup
        t_max = self.config.sim_time

        # command scaling
        vx_cmd_scale = self.config.cmd_scaling
        vx_cmd = self.config.cmd_default

        # instantiate the data logging arrays
        num_nodes = int(np.ceil(t_max / self.config.sim_dt)) + 1
        t_data = np.zeros((num_nodes, 1))
        q_data = np.zeros((num_nodes, model.nq))
        v_data = np.zeros((num_nodes, model.nv))
        u_data = np.zeros((num_nodes, model.nu))
        c_data = np.zeros((num_nodes, 2))
        cmd_data = np.zeros((num_nodes, 1))

        # initialize the sim step counter
        counter = 0

        # main simulation loop
        while  (counter < num_nodes):

            # get the current sim time and state
            t_sim = data.time

            # update the joystick command
            if joystick.isConnected and self.config.visualization == True:
                vx_cmd = update_joystick_command(joystick, vx_cmd_scale)

            # compute the control
            tau = controller.compute_input(data, vx_cmd)

            # set the torques
            data.ctrl[:] = tau

            # log the data
            t_data[counter] = t_sim
            q_data[counter, :] = data.qpos
            v_data[counter, :] = data.qvel
            u_data[counter, :] = tau
            c_data[counter, :] = controller.parse_contact(data)
            cmd_data[counter, :] = vx_cmd

            # step the simulation
            mujoco.mj_step(model, data)

            # render at desired rate
            if self.config.visualization == True:

                # render at desired rate
                if t_sim - last_render > 1.0 / hz_render:

                    # move the camera to point at the robot
                    cam.lookat[0] = data.qpos[idx.POS.POS_X]

                    # update the scene, render
                    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    mujoco.mjr_render(viewport, scene, context)

                    # display the simulation time overlay
                    label_text = f"Sim Time: {t_sim:.2f} sec \nWall Time: {wall_elapsed:.2f} sec\nCmd Vel: {vx_cmd:.2f} m/s"
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

            # increment the counter
            counter += 1

        # save the logged info to a npz file
        file_name = "./model_based/biped/biped_data.npz"
        np.savez(file_name,
                t_log=t_data,
                q_log=q_data,
                v_log=v_data,
                u_log=u_data,
                c_log=c_data,
                cmd_log=cmd_data)
        print(f"Saved simulation data to {file_name}")

        return t_data, q_data, v_data, u_data, c_data, cmd_data

##################################################################################

# main function
if __name__ == "__main__":
    
    # create simulation config
    sim_config = SimulationConfig(
        visualization=True, # visualize or not
        sim_dt=0.002,        # sim time step
        sim_time=10.0,       # total sim time
        cmd_scaling=1.0,     # scaling of the command for joysticking 
        cmd_default=0.0      # default forward velocity command
    )

    # create simulation object
    simulation = Simulation(sim_config)

    # run the simulation
    t0 = time.time()
    t_log, q_log, v_log, u_log, c_log, cmd_log = simulation.simulate()
    t1 = time.time()
    print(f"Simulation time: {t1 - t0:.2f} seconds")
