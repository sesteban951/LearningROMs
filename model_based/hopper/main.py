# standard includes
import numpy as np
import math
import time

# pacakge includes
import mujoco 
import glfw

# custom includes 
from indeces import Hopper_IDX
from utils import Joy

##################################################################################

class Controller:

    def __init__(self, model_file):

        # load the model file
        model = mujoco.MjModel.from_xml_path(model_file)
        sim_dt = model.opt.timestep

        # get gravity 
        self.gravity = abs(model.opt.gravity[2])

        # create joystick object
        self.joystick = Joy()

        # get some IDs
        upper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        lower_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lower_leg")
        leg_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide")
        body_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "body_motor")
        leg_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "leg_motor")

        # extract upper and lower body mass
        self.upper_body_mass = model.body_mass[upper_body_id]
        self.lower_body_mass = model.body_mass[lower_body_id]

        # cache sensor IDs
        self.sid_torso = model.sensor("torso_touch").id
        self.sid_foot  = model.sensor("foot_touch").id

        # extract the joint stiffness and damping
        self.k_leg = model.jnt_stiffness[leg_joint_id]
        self.b_leg = model.dof_damping[leg_joint_id]

        # extract the gear ratio
        self.gear_body = model.actuator_gear[body_actuator_id, 0]
        self.gear_leg = model.actuator_gear[leg_actuator_id, 0]

        # create indexing object
        self.idx = Hopper_IDX()

        # create gain arrays 
        # Note: these gains do not take the gear ratio into account
        self.kp_body = 20.0
        self.kd_body = 1.0

        self.kp_leg_air = 1000.0
        self.kd_leg_air = 10.0

        self.kp_leg_ground = 1000.0
        self.kd_leg_ground = 10.0

        # raibert stepping gains
        self.kp_raibert = 1.0
        self.kd_raibert = 0.15

        # velocity command parameters
        self.vx_cmd_scale = 1.0    # m/s per unit joystick command
        self.vx_cmd = 0.5          # desired forward velocity (used with joystick if connected)
        
        # low pass filter
        f_cutoff = 0.75
        omega_c = 2.0 * np.pi * f_cutoff
        self.vx_cmd_alpha = np.exp(-omega_c * sim_dt)
        self.vx_cmd_prev = 0.0    
        self.vx_cmd_curr = 0.0    

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
            self.vx_cmd = (  (1.0 - self.vx_cmd_alpha) * self.vx_cmd_curr 
                           + self.vx_cmd_alpha* self.vx_cmd_prev)
            self.vx_cmd_prev = self.vx_cmd

            # deadband
            if abs(self.vx_cmd) < 0.05:
                self.vx_cmd = 0.0

        # no joystick
        else:
            # will just use the default that was set in the init
            pass

    # simple controller
    def compute_input(self, t, data):

        # extract the time and state
        q = data.qpos.copy()
        v = data.qvel.copy()

        # parse the state
        pos_leg = q[self.idx.POS.POS_LEG]
        vel_leg = v[self.idx.VEL.VEL_LEG]

        vx_body = v[self.idx.VEL.VEL_X]
        theta_body = q[self.idx.POS.EUL_Y]
        thetadot_body = v[self.idx.VEL.ANG_Y]

        # parse contact information
        foot_in_contact, torso_in_contact = self.parse_contact(data)

        # update the velocity command from joystick
        self.update_joystick_command()
        
        # Ground
        if foot_in_contact:

            # TODO: implement state machine. 
            #       1) Let the robot compress, then when beginning to extend, switch to thrusting
            #       2) control theta to exit the ground phase at desired angle

            # feedforward forces
            F_mass_ff = -self.gravity * (self.upper_body_mass)
            F_spring_ff = self.k_leg * pos_leg + self.b_leg * vel_leg

            # feedback forces (NOTE: something is weird about the sign here)
            pos_leg_des_gnd = 0.5
            vel_leg_des_gnd = 0.0
            F_fb = (- self.kp_leg_ground * (pos_leg_des_gnd - pos_leg) 
                    - self.kd_leg_ground * (vel_leg_des_gnd - vel_leg))
            
            # total leg force
            F_leg = 0.0
            F_leg += F_mass_ff
            F_leg += F_spring_ff
            F_leg += F_fb

            # compute the theta torque
            T_body = 0.0

        # Flight
        else:

            # TODO: implement a raibert controller: "Legged Robots that Balance", eq. (2.4)

            # desired leg position and velocity
            pos_leg_des_air = 0.5
            vel_leg_des_air = 0.0

            # compute the force
            F_leg = (  self.kp_leg_air * (pos_leg_des_air - pos_leg) 
                     + self.kd_leg_air * (vel_leg_des_air - vel_leg))

            # desired angle
            theta_des = self.kd_raibert * (self.vx_cmd - vx_body)

            # clip the desired angle
            theta_des = np.clip(theta_des, -0.7, 0.7)

            # compute the theta torque
            T_body = self.kp_body * (theta_des - theta_body) + self.kd_body * (0.0 - thetadot_body)

        # apply gear ratio
        T_body /= self.gear_body
        F_leg /= self.gear_leg

        # clip
        F_leg = np.clip(F_leg, -1, 1)
        T_body = np.clip(T_body, -1, 1)

        # print some info
        print("vx_cmd: {:2f}, vx: {:2f}, vx_err: {:2f}".format(self.vx_cmd, vx_body, self.vx_cmd - vx_body))
        # print(f"F_leg: {F_leg:.2f}, T_body: {T_body:.2f}, Contact: {foot_in_contact}")

        # compute the torque
        tau = np.array([T_body, F_leg])
        
        return tau
    
    # parse contact information
    def parse_contact(self, data):

        # read the sensor values
        foot_force  = data.sensordata[self.sid_foot]
        torso_force = data.sensordata[self.sid_torso]

        # thresholds avoid noise
        foot_in_contact  = foot_force  > 1e-6
        torso_in_contact = torso_force > 1e-6

        return foot_in_contact, torso_in_contact
        
    
##################################################################################

# main function
if __name__ == "__main__":

    # model path
    model_file = "./models/hopper.xml"

    # load the file
    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)

    # change the sim timestep
    model.opt.timestep = 0.002

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
    idx = Hopper_IDX()

    # create controller object
    controller = Controller(model_file)

    # set the initial state
    key_name = "default"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    data.qpos = model.key_qpos[key_id]
    data.qvel = model.key_qvel[key_id]

    # simulation setup
    hz_render = 50.0
    hz_control = 50.0
    t_max = 60.0

    # compute the decimation
    sim_dt = model.opt.timestep
    control_dt = 1.0 / hz_control
    decimation = round(control_dt / sim_dt)
    counter = 0

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

        # control at desired Hz
        if counter % decimation == 0:

            # compute the control
            u = controller.compute_input(t_sim, data)

            # reset the counter
            counter = 0

        # step the counter
        counter += 1

        # set the torques
        data.ctrl[:] = u

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
