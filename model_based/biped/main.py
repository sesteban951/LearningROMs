# standard includes
import numpy as np
import time          

# pacakge includes
import mujoco 
import glfw

# custom includes 
from indeces import Biped_IDX
from model_based.biped.utils import InverseKinematics, bezier_curve

##################################################################################

class Controller:

    def __init__(self, model_file):

        # load the model file
        model = mujoco.MjModel.from_xml_path(model_file)
        
        # create indexing object
        self.idx = Biped_IDX()

        # create IK object
        self.ik = InverseKinematics(model_file)

        # gains (NOTE: these does not take gear reduction into account)
        self.kp = np.array([250.0, 250.0, 250.0, 250.0])
        self.kd = np.array([15.0, 15.0, 15.0, 15.0])

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
        self.q_joints = np.zeros(4)
        self.v_joints = np.zeros(4)

        # Timing variables
        self.t_phase = 0.0
        self.num_steps = 0
        self.first_step = True

        # ROM variables
        self.T_SSP = 0.5

        # frame variables
        self.swing_foot = None
        self.stance_foot = None

    # update interrnal state and time
    def update_state(self, data):

        # update time
        self.time = data.time

        # update joint positions and velocities
        self.q = data.qpos.copy()
        self.v = data.qvel.copy()

        # update the joint state
        self.q_joints = self.q[self.idx.POS.POS_LH : self.idx.POS.POS_LH + 4].copy()
        self.v_joints = self.v[self.idx.VEL.VEL_LH : self.idx.VEL.VEL_LH + 4].copy()

    # update the timing variables
    def update_timing(self):

        # copmute the current phase value
        self.t_phase = self.time % self.T_SSP
    
        # compute the number of steps taken
        num_steps_old = self.num_steps
        self.num_steps = int(self.time // self.T_SSP)

        # check if there has been a step increment
        new_step = (num_steps_old != self.num_steps)
        if (new_step == True) or (self.first_step == True):

            # update the swing/stance foot info
            self.update_frames()

            # set the first step flag to false
            self.first_step = False

    # update the swing and stance foot frames
    def update_frames(self):

        # left is swing, right is stance
        if (self.num_steps % 2) == 0:
            self.swing_foot = "left"
            self.stance_foot = "right"
        else:
            self.swing_foot = "right"
            self.stance_foot = "left"

    # compute the foot targets
    def compute_foot_targets(self):
        
        # left foot is swing
        if self.swing_foot == "left":
            # right foot is stationary
            p_right_des = np.array([[0.2], [-0.7]])
            v_right_des = np.array([[0.0], [0.0]])

            px_0 = -0.2
            pz_0 = -0.7
            px_1 = -0.2
            pz_1 = -0.45
            px_2 = -0.2
            pz_2 = -0.7
            ctrl_pts_swing = np.array([[px_0, pz_0],
                                       [px_1, pz_1],
                                       [px_2, pz_2]])
            t_eval = np.array([self.t_phase / self.T_SSP])
            p_left_des, v_left_des = bezier_curve(t_eval, ctrl_pts_swing)

            p_left_des = p_left_des[0].reshape(2,1)
            v_left_des = v_left_des[0].reshape(2,1)

        # right foot is swing
        elif self.swing_foot == "right":
            # left foot is stationary
            p_left_des = np.array([[-0.2], [-0.7]])
            v_left_des = np.array([[0.0], [0.0]])

            px_0 = 0.2
            pz_0 = -0.7
            px_1 = 0.2
            pz_1 = -0.45
            px_2 = 0.2
            pz_2 = -0.7
            ctrl_pts_swing = np.array([[px_0, pz_0],
                                       [px_1, pz_1],
                                       [px_2, pz_2]])
            t_eval = np.array([self.t_phase / self.T_SSP])
            p_right_des, v_right_des = bezier_curve(t_eval, ctrl_pts_swing)

            p_right_des = p_right_des[0].reshape(2,1)
            v_right_des = v_right_des[0].reshape(2,1)

        return p_left_des, p_right_des, v_left_des, v_right_des

    # compute the control input
    def compute_input(self, data):

        # update the internal state
        self.update_state(data)

        # update the timing variables
        self.update_timing()

        # compute desired foot positions and velocities
        p_left_des, p_right_des, v_left_des, v_right_des = self.compute_foot_targets()

        # do IK
        q_left_des, v_left_des = self.ik.ik_feet_in_base(p_left_des, v_left_des)
        q_right_des, v_right_des = self.ik.ik_feet_in_base(p_right_des, v_right_des)

        # build the full desired joint vectors
        q_joints_des = np.zeros(4)
        q_joints_des[0:2] = q_left_des.flatten()
        q_joints_des[2:4] = q_right_des.flatten()
        
        v_joints_des = np.zeros(4)
        v_joints_des[0:2] = v_left_des.flatten()
        v_joints_des[2:4] = v_right_des.flatten()

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
    window = glfw.create_window(1080, 720, "Robot", None, None)
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
    t_max = 15.0

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
        data.qpos[idx.POS.POS_X] = 0.0
        data.qpos[idx.POS.POS_Z] = 1.0
        data.qpos[idx.POS.EUL_Y] = 0.0
        data.qvel[idx.VEL.VEL_X] = 0.0
        data.qvel[idx.VEL.VEL_Z] = 0.0
        data.qvel[idx.VEL.ANG_Y] = 0.0

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
