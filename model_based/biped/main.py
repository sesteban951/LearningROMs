# standard includes
import numpy as np
import time          
import math
import matplotlib.pyplot as plt

# pacakge includes
import mujoco 
import glfw

# custom includes 
from indeces import Biped_IDX
from inverse_kinematics import InverseKinematics

##################################################################################

class Controller:

    def __init__(self, model_file):

        # load the model file
        model = mujoco.MjModel.from_xml_path(model_file)
        
        # create indexing object
        self.idx = Biped_IDX()

        # create IK object
        self.ik = InverseKinematics(model_file)

        # gains (NOTE: this does not take gear reduction into account)
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

    def compute_input(self, data):

        # extract the joint positions and velocities
        q = data.qpos.copy()
        v = data.qvel.copy()

        # extract the joint positions and velocities
        q_joints = q[self.idx.POS.POS_LH : self.idx.POS.POS_LH + 4].copy()
        v_joints = v[self.idx.VEL.VEL_LH : self.idx.VEL.VEL_LH + 4].copy()

        # squats
        p_left_des = np.array([-0.20, -0.7]).reshape(2,1)
        p_right_des = np.array([0.25, -0.7]).reshape(2,1)
        v_left_des = np.zeros((2,1))
        v_right_des = np.zeros((2,1))

        p_left_des[0] += 0.05 * math.cos(2.0 * math.pi * 0.5 * data.time)
        p_left_des[1] += 0.05 * math.sin(2.0 * math.pi * 0.5 * data.time)
        p_right_des[0] += 0.05 * math.cos(2.0 * math.pi * 0.5 * data.time)
        p_right_des[1] += 0.05 * math.sin(2.0 * math.pi * 0.5 * data.time)

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
        tau = self.kp * (q_joints_des - q_joints) + self.kd * (v_joints_des - v_joints)

        # account for the gear ratio
        tau[0] /= self.gear_ratio_LH
        tau[1] /= self.gear_ratio_LK
        tau[2] /= self.gear_ratio_RH
        tau[3] /= self.gear_ratio_RK

        # clip the torques
        tau = np.clip(tau, -1.0, 1.0)

        return tau

    # compute a bezier curve based on control points
    def bezier_curve(self, t_pts, ctrl_pts):
        """
        Compute Bezier Curve from control points

        Args:
            t_pts (np.array): list of time points (0 <= t <= 1)
            ctrl_pts (np.array): Control points of shape (n, d) where n is the number of control points
                                  and d is the dimension (2 for 2D, 3 for 3D).
        Returns: 
            y (np.array): Points on the Bezier curve of shape (m, d)
            ydot (np.array): Derivative of the Bezier curve at the points of shape (m, d)
        """
        
        # determine the degree and dimension of the curve
        deg = ctrl_pts.shape[0] - 1 # (e.g., deg = 1, curve is linear)
        dim = ctrl_pts.shape[1]

        # evaluation points
        num_eval_pts = t_pts.shape[0]

        # build the curves
        y = np.zeros((num_eval_pts, dim))
        ydot = np.zeros((num_eval_pts, dim))

        # compute the curve
        for i in range(deg + 1):
            bernstein = math.comb(deg, i) * (t_pts ** i) * ((1 - t_pts) ** (deg - i))
            y += np.outer(bernstein, ctrl_pts[i])

        # precompute the scaling for the derivative
        ctrl_diff_coeffs = deg * (ctrl_pts[1:] - ctrl_pts[:-1])

        # compute the derivative
        for i in range(deg):
            bernstein = math.comb(deg - 1, i) * (t_pts ** i) * ((1 - t_pts) ** (deg - 1 - i))
            ydot += np.outer(bernstein, ctrl_diff_coeffs[i])

        return y, ydot

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
