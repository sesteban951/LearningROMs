 # standardincludes
import numpy as np
import time

# pacakge includes
import mujoco 
import glfw
from pydrake.all import *

# custom includes 
from indeces import HotdogMan_IDX
from inverse_kinematics import InverseKinematics

##################################################################################

# main function
if __name__ == "__main__":

    # model path
    model_file = "./models/hotdog_man.xml"

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
    idx = HotdogMan_IDX()

    # create IK object
    ik = InverseKinematics(model_file)

    # set the gains
    Kp = np.array([150, 150, 150, 150])
    Kd = np.array([15, 15, 15, 15])

    # default base position
    q_base_init = np.array([0.0, 0.9, 0.0])
    v_base_init = np.array([0.0, 0.0, 0.0])
    
    # default joint positions
    default_joints = np.array([-0.5, 0.5, -0.1, 0.5])
    q_joints_init = default_joints.copy()
    v_joints_init = np.zeros(4)

    # set the initial state
    data.qpos[idx.POS.POS_X] = q_base_init[0]
    data.qpos[idx.POS.POS_Z] = q_base_init[1]
    data.qpos[idx.POS.EUL_Y] = q_base_init[2]
    data.qpos[idx.POS.POS_LH : idx.POS.POS_LH + 4] = q_joints_init
    data.qvel[idx.VEL.VEL_X] = v_base_init[0]
    data.qvel[idx.VEL.VEL_Z] = v_base_init[1]
    data.qvel[idx.VEL.ANG_Y] = v_base_init[2]
    data.qvel[idx.VEL.VEL_LH : idx.VEL.VEL_LH + 4] = v_joints_init

    # simulation setup
    hz_render = 50.0
    hz_control = 100.0
    t_max = 15.0

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

        # get the current joint positions and velocities
        q_joints = data.qpos[idx.POS.POS_LH : idx.POS.POS_LH + 4].copy()
        v_joints = data.qvel[idx.VEL.VEL_LH : idx.VEL.VEL_LH + 4].copy()

        # control at desired Hz
        if counter % decimation == 0:

            # desired foot position in base frame
            c_left = np.array([[0.2], [-0.65]])
            c_right = np.array([[-0.2], [-0.65]])

            f = 1.0
            w = 2.0 * np.pi * f
            A = 0.1
            c_left[0] += A * np.sin(w * t_sim)
            c_left[1] += A * np.cos(w * t_sim)
            c_right[0] += A * np.sin(w * t_sim + np.pi)
            c_right[1] += A * np.cos(w * t_sim + np.pi)
            c_left_x = A * w  * np.cos(w * t_sim)
            c_left_z = -A * w * np.sin(w * t_sim)
            c_right_x = A * w * np.cos(w * t_sim + np.pi)
            c_right_z = -A * w * np.sin(w * t_sim + np.pi)

            p_des_base_l = c_left
            p_des_base_r = c_right
            v_des_base_l = np.array([[c_left_x], [c_left_z]])
            v_des_base_r = np.array([[c_right_x], [c_right_z]])

            q_des_l, qdot_des_l = ik.ik_feet_in_base(p_des_base_l, v_des_base_l)
            q_des_r, qdot_des_r = ik.ik_feet_in_base(p_des_base_r, v_des_base_r)

            # PD control
            q_joints_des = np.vstack((q_des_l, q_des_r)).flatten()
            v_joints_des = np.vstack((qdot_des_l, qdot_des_r)).flatten()

            # reset the counter 
            counter = 0

        # fix some states of the robot
        data.qpos[idx.POS.POS_X] = 0.0
        data.qpos[idx.POS.POS_Z] = 1.0
        data.qpos[idx.POS.EUL_Y] =0.0

        # step the counter
        counter += 1

        # set the torques
        tau = Kp * (q_joints_des - q_joints) + Kd * (v_joints_des - v_joints)
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
