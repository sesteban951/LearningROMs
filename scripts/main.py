 # standardincludes
import numpy as np
import time

# pacakge includes
import mujoco 
import glfw
from pydrake.all import *

# custom includes 
from indeces import HotdogMan_IDX

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

    # set the gains
    Kp = np.array([75, 75, 75, 75])
    Kd = np.array([5, 5, 5, 5])

    # default base position
    q_base_init = np.array([0.0, 0.8, 0.0])
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
    hz_control = 50.0
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

        # control at desired Hz
        if counter % decimation == 0:

            # get the current joint positions and velocities
            q_joints = data.qpos[idx.POS.POS_LH : idx.POS.POS_LH + 4].copy()
            v_joints = data.qvel[idx.VEL.VEL_LH : idx.VEL.VEL_LH + 4].copy()

            # PD control
            q_joints_des = default_joints + 0.1 * np.sin(2 * np.pi * 0.2 * t_sim)
            v_joints_des = np.zeros(4)

            # reset the counter 
            counter = 0

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
