# standard includes
import numpy as np
import time

# pacakge includes
import mujoco 
import glfw

# custom includes 
from indeces import Hopper_IDX

##################################################################################

class Controller:

    def __init__(self, model_file):
        
        # create indexing object
        self.idx = Hopper_IDX()

        # create gain arrays 
        # Note: that torques are normalized and there is a gear ratio, hence small gains
        self.Kp = np.array([1.5, 1.5])
        self.Kd = np.array([0.05, 0.05])

    def compute_input(self, t, state):

        # extract the time and state
        q = state[:self.idx.POS.SIZE].flatten()
        v = state[self.idx.POS.SIZE:].flatten()

        # extract controlled joints
        q_base_act = q[self.idx.POS.EUL_Y]
        q_leg_act = q[self.idx.POS.POS_LEG]
        v_base_act = v[self.idx.VEL.ANG_Y]
        v_leg_act = v[self.idx.VEL.VEL_LEG]
        q_act = np.array([q_base_act, q_leg_act])
        v_act = np.array([v_base_act, v_leg_act])

        # default desired positions and velocities
        q_base_des = 0.0
        q_leg_des = 0.4 * np.sin(2 * np.pi * 2.0 * t)
        v_base_des = 0.0
        v_leg_des = 0.0
        q_des = np.array([q_base_des, q_leg_des])
        v_des = np.array([v_base_des, v_leg_des])

        # compute the torque
        tau = self.Kp * (q_act - q_des) + self.Kd * (v_act - v_des)

        return tau
        
    
##################################################################################

# main function
if __name__ == "__main__":

    # model path
    model_file = "./models/hopper.xml"

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
    idx = Hopper_IDX()

    # create controller object
    controller = Controller(model_file)

    # set the initial state
    key_frame_name = "default"
    key_name = "default"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    data.qpos = model.key_qpos[key_id]
    data.qvel = model.key_qvel[key_id]
    
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

            # build full state
            q = data.qpos.copy()
            v = data.qvel.copy()
            state = np.hstack((q, v)).reshape(-1,1)

            # compute the control
            u = controller.compute_input(t_sim, state)

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
