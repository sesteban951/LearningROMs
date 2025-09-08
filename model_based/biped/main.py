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
        
        # create indexing object
        self.idx = Biped_IDX()

        # create IK object
        self.ik = InverseKinematics(model_file)

    def compute_joint_target(self, t, state):

        # squats
        p_left_des = np.array([-0.20, -0.7]).reshape(2,1)
        p_right_des = np.array([0.25, -0.7]).reshape(2,1)
        v_left_des = np.zeros((2,1))
        v_right_des = np.zeros((2,1))

        p_left_des[1] += 0.025 * np.sin(2 * np.pi * 0.5 * t)
        p_right_des[1] += 0.025 * np.sin(2 * np.pi * 0.5 * t)

        # compute the desired joint positions and velocities
        q_left_des, v_left_des = self.ik.ik_feet_in_base(p_left_des, v_left_des)
        q_right_des, v_right_des = self.ik.ik_feet_in_base(p_right_des, v_right_des)

        q_joints_des = np.vstack((q_left_des, q_right_des)).flatten()
        v_joints_des = np.vstack((v_left_des, v_right_des)).flatten()

        
        # Example usage
        t = np.linspace(0, 1, 100)
        P = np.array([[0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.1, 0.2],
                    [0.2, 0.0],
                    [0.2, 0.0],
                    [0.2, 0.0]])

        y, ydot = self.bezier_curve(t, P)

        # --- 2x2 subplot layout ---
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # (1) Foot trajectory
        axs[0,0].plot(y[:,0], y[:,1], 'b-', label="Bezier curve")
        axs[0,0].scatter(P[:,0], P[:,1], c='r', zorder=5, label="Control points")
        axs[0,0].legend()
        axs[0,0].axis("equal")
        axs[0,0].set_xlabel("x")
        axs[0,0].set_ylabel("z")
        axs[0,0].set_title("Foot trajectory")

        # (2) dx/dt
        axs[0,1].plot(t, ydot[:,0], 'g-', label="dx/dt")
        axs[0,1].legend()
        axs[0,1].set_xlabel("t")
        axs[0,1].set_ylabel("Velocity")
        axs[0,1].set_title("X derivative")

        # (3) dz/dt
        axs[1,0].plot(t, ydot[:,1], 'm-', label="dz/dt")
        axs[1,0].legend()
        axs[1,0].set_xlabel("t")
        axs[1,0].set_ylabel("Velocity")
        axs[1,0].set_title("Z derivative")

        # (4) both derivatives together
        axs[1,1].plot(t, ydot[:,0], 'g-', label="dx/dt")
        axs[1,1].plot(t, ydot[:,1], 'm-', label="dz/dt")
        axs[1,1].legend()
        axs[1,1].set_xlabel("t")
        axs[1,1].set_ylabel("Velocity")
        axs[1,1].set_title("Derivatives comparison")

        plt.tight_layout()
        plt.show()
        exit(0)

        return q_joints_des, v_joints_des

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

    # set the gains (note that torques are normalized and there is a gear ratio, hence small gains)
    Kp = np.array([1.5, 1.5, 1.5, 1.5])
    Kd = np.array([0.04, 0.04, 0.04, 0.04])

    # default base position
    q_base_init = np.array([0.0, 1.0, 0.0])
    v_base_init = np.array([0.0, 0.0, 0.0])
    
    # default joint positions
    default_joints = np.array([0.5,-0.5, 0.1, -0.5])
    q_joints_init = default_joints.copy()
    v_joints_init = np.zeros(4)

    # create controller object
    control = Controller(model_file)

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

        # build full state
        q = data.qpos.copy()
        v = data.qvel.copy()
        state = np.hstack((q, v)).reshape(-1,1)

        # get the current joint positions and velocities
        q_joints = data.qpos[idx.POS.POS_LH : idx.POS.POS_LH + 4].copy()
        v_joints = data.qvel[idx.VEL.VEL_LH : idx.VEL.VEL_LH + 4].copy()

        # control at desired Hz
        if counter % decimation == 0:

            # default desired positions and velocities
            # q_joints_des = default_joints
            # v_joints_des = np.zeros(4)

            # compute the desired joint positions and velocities
            q_joints_des, v_joints_des = control.compute_joint_target(t_sim, state)

            # compute the control
            u = Kp * (q_joints_des - q_joints) + Kd * (v_joints_des - v_joints)

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
