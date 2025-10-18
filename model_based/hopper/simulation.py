# standard includes
import numpy as np
import time
from dataclasses import dataclass

# pacakge includes
import mujoco 
import glfw

# custom includes 
from .indeces import Hopper_IDX
from .utils import Joy

##################################################################################
# CONTROLLER
##################################################################################

class Controller:

    def __init__(self, model_file):

        # load the model file
        model = mujoco.MjModel.from_xml_path(model_file)

        # get gravity 
        self.gravity = abs(model.opt.gravity[2])

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

    # simple controller
    def compute_input(self, data, vel_cmd):

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
        torso_force, foot_force = self.parse_contact(data)
        foot_in_contact = (foot_force > 1e-3)

        # update the velocity command from joystick
        vx_cmd = vel_cmd
        
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
            theta_des = self.kd_raibert * (vx_cmd - vx_body)

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

        # compute the torque
        tau = np.array([T_body, F_leg])
        
        return tau
    
    # parse contact information
    def parse_contact(self, data):

        # read the sensor values
        torso_force = data.sensordata[self.sid_torso]
        foot_force  = data.sensordata[self.sid_foot]

        return torso_force, foot_force
        

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
        model_file = "./models/hopper.xml"

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
        idx = Hopper_IDX()

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

        # close the glfw window
        if self.config.visualization == True:
            glfw.destroy_window(window)
            glfw.terminate()

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
        cmd_default=0.2      # default forward velocity command
    )

    # create simulation object
    simulation = Simulation(sim_config)

    # run the simulation
    t0 = time.time()
    t_log, q_log, v_log, u_log, c_log, cmd_log = simulation.simulate()
    t1 = time.time()
    print(f"Simulation took {t1 - t0:.2f} seconds")
