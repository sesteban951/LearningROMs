# standard imports
import mujoco
import numpy as np
import math

# joystick
import pygame


##################################################################################

class InverseKinematics:

    def __init__(self, model_file):

        # load the mujoco model 
        model  = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model)

        # identify the links and their lengths (assuming symmetric)
        thigh_name = "left_thigh"
        shin_name = "left_shin"
         
        # get capsule lengths
        thigh_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, thigh_name)
        shin_geom_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, shin_name)
        thigh_half_length  = model.geom_size[thigh_geom_id, 1] # index 1 is half length of capsule
        shin_half_length   = model.geom_size[shin_geom_id, 1]  # index 1 is half length of capsule
        self.L1 = 2 * thigh_half_length
        self.L2 = 2 * shin_half_length
        self.L_tot = self.L1 + self.L2

    # base frame Jacobian, base -> foot
    def J_feet_in_base(self, q):

        # unpack the joint angles
        q_H = q[0][0]
        q_K = q[1][0]

        # compute the Jacobian
        J = np.zeros((2,2))
        J[0,0] = self.L1 * np.cos(q_H) + self.L2 * np.cos(q_H + q_K)
        J[0,1] = self.L2 * np.cos(q_H + q_K)
        J[1,0] = self.L1 * np.sin(q_H) + self.L2 * np.sin(q_H + q_K)
        J[1,1] = self.L2 * np.sin(q_H + q_K)

        return J
    
    # FK, foot state in base frame
    def fk_feet_in_base(self, q_leg, v_leg):

        # unpack the joint angles
        q_H = q_leg[0][0]
        q_K = q_leg[1][0]

        # compute the foot position in base frame
        px =  self.L1 * np.sin(q_H) + self.L2 * np.sin(q_H + q_K)
        pz = -self.L1 * np.cos(q_H) - self.L2 * np.cos(q_H + q_K)
        p = np.array([[px], [pz]])

        # compute the foot velocity in base frame
        J = self.J_feet_in_base(q_leg)
        v = J @ v_leg

        return p, v

    # FK, foot state in world frame
    def fk_feet_in_world(self, q_base, v_base,
                               q_leg,  v_leg):

        # unpack the base state
        p_base_W = q_base[0:2]
        theta_B = q_base[2][0]
        v_base_W = v_base[0:2]
        omega_B = v_base[2][0]

        # build rotation matrices
        R_base_W = np.array([[np.cos(theta_B), -np.sin(theta_B)],
                             [np.sin(theta_B),  np.cos(theta_B)]])
        Omega_skew_B = np.array([[0, -omega_B],
                                 [omega_B, 0]])

        # unpack the joint angles
        q_H = q_leg[0][0]
        q_K = q_leg[1][0]

        # compute the foot position in base frame
        px =  self.L1 * np.sin(q_H) + self.L2 * np.sin(q_H + q_K)
        pz = -self.L1 * np.cos(q_H) - self.L2 * np.cos(q_H + q_K)
        p_B = np.array([[px], [pz]])

        # compute the foot velocity in base frame
        J = self.J_feet_in_base(q_leg)
        v_B = J @ v_leg

        # compute the foot position in world frame
        p = p_base_W + R_base_W @ p_B
        v = v_base_W + R_base_W @ (Omega_skew_B @ p_B + v_B)

        return p, v

    # IK, foot state in base frame
    def ik_feet_in_base(self, p, v):

        # unpack the position
        px = p[0][0]
        pz = p[1][0]
        
        # compute the distance from base to foot
        L = np.sqrt(px**2 + pz**2)

        # quick check to see if the position is reachable
        if L > (self.L1 + self.L2):

            # print warning
            print("IK: {px:.2f}, {pz:.2f} is out of reach".format(px=px, pz=pz))
    
            # project the point to the reachable space
            p = p * (self.L_tot - 1e-3) / L
            px = p[0][0]
            pz = p[1][0]
            L = self.L_tot - 1e-3

            print("\tProjecting to {px:.2f}, {pz:.2f}".format(px=p[0][0], pz=p[1][0]))

        # compute knee angle
        acos_arg = (L**2 - self.L1**2 - self.L2**2) / (- 2.0 * self.L1 * self.L2)
        q_K = np.arccos(acos_arg) - np.pi

        # compute hip angle
        acos_arg = (self.L2**2 - self.L1**2 - L**2) / (- 2.0 * self.L1 * L)
        q_H = np.arctan2(px,-pz) + np.arccos(acos_arg)

        # pack into array
        q_joints = np.array([[q_H], [q_K]])

        # compute the Jacobian
        J = self.J_feet_in_base(q_joints)

        # compute the joint velocities
        qdot_joints = np.linalg.pinv(J) @ v

        return q_joints, qdot_joints

    # IK, robot in world frame 
    # assuming you can contorl everything, but pass in only what you can contorl
    def ik_world(self, p_base_des_W, o_base_des_W, p_left_des_W, p_right_des_W,
                       v_base_des_W, w_base_des_W, v_left_des_W, v_right_des_W):
        
        # compute rotation matrix
        R_base_W = np.array([[np.cos(o_base_des_W), -np.sin(o_base_des_W)],
                             [np.sin(o_base_des_W),  np.cos(o_base_des_W)]])
        
        # compute skew symmetric matrix
        w_base_des_B = w_base_des_W
        Omega_skew_B = np.array([[0, -w_base_des_B],
                                  [w_base_des_B, 0]])

        # compute desired foot positions in base frame
        p_left_des_B = R_base_W.T @ (p_left_des_W - p_base_des_W)
        p_right_des_B = R_base_W.T @ (p_right_des_W - p_base_des_W)

        # compute desired foot velocities in base frame
        v_left_des_B = R_base_W.T @ (v_left_des_W - v_base_des_W) - Omega_skew_B @ p_left_des_B
        v_right_des_B = R_base_W.T @ (v_right_des_W - v_base_des_W) - Omega_skew_B @ p_right_des_B

        # compute IK in base frame for the legs
        q_left, qdot_left = self.ik_feet_in_base(p_left_des_B, v_left_des_B)
        q_right, qdot_right = self.ik_feet_in_base(p_right_des_B, v_right_des_B)

        # build the full state
        q_base = np.array([[p_base_des_W[0][0]],
                           [p_base_des_W[1][0]],
                           [o_base_des_W]])
        v_base = np.array([[v_base_des_W[0][0]],
                           [v_base_des_W[1][0]],
                           [w_base_des_B]])
        
        # build the final state
        q = np.vstack((q_base, q_left, q_right))
        v = np.vstack((v_base, qdot_left, qdot_right))

        return q, v


##################################################################################

# compute a bezier curve based on control points
def bezier_curve(t_pts, ctrl_pts):
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

    # initialize the curves
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

class Joy:

    def __init__(self):

        # initialize the joystick
        pygame.init()
        pygame.joystick.init()

        # check if a joystick is connected
        self.isConnected = False
        if pygame.joystick.get_count() == 0:
            
            # print warning
            print("No joystick connected.")

        # found a joystick
        else:

            # set flag
            self.isConnected = True
            
            # get the first joystick
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

            # print info
            print("Joystick connected: [{}]".format(self.joystick.get_name()))

        # joystick parameters
        self.deadzone = 0.05

        # axes values
        self.LS_X = 0.0
        self.LS_Y = 0.0
        self.RS_X = 0.0
        self.RS_Y = 0.0
        self.RT = 0.0
        self.LT = 0.0

        # button values
        self.A = False
        self.B = False
        self.X = False
        self.Y = False
        self.LB = False
        self.RB = False

        # D-Pad values (another axis)
        self.D_X = False
        self.D_Y = False

    # update the joystick inputs
    def update(self):

        # no joystick connected
        if self.joystick is None:

            print("No joystick connected. Cannot update inputs.")

        # joystick connected
        else:
            
            # process events
            pygame.event.pump()

            # read the axes
            self.LS_X = self.joystick.get_axis(0)   # right is (+)
            self.LS_Y = -self.joystick.get_axis(1)  # up is (+)
            self.RS_X = self.joystick.get_axis(3)   # right is (+)
            self.RS_Y = -self.joystick.get_axis(4)  # up is (+)
            self.LT = (self.joystick.get_axis(2) + 1.0) / 2.0  # unpressed is 0.0, fully pressed is +1.0
            self.RT = (self.joystick.get_axis(5) + 1.0) / 2.0  # unpressed is 0.0, fully pressed is +1.0

            # read the buttons
            self.A = self.joystick.get_button(0)
            self.B = self.joystick.get_button(1)
            self.X = self.joystick.get_button(2)
            self.Y = self.joystick.get_button(3)
            self.LB = self.joystick.get_button(4)
            self.RB = self.joystick.get_button(5)

            # read the D-Pad
            hat = self.joystick.get_hat(0)
            self.D_X = hat[0]  # -1 is left, +1 is right
            self.D_Y = hat[1]  # -1 is down, +1 is up

            # apply deadzone
            if abs(self.LS_X) < self.deadzone:
                self.LS_X = 0.0
            if abs(self.LS_Y) < self.deadzone:
                self.LS_Y = 0.0
            if abs(self.RS_X) < self.deadzone:
                self.RS_X = 0.0
            if abs(self.RS_Y) < self.deadzone:
                self.RS_Y = 0.0

##################################################################################


if __name__ == "__main__":

    joystick = Joy()

    while True:

        joystick.update()

        pygame.time.wait(10)