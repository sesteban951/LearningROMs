import mujoco
import numpy as np

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
    def fk_feet_in_base(self, q, qdot):

        # unpack the joint angles
        q_H = q[0][0]
        q_K = q[1][0]

        # compute the foot position in base frame
        px =  self.L1 * np.sin(q_H) + self.L2 * np.sin(q_H + q_K)
        pz = -self.L1 * np.cos(q_H) - self.L2 * np.cos(q_H + q_K)
        p = np.array([[px], [pz]])

        # compute the foot velocity in base frame
        J = self.J_feet_in_base(q)
        v = J @ qdot

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
