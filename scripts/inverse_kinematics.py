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

    # IK, foot position in base frame
    def ik_base(self, p, v):

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
        q = np.array([[q_H], [q_K]])

        # compute the Jacobian
        J = np.zeros((2,2))
        J[0,0] = self.L1 * np.cos(q_H) + self.L2 * np.cos(q_H + q_K)
        J[0,1] = self.L2 * np.cos(q_H + q_K)
        J[1,0] = self.L1 * np.sin(q_H) + self.L2 * np.sin(q_H + q_K)
        J[1,1] = self.L2 * np.sin(q_H + q_K)

        # compute the joint velocities
        qdot = np.linalg.pinv(J) @ v

        return q, qdot

