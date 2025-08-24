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
        thigh_half_length  = model.geom_size[thigh_geom_id, 1]
        shin_half_length   = model.geom_size[shin_geom_id, 1]
        L1 = 2 * thigh_half_length
        L2 = 2 * shin_half_length

        print(f"Thigh length: {L1}, Shin length: {L2}")

if __name__ == "__main__":
    
    ik = InverseKinematics("./models/hotdog_man.xml")
