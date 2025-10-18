# standard import
import numpy as np
import time

# local imports
from model_based.biped.simulation import Simulation as BipedSimulation
from model_based.biped.simulation import SimulationConfig as BipedConfig
from model_based.hopper.simulation import Simulation as HopperSimulation
from model_based.hopper.simulation import SimulationConfig as HopperConfig

##################################################################################
# MAIN SIMULATION
##################################################################################

if __name__ == "__main__":

    # which robot to simulate
    robot = "biped"
    # robot = "hopper"

    # Biped simulation
    if robot == "biped":
        # create simulation config
        sim_config = BipedConfig(
            visualization=True, # visualize or not
            sim_dt=0.002,        # sim time step
            sim_time=5.0,       # total sim time
            cmd_scaling=1.0,     # scaling of the command for joysticking 
            cmd_default=0.2      # default forward velocity command
        )

        # create simulation object
        simulation = BipedSimulation(sim_config)
        logs = simulation.simulate()

    # Hopper simulation
    elif robot == "hopper":

        # create simulation config
        sim_config = HopperConfig(
            visualization=True, # visualize or not
            sim_dt=0.002,        # sim time step
            sim_time=10.0,       # total sim time
            cmd_scaling=1.0,     # scaling of the command for joysticking 
            cmd_default=0.2      # default forward velocity command
        )

        # create simulation object
        simulation = HopperSimulation(sim_config)
        logs = simulation.simulate()

    # Unknown robot
    else:
        raise ValueError(f"Unknown robot type: {robot}")
    
    # save the logs
    t_log, q_log, v_log, u_log, c_log, cmd_log = logs
    if robot == "biped":
        np.savez("./model_based/biped/biped_data.npz",
                 t_log=t_log, q_log=q_log, v_log=v_log, u_log=u_log, c_log=c_log, cmd_log=cmd_log)
    elif robot == "hopper":
        np.savez("./model_based/hopper/hopper_data.npz",
                 t_log=t_log, q_log=q_log, v_log=v_log, u_log=u_log, c_log=c_log, cmd_log=cmd_log)

    print("Simulation complete and data saved.")