# standard import
import numpy as np
import time

# local imports
from model_based.biped.simulation import Simulation as BipedSimulation
from model_based.biped.simulation import SimulationConfig as BipedSimulationConfig
from model_based.hopper.simulation import Simulation as HopperSimulation
from model_based.hopper.simulation import SimulationConfig as HopperSimulationConfig

##################################################################################
# MAIN SIMULATION
##################################################################################

if __name__ == "__main__":

    # which robot to simulate
    # robot = "biped" 
    robot = "hopper"


    if robot == "biped":
        # create simulation config
        sim_config = BipedSimulationConfig(
            visualization=True, # visualize or not
            sim_dt=0.002,        # sim time step
            sim_time=10.0,       # total sim time
            cmd_scaling=1.0,     # scaling of the command for joysticking 
            cmd_default=0.2      # default forward velocity command
        )

        # create simulation object
        simulation = BipedSimulation(sim_config)
        simulation.simulate()

    elif robot == "hopper":

        # create simulation config
        sim_config = HopperSimulationConfig(
            visualization=True, # visualize or not
            sim_dt=0.002,        # sim time step
            sim_time=10.0,       # total sim time
            cmd_scaling=1.0,     # scaling of the command for joysticking 
            cmd_default=0.2      # default forward velocity command
        )

        # create simulation object
        simulation = HopperSimulation(sim_config)
        simulation.simulate()

    else:
        raise ValueError(f"Unknown robot type: {robot}")
