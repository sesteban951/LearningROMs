# standard import
import numpy as np
import time

# local imports
from model_based.biped.simulation import Simulation as BipedSimulation
from model_based.biped.simulation import SimulationConfig as BipedSimulationConfig
from model_based.hopper.simulation import Simulation as HopperSimulation
from model_based.hopper.simulation import SimulationConfig as HopperSimulationConfig

##################################################################################
# PERFORM SIMULATION
##################################################################################

if __name__ == "__main__":

    # create simulation config
    # sim_config = BipedSimulationConfig(
    #     visualization=True, # visualize or not
    #     sim_dt=0.002,        # sim time step
    #     sim_time=10.0,       # total sim time
    #     cmd_scaling=1.0,     # scaling of the command for joysticking 
    #     cmd_default=0.2      # default forward velocity command
    # )

    # # create simulation object
    # simulation = BipedSimulation(sim_config)

    # # run the simulation
    # t0 = time.time()
    # t_log, q_log, v_log, u_log, c_log, cmd_log = simulation.simulate()
    # t1 = time.time()
    # print(f"Simulation took {t1 - t0:.2f} seconds")

