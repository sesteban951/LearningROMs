# Parallel Simulation (RL Based Control)
Run scripts from the project root.

1) Ensure all required packages are installed. 
2) ```parallel_sim.py``` contains a class that can perform three types of rollouts:
    - Zero Input Rollouts: a zero input vector goes into the system.
    - Policy Input Rollouts: load an RL policy and use this for controlling the system. Here, you are 
                             simulating closed loop control. 
    - Random Input Rollouts: sample a random input sequence from a distribution and apply random control during rollout. 
3) See the example usage to produce ```{robot_name}_data.npz``` data.
4) See ```plot_sim_data.py``` to plot some of that roll out data.

## Notes
Using this code in a jupyter notbook ```.ipynb``` might cause some path issues. Perhaps it might be better to produce the data and load in the local ```.npz``` data. 