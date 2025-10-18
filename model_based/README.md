# Single Simulation (Model Based Control)
Run ```main_sim.py``` script from project root for single simulation. Note that you may need to run via:
```bash
python -m model_based.main_sim      # single sim with visuals
```
since the code has been turned into a proper package. 

# Parallel Simulation (Model Based Control)
Run ```parallel_sim.py``` script from the project root via:
```bash
python -m model_based.parallel_sim  # parallel sim without visuals
```

1) Ensure all required packages are installed. 
2) ```parallel_sim.py``` contains a class that can perform three types of rollouts:
3) See the example usage to produce ```.npz``` data.

## Notes
- Use ```plot_sim_data.py``` to plot some of that roll out (or single sim) data.
- Using this code in a jupyter notbook ```.ipynb``` might cause some path issues. Perhaps it might be better to produce the data and load in the local ```.npz``` data. 
