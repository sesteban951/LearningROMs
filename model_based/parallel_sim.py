# parallel imports
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Avoid BLAS oversubscription (much faster with many processes)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# standard import
import numpy as np
import time
import random

# local imports
from model_based.biped.simulation import (  # biped
    Simulation as BipedSimulation,
    SimulationConfig as BipedConfig,
)
from model_based.hopper.simulation import ( # hopper
    Simulation as HopperSimulation,
    SimulationConfig as HopperConfig,
)

##################################################################################
# GLOBAL VARS (not really good practice, but needed for multiprocessing)
##################################################################################

# global variables
_SEED = None  # global seed
_ROBOT = None # global robot type 
_SIM = None   # global simulation object

##################################################################################
# PARALLEL SIMULATION UTILS
##################################################################################

# worker initializer
def _init_worker(robot: str, cfg_kwargs: dict, seed: int):
    """
    Runs ONCE per worker process. Create exactly one simulator here and reuse it.
    """
    # set global vars
    global _SIM, _ROBOT, _SEED
    _ROBOT = robot
    _SEED = seed

    # create simulation object
    if robot == "biped":
        _SIM = BipedSimulation(BipedConfig(**cfg_kwargs))
    elif robot == "hopper":
        _SIM = HopperSimulation(HopperConfig(**cfg_kwargs))
    else:
        raise ValueError(f"Unknown robot type: {robot}")
    
    # seed Python & NumPy RNGs in THIS process (for reproducibility)
    # use pid to decorrelate across workers
    pid = os.getpid()
    random.seed(seed ^ pid)
    np.random.seed(seed ^ (pid << 1))

# worker function
def _worker(run_id: int):
    """
    Reuse the per-process simulator; set a per-run command and simulate.
    Saves results; returns a tiny summary.
    """
    global _SIM, _ROBOT, _SEED

    # derive a per-run RNG so runs are reproducible and independent
    rng = np.random.default_rng(_SEED + run_id)

    # choose a command per run (example ranges)
    if _ROBOT == "biped":
        cmd = rng.uniform(-0.5, 0.5)
    elif _ROBOT == "hopper":
        cmd = 0.5 + rng.uniform(-1.0, 1.0)
    else:
        raise ValueError(f"Unknown robot type: {_ROBOT}")

    # mutate only HOT param(s)
    _SIM.config.cmd_default = float(cmd)

    # run one episode
    t_log, q_log, v_log, u_log, c_log, cmd_log = _SIM.simulate()

    # save (ensure dir exists)
    out_dir = "./model_based/data"
    # os.makedirs(out_dir, exist_ok=True)
    # save_path = os.path.join(out_dir, f"{_ROBOT}_run{run_id:06d}.npz")
    # np.savez_compressed(
    #     save_path,
    #     t_log=t_log, q_log=q_log, v_log=v_log, u_log=u_log, c_log=c_log, cmd_log=cmd_log
    # )

    return {"run_id": run_id, "steps": len(t_log), "t_final": float(t_log[-1]) if len(t_log) else 0.0}


##################################################################################
# PERFORM SIMULATION
##################################################################################

if __name__ == "__main__":

    # set random seed for reproducibility
    seed = 1
    # seed = int(time.time())

    # robot type
    # robot = "hopper" 
    robot = "biped"  

    # base configuration (can change per simulation if desired)
    base_cfg = {
        "visualization": False,  # visualize or not
        "sim_dt": 0.002,         # sim time step
        "sim_time": 5.0,         # total sim time
        "cmd_default": 0.2       # default forward velocity command
    }

    # workers and runs
    num_workers = 8   # mp.cpu_count()
    num_runs = 128     # workers will reuse the same sim object for multiple runs

    # safer start for MuJoCo/GL
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Print info
    print(f"Starting {num_runs} simulations with {num_workers} workers...")

    # run simulations in parallel
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,           # <- build ONE sim per worker
        initargs=(robot, base_cfg, seed),   # <- fixed config per worker
    ) as ex:
        futs = [ex.submit(_worker, i) for i in range(num_runs)]
        for f in as_completed(futs):
            try:
                res = f.result()
                print(f"Run {res['run_id']:04d} successful.")
            except Exception as e:
                print("Run failed:", repr(e))

    # print some performance info
    total_time = time.time() - t0
    print(f"Total time for {num_runs} simulations: {total_time:.2f} seconds")
    print(f"Average time per simulation: {total_time / num_runs:.2f} seconds")
