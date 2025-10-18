# standard import
import numpy as np
import time

# parallel imports
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# local imports
from model_based.biped.simulation import Simulation as BipedSimulation
from model_based.biped.simulation import SimulationConfig as BipedConfig
from model_based.hopper.simulation import Simulation as HopperSimulation
from model_based.hopper.simulation import SimulationConfig as HopperConfig

##################################################################################
# PARALLEL SIMULATION UTILS
##################################################################################

def _worker(run_id: int,
            sim_dt: float = 0.002,
            sim_time: float = 10.0,
            cmd_scaling: float = 1.0,
            cmd_default: float = 0.2,
            out_dir: str = "./model_based/hopper") -> str:
    """
    One rollout (headless), then save to a unique NPZ file.
    Note: Your Simulation.simulate() may already save; we save again here
    with a unique name to avoid overwriting.
    """
    cfg = BipedConfig(
        visualization=True,     # headless for speed & safety
        sim_dt=sim_dt,
        sim_time=sim_time,
        cmd_scaling=cmd_scaling,
        cmd_default=cmd_default,
    )
    sim = BipedSimulation(cfg)
    t, q, v, u, c, cmd = sim.simulate()

    # os.makedirs(out_dir, exist_ok=True)
    # out_path = os.path.join(out_dir, f"biped_data_{run_id:06d}.npz")
    # np.savez_compressed(out_path,
    #                     t_log=t, q_log=q, v_log=v, u_log=u, c_log=c, cmd_log=cmd)
    # return out_path


##################################################################################
# PERFORM SIMULATION
##################################################################################

if __name__ == "__main__":

    # Avoid BLAS oversubscription (much faster with many processes)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # workers and runs
    workers = 4
    runs = 16

    # Cross-platform safe
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Warning: mp start method already set")

    # Print info
    print(f"Starting {runs} simulations with {workers} workers...")

    # Create pool and run
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, i) for i in range(runs)]
        for f in as_completed(futs):
            print("Saved:", f.result())