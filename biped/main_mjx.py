# standard imports
import numpy as np
import time

# mujoco imports
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer

# JAX imports
import jax.numpy as jnp         # standard jax numpy
import jax.lax as lax           # for lax.scan
from jax import jit, vmap       # jit and vmap for speed and vectorization

# custom imports
from indeces import HotdogMan_IDX

##################################################################################

# main function
if __name__ == "__main__":

    # model path
    model_file = "./models/hotdog_man.xml"

    # load the model 
    mj_model = mujoco.MjModel.from_xml_path(model_file)
    mj_data = mujoco.MjData(mj_model)

    # place on GPU to get mjx functionality
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    print(mj_data.qpos, type(mj_data.qpos))
    print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

    # create indexing object
    idx = HotdogMan_IDX()

    # zero everything first
    qpos = jnp.zeros(mjx_model.nq)
    qpos = qpos.at[idx.POS.POS_Z].set(0.9)
    qpos = qpos.at[idx.POS.POS_LH].set(0.5)
    qpos = qpos.at[idx.POS.POS_LK].set(-0.5)
    qpos = qpos.at[idx.POS.POS_RH].set(0.1)
    qpos = qpos.at[idx.POS.POS_RK].set(-0.5)
    qvel = jnp.zeros(mjx_model.nv)

    # set the initial configuration (need to do this after since jnp arrays are immutable)
    mjx_data = mjx_data.replace(qpos=qpos, 
                                qvel=qvel) 

    print("Initial qpos:", mjx_data.qpos)

    # configuration parameters
    t_max = 3.0
    hz_render = 50.0

    # integration parameters
    sim_dt = mjx_model.opt.timestep
    render_dt = 1.0 / hz_render
    num_steps = round(t_max / sim_dt)
    render_decimation = round(render_dt / sim_dt)

    print(f"sim dt: {sim_dt}")
    print(f"render dt: {render_dt}")
    print(f"num_steps: {num_steps}")
    print(f"n_skip: {render_decimation}")

    # pre-compile the step function
    mjx_step_jit = jit(mjx.step)

    def step_fn(carry, _):
        mjx_data, t = carry
        mjx_data = mjx_step_jit(mjx_model, mjx_data)
        state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
        return (mjx_data, t+1), state

    # run scan
    (_, _), states = lax.scan(step_fn, (mjx_data, 0), None, length=num_steps)

    print("States shape:", states.shape)
    print("States dtype:", states.dtype)
    print("States device:", states.devices())

    # move states back to CPU
    states = np.array(states)
    print("States shape:", states.shape)
    print("States dtype:", states.dtype)

    # replay with mujoco viewer
    t_sim_start = time.time()
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        
        # replay the simulation
        for i in range(0, num_steps, render_decimation):

            t0 = time.time()

            print("Wall Time:", time.time() - t_sim_start, "s")

            # update MuJoCo data with MJX state
            mj_data.qpos[:] = states[i, :mj_model.nq]
            mj_data.qvel[:] = states[i, mj_model.nq:]
            mujoco.mj_forward(mj_model, mj_data)

            # render frame
            viewer.sync()

            t1 = time.time()

            # sleep to maintain real-time
            elapsed = t1 - t0
            if elapsed < render_dt:
                time.sleep(render_dt - elapsed)

