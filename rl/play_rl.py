# standard imports
import numpy as np
import time

# jax imports
import jax
import jax.numpy as jnp

# brax improts
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training import types

# mujoco imports
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer

# for importing policy
import pickle

# custom imports
from envs.cart_pole_env import CartPoleEnv, CartPoleConfig

#################################################################

if __name__ == "__main__":

    # Load environment
    env = envs.get_environment("cart_pole")

    # Load trained params
    # NOTE: change the path to your saved policy
    params_path = "./rl/policy/cart_pole_policy_2025_08_27_18_56_47.pkl"
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    # Rebuild the policy fn
    #   - In your training script, the first thing returned was `make_policy`
    #   - So we need to rebuild that the same way PPO does:

    # get env flags
    obs_size = env.observation_size
    act_size = env.action_size
    normalize_obs = True
    if normalize_obs == True:
        preprocess_observations_fn = running_statistics.normalize
    else:  
        preprocess_observations_fn = types.identity_observation_preprocessor

    networks = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    
    print("Rebuilding policy function...")

    # Step 1: build fn factory
    inference_fn_factory = ppo_networks.make_inference_fn(networks)

    # Step 2: create an actual policy fn with trained params
    deterministic = True   # for inference, we want deterministic actions
    policy_fn = inference_fn_factory(params=params,
                                     deterministic=deterministic)

    print("Policy function built.")

    print("Jitting Functions.")
    # jit important functions
    policy_fn_jit = jax.jit(lambda obs: policy_fn(obs, jax.random.PRNGKey(0))[0])
    step_fn_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)
    obs_jit = jax.jit(env._compute_obs)

    print("Warm up...")

    # # Warm up State
    # key = jax.random.PRNGKey(0)
    # state = reset_jit(rng=key)
    # for _ in range(5):   # a few steps is enough
    #     key, subkey = jax.random.split(key)
    #     obs = obs_jit(state.pipeline_state)
    #     act, _ = policy_fn_jit(state.obs, subkey)
    #     act, _ = policy_fn_jit(state.obs, subkey)
    #     state = step_fn_jit(state, act)

    print("Warm up complete.")

    # import the mujoco model
    config = CartPoleConfig()
    model_path = config.model_path
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # compute some sim and control parameters
    sim_step_dt = mjx_model.opt.timestep
    sim_steps_per_ctrl = config.physics_steps_per_control_step
    sim_step_counter = 0

    # initial state
    mj_data.qpos = np.zeros(mj_model.nq)
    mj_data.qvel = np.zeros(mj_model.nv)
    mj_data.qpos[0] = 0.0       # cart position
    mj_data.qpos[1] = np.pi   # pole angle

    # wall clock timing variables
    t_sim = 0.0
    wall_start = time.time()
    last_render = 0.0

    # start the interactive simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        # Set camera parameters
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])   # look at x, y, z
        viewer.cam.distance = 3.0                           # distance from lookat
        viewer.cam.elevation = -20.0                        # tilt down/up
        viewer.cam.azimuth = 90.0                           # rotate around lookat

        while viewer.is_running():

            # get the current sim time and state
            t_sim = mj_data.time

            # query controller at the desired rate
            if sim_step_counter % sim_steps_per_ctrl == 0:

                print(f"Sim Time: {t_sim:.3f} s")

                # get current state
                qpos = jnp.array(mj_data.qpos)
                qvel = jnp.array(mj_data.qvel)

                # update the mjx_data
                mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

                # print(f"State: pos {qpos}, vel {qvel}   ")

                # compute the observation
                obs = obs_jit(mjx_data)   # obs is a jax array

                # print(f"Observation: {obs}")

                # compute the action
                # act, _ = policy_fn_jit(obs, subkey)  # act is a jax array
                act = policy_fn_jit(obs)  # act is a jax array
            
                # update the controls
                mj_data.ctrl[:] = np.array(act)

                # print(f"Action: {act}")

            # increment counter
            sim_step_counter += 1

            # step the simulation
            mujoco.mj_step(mj_model, mj_data)

            # sync the viewer
            viewer.sync()

            # sync the sim time with the wall clock time
            wall_elapsed = time.time() - wall_start
            if t_sim > wall_elapsed:
                time.sleep(t_sim - wall_elapsed)

