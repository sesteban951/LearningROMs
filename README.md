# Enviornment

Recommend creating a ```conda``` enviornment. See:
```bash
https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation
```

# Dependencies

## JAX
Install Jax via ```pip``` from these instructions ```https://docs.jax.dev/en/latest/installation.html```, for example:
```bash
pip install --upgrade pip

# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12]"
```

## Mujoco
Used for some CPU Mujoco simulation.

Install Mujoco via ```pip``` from these instructions ```https://mujoco.readthedocs.io/en/stable/python.html```, for example:
```bash
pip install mujoco
```

## Mujoco MJX
Used for some GPU Mujoco simulation.

Install MJX via ```pip``` from these instructions ```https://mujoco.readthedocs.io/en/stable/mjx.html```, for example:
```bash
pip install mujoco-mjx
```

## Brax
Used for reinforcement learning.

Install BRAX via instructions: ```https://github.com/google/brax```, for example:
```bash
pip install brax
```

## Tensorboard
Used for logging RL progress.

For example:
```bash
conda install -y -c conda-forge tensorboard
```
and use it to view logs, for example:
```bash
tensorboard --logdir=./rl/log --port=6006
```
and opening ```http://localhost:6006/``` in your internet browser.

## PyGame
Used for joysticking in some simulations.

Install via:
```bash
conda install -c conda-forge pygame
```
OR
```bash
pip install pygame
```

# Usage
Haven't tested this extensively, but if you want to bypass using the GPU, you can run a script like this:
```bash
# tell JAX to use CPU and ignore all GPUs
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu /home/sergio/miniconda3/envs/env_rom/bin/python   /home/sergio/projects/LearningROMs/data/parallel_sim.py
```
Here you ask to not see any GPUs, tell jax to use the CPU, use the conda env python, and run a python scipt.
