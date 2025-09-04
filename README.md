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

Install BRAX via instructions: ```https://github.com/google/brax```

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