# PPO Config Guide
Here we use PPO to train our RL policies. Below are the key hyperparameters you can adjust to optimize training performance. Here is a example for the cart-pole RL environment:
```python
ppo_config = dict(
    num_timesteps=60_000_000,      # total training timesteps
    num_evals=10,                  # number of evaluations
    reward_scaling=0.1,            # reward scale
    episode_length=150,            # max episode length
    normalize_observations=True,   # normalize observations
    unroll_length=10,              # PPO unroll length
    num_minibatches=32,            # PPO minibatches
    num_updates_per_batch=8,       # PPO updates per batch
    discounting=0.97,              # gamma
    learning_rate=1e-3,            # optimizer LR
    clipping_epsilon=0.2,          # PPO clipping epsilon
    entropy_cost=1e-3,             # entropy bonus
    num_envs=1024,                 # parallel envs
    batch_size=1024,               # batch size
    seed=0,                        # RNG seed
)
```

## Run length & evaluation

```num_timesteps=15_000_000```
Total environment steps to generate over the whole run (across all envs). With parallel envs, wall-clock time depends heavily on num_envs and unroll_length.

```num_evals=10```
How many evaluation passes Brax performs during training (spaced across the run). Metrics like eval/episode_reward get logged from these passes. Increasing this gives smoother learning curves but adds a little overhead.

## Reward scaling & episode control

```reward_scaling=0.1```
Multiplicative factor applied to raw rewards before PPO/GAE (Generalized Advantage Estimation). Useful when raw rewards are large or have bad scale—keeps value/advantage magnitudes in a numerically friendly range and stabilizes updates.
Tip: If advantages keep exploding/NaN’ing, try smaller scaling (e.g., 0.01). If learning is sluggish and entropy collapses, sometimes slightly higher helps.

```episode_length=1000```
Max number of steps per episode (truncation). For bipeds, this acts like a horizon cap for return/GAE and sets the longest credit-assignment window. If your gait takes ~1–3 s to stabilize, ensure this is big enough to see multiple strides.

```normalize_observations=True```
Online running mean/variance normalization of observations. Almost always helpful, especially with heterogeneous state vectors (joint angles, velocities, contacts). Keeps gradients well-scaled.

## Rollout collection & batching

```unroll_length=10```
Number of steps each env runs before we stop to compute GAE and do updates (“rollout length”). Longer unrolls improve advantage estimates but increase on-device memory and make updates less frequent.

```num_envs=2048```
How many env replicas step in parallel. More envs ⇒ lower gradient variance and faster wall-clock training (if you have the device memory/throughput). For contact-rich bipeds, large num_envs is great if you can afford it.

```batch_size=2048```
The number of transition samples used per PPO update (before splitting into minibatches).
Important relationship: In Brax PPO, the data collected per update is typically
```python
samples_per_update ≈ num_envs * unroll_length
```
and you usually set
```python
batch_size = num_envs * unroll_length
```
so you use the whole freshly collected batch once per update cycle.

## PPO optimization loop

```num_minibatches=64```
How many chunks you split ```batch_size``` into for each update epoch.
```minibatch_size = batch_size / num_minibatches```.
With ```batch_size=20,480```, ```minibatch_size=320``` (nice). With ```batch_size=2,048```, ```minibatch_size=32```. Very small minibatches can make updates noisy; very large ones can underutilize the device.

```num_updates_per_batch=8```
Number of passes (epochs) you make over that batch each PPO iteration. More epochs = lower bias but higher risk of overfitting / policy collapse if ```clipping_epsilon``` is small or entropy is too low. 3–10 is common; 8 is a strong default.

```discounting=0.97```
The discount γ used for returns/GAE. For 10 ms control (100 Hz), the effective half-life is short; for 20 ms control (50 Hz), 0.97 corresponds to ~33-step time constant. For bipeds, 0.97–0.995 are typical. Higher γ helps long-horizon tasks but can make value learning tougher.

```learning_rate=5e-4```
Optimizer step size (Adam in Brax). With large ```num_envs``` and well-normalized obs/rewards, 3e-4 to 1e-3 is common. If you see oscillatory returns or frequent clip saturation, try 3e-4 or 2e-4.

```clipping_epsilon=0.2```
PPO’s policy ratio clip (|r-1| ≤ ε). Larger ε allows more aggressive policy updates; too large risks instability, too small slows learning. 0.1–0.3 is standard. If you crank up ```num_updates_per_batch```, consider slightly smaller ε.

```entropy_cost=1e-4```
Coefficient on the entropy bonus to encourage exploration. For continuous control with many DoF, typical range is 1e-4 to 1e-2. If your policy collapses to near-deterministic early (entropy plummets), raise this (e.g., 5e-4 or 1e-3). If it never settles, lower it.

## Parallelism, seeding, and reproducibility
```seed=0```
PRNG seed for env/policy initialization. For serious runs, try multiple seeds to estimate variance.


# Overview
The PPO training function has many parameters that can be adjusted to optimize training performance. Below is a description of the key parameters straight from ```ppo.train()``` function.
`environment`: the environment to train

`num_timesteps`: the total number of environment steps to use during training

`max_devices_per_host`: maximum number of chips to use per host process

`wrap_env`: If `True`, wrap the environment for training. Otherwise use the environment as is.

`madrona_backend`: whether to use Madrona backend for training

`augment_pixels`: whether to add image augmentation to pixel inputs

`num_envs`: the number of parallel environments to use for rollouts  
    NOTE: `num_envs` must be divisible by the total number of chips since each
    chip gets `num_envs // total_number_of_chips` environments to roll out  
    NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
    data generated by `num_envs` parallel envs gets used for gradient
    updates over `num_minibatches` of data, where each minibatch has a
    leading dimension of `batch_size`

`episode_length`: the length of an environment episode

`action_repeat`: the number of timesteps to repeat an action

`wrap_env_fn`: a custom function that wraps the environment for training. If not specified, the environment is wrapped with the default training wrapper.

`randomization_fn`: a user-defined callback function that generates randomized environments

`learning_rate`: learning rate for PPO loss

`entropy_cost`: entropy reward for PPO loss, higher values increase entropy of the policy

`discounting`: discounting rate

`unroll_length`: the number of timesteps to unroll in each environment. The PPO loss is computed over 
`unroll_length` timesteps

`batch_size`: the batch size for each minibatch SGD step

`num_minibatches`: the number of times to run the SGD step, each with adifferent minibatch with leading dimension of `batch_size`

`num_updates_per_batch`: the number of times to run the gradient update over all minibatches before 
doing a new environment rollout

`num_resets_per_eval`: the number of environment resets to run between each eval. The environment resets occur on the host

`normalize_observations`: whether to normalize observations

`reward_scaling`: float scaling for reward

`clipping_epsilon`: clipping epsilon for PPO loss

`gae_lambda`: General advantage estimation lambda

`max_grad_norm`: gradient clipping norm value. If `None`, no clipping is done

`normalize_advantage`: whether to normalize advantage estimate

`network_factory`: function that generates networks for policy and value functions

`seed`: random seed

`num_evals`: the number of evals to run during the entire training run. Increasing the number of evals increases total training time

`eval_env`: an optional environment for eval only, defaults to `environment`

`num_eval_envs`: the number of envs to use for evaluation. Each env will run 1 episode, and all envs run in parallel during eval.

`deterministic_eval`: whether to run the eval with a deterministic policy

`log_training_metrics`: whether to log training metrics and callback to `progress_fn`

`training_metrics_steps`: the number of environment steps between logging training metrics

`progress_fn`: a user-defined callback function for reporting/plotting metrics

`policy_params_fn`: a user-defined callback function that can be used for saving custom policy checkpoints or creating policy rollouts and videos

`save_checkpoint_path`: the path used to save checkpoints. If `None`, no checkpoints are saved.

`restore_checkpoint_path`: the path used to restore previous model params

`restore_params`: raw network parameters to restore the `TrainingState` from. These override 
`restore_checkpoint_path`. These parameters can be obtained from the return values of `ppo.train()`.

`restore_value_fn`: whether to restore the value function from the checkpoint or use a random initialization

`run_evals`: if `True`, use the evaluator `num_eval` times to collect distinct eval rollouts. If `False`, `num_eval_envs` and `eval_env` are ignored.  `progress_fn` is then expected to use training metrics.
