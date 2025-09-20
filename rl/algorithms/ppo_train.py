# standard imports
from datetime import datetime
import os

# jax imports
import jax
import functools

# brax imports
from brax.training.agents.ppo import train as ppo

# flax imports
import flax.linen as nn

# for saving results
import pickle

# for logging
from tensorboardX import SummaryWriter

# PPO Training class
class PPO_Train:

    def __init__(self, env, ppo_config):

        # set the environment
        self.env = env

        # TODO: if you want to eventually do domain randomization, you need a env copy
        self.eval_env = env

        # set the ppo config
        self.ppo_config = ppo_config

        # check if the save path exists, if not create it
        self.save_path = "./rl/policy"
        self.check_save_path()

        # print info
        print(f"Created PPO training instance for: [{self.env.__class__.__name__}]")

        # SummaryWriter object for logging
        self.writer = None
        self.times = None

        # final params after training
        self.params = None

    # check if the save file path exists
    def check_save_path(self):

        # check if the directory exists
        if not os.path.exists(self.save_path):

            # notify that the directory does not exist
            print(f"Directory [{self.save_path}] does not exist, creating it now...")

            # create the directory
            os.makedirs(self.save_path)
            print(f"Created directory: [{self.save_path}]")

    # progress function for logging
    def progress(self, num_steps, metrics):

        # try to get eval reward if it exists
        reward = metrics.get("eval/episode_reward", None)
        if reward is not None:
            print(f"  Step: {num_steps:,}, Reward: {reward:.1f}")
        else:
            print(f"  Step: {num_steps}, Reward: N/A")

        self.times.append(datetime.now())

        # Write all metrics to tensorboard
        for key, val in metrics.items():

            # convert jax arrays to floats
            if isinstance(val, jax.Array):
                val = float(val)

            # log to tensorboard
            self.writer.add_scalar(key, val, num_steps)

    # main training function
    def train(self):

        # get current datetime for logging
        robot_name = self.env.robot_name
        current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = f"./rl/log/{robot_name}_log_{current_datetime}"

        # print info
        print(f"Logging to: [{log_file}].")

        # create a SummaryWriter for logging
        self.writer = SummaryWriter(log_file)
        self.times = [datetime.now()]

        # create the training function
        train_fn = functools.partial(
            ppo.train,
            environment=self.env,
            **self.ppo_config
        )

        # train the PPO agent
        #   - (make_policy) makes the policy function
        #   - (params) are the trained parameters
        #   - (metrics) final training metrics
        print("Beginning Training...")
        self.make_policy, self.params, self.metrics = train_fn(
            environment=self.env,
            progress_fn=self.progress,
        )
        print("Training complete.")
        print(f"time to jit: {self.times[1] - self.times[0]}")
        print(f"time to train: {self.times[-1] - self.times[1]}")

        # print the final metrics
        print("Final training metrics:")
        for key, val in self.metrics.items():

            # convert jax arrays to floats
            if isinstance(val, jax.Array):
                val = float(val)

            print(f"  {key}: {val}")

        # save the trained policy
        print("Saving trained policy...")
        save_file = f"{self.save_path}/{robot_name}_policy_{current_datetime}.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(self.params, f)
        print(f"Saved trained policy to: [{save_file}]")
