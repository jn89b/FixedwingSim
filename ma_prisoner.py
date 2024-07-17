import gym
from gym import spaces
import numpy as np
import ray
#from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.ppo import PPO


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def step(self, action):
        # Apply the action and return the next state, reward, done, and info
        state = np.random.random(3)  # Example next state
        reward = 1.0  # Example reward
        done = False  # Example done flag
        info = {}  # Example info
        return state, reward, done, info

    def reset(self):
        # Reset the environment to an initial state
        return np.random.random(3)  # Example initial state

# Register the custom environment
import ray.rllib.env.env_context as env_context
from ray.tune.registry import register_env

def env_creator(env_config):
    return CustomEnv()

register_env("custom_env", env_creator)

# Configure RLlib to use the custom environment with observation and action normalization
config = {
    "env": "custom_env",
    "env_config": {},
    "normalize_actions": True,
    "normalize_observations": True,
    # other configuration options
}

# Initialize Ray and the RLlib trainer
ray.init()
trainer = PPO(config=config)

# Example training loop
for _ in range(10):
    result = trainer.train()
    print(result)

ray.shutdown()
