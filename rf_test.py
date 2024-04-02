import numpy as np
import gymnasium as gym
import fixedwing_sim



# Make sure your package is imported to ensure the registration code runs
# import fixedwing_sim  # This should trigger your environment registration

# import src
print(gym.envs.registry.keys())
env = gym.make('CustomEnv-v0')
print("Env created")
