import numpy as np
import gymnasium as gym
import fixedwing_sim


# Make sure your package is imported to ensure the registration code runs
# import fixedwing_sim  # This should trigger your environment registration

# import src
print(gym.envs.registry.keys())
env = gym.make('CustomEnv-v0')
print("Env created")

# # Now you can use your environment
# observation = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # Replace this with your action selection mechanism
#     observation, reward, done, info = env.step(action)
#     env.render()
# env.close()
