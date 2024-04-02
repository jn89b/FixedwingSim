# test if imports work
from guidance_control.plane_opt_control import PlaneOptControl
import gymnasium as gym
import numpy as np
import fixedwing_sim # need this to import the gym environment


from gymnasium import spaces, logger
from gymnasium.utils import seeding

from jsbim_backend.aircraft import Aircraft, x8
from jsbim_backend.simulator import FlightDynamics
from sim_interface import CLSimInterface, OpenGymInterface
from guidance_control.autopilot import X8Autopilot
# from src.image_processing import AirSimImages

print(gym.envs.registry.keys())

env = gym.make('CustomEnv-v0')