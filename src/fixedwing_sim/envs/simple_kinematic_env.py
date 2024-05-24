import math
import gymnasium
import numpy as np

from typing import Tuple, Dict, Any
from gymnasium import spaces, logger
from gymnasium.utils import seeding

class SimpleKinematicEnv(gymnasium.Env):
    def __init__(self) -> None:
        """
        This is a environment that has the kinematic equations of motion for a fixed-wing aircraft.
        See if we can do some basic tasks such as goal following and then
        do some more complex tasks such as pursuer avoidance.
        The real test is to see if we can send it to an autopilot and have it fly.
        """
        super().__init__(SimpleKinematicEnv, self).__init__()
        
        self.action_space = self.init_action_space()
        self.ego_obs_space = self.init_observation_space()
        
        self.observation_space = spaces.Dict({
            'ego': self.ego_obs_space,
            'actual_ego': self.actual_ego_obs_space
        })        
        
        self.time_constant = 550 #the time constant of the system
        self.time_limit = self.time_constant
        
    def compute_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute the distance between two points.
        """
        return np.linalg.norm(p1 - p2)
    
    