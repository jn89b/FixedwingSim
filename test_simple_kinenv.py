import casadi as ca
import numpy as np
import gymnasium as gym
from src.models.Plane import Plane

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(10),
    'u_theta_max': np.deg2rad(10),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   15,
    'v_cmd_max':   30
}

state_constraints = {
    'x_min': -1000, #-np.inf,
    'x_max': 1000, #np.inf,
    'y_min': -1000, #-np.inf,
    'y_max': 1000, #np.inf,
    'z_min': 30,
    'z_max': 100,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(20),
    'theta_max': np.deg2rad(20),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 15,
    'airspeed_max': 30
}

env = gym.make('SimpleKinematicEnv',
               control_constraints=control_constraints,
               state_constraints=state_constraints)

"""
More idiot checks to make sure the environment is working as expected
- Am i mapping the normal actions to the correct control inputs?
- Am i mapping the normal states to the correct states?
- Am i getting the correct reward?
- Am i getting the correct next state?
- Am i getting the correct done flag?
"""


