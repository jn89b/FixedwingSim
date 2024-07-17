import numpy as np
from pettingzoo.utils import parallel_to_aec
from src.fixedwing_sim.envs.ma_pursuer_evader_env import PursuerEvaderEnv
import gym

def create_env():
    pursuer_control_constraints = {
        'u_phi_min':  -np.deg2rad(45),
        'u_phi_max':   np.deg2rad(45),
        'u_theta_min':-np.deg2rad(10),
        'u_theta_max': np.deg2rad(10),
        'u_psi_min':  -np.deg2rad(45),
        'u_psi_max':   np.deg2rad(45),
        'v_cmd_min':   15,
        'v_cmd_max':   30
    }

    pursuer_state_constraints = {
        'x_min': -750, 
        'x_max': 750,
        'y_min': -750,
        'y_max': 750,
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

    evader_control_constraints = {
        'u_phi_min':  -np.deg2rad(45),
        'u_phi_max':   np.deg2rad(45),
        'u_theta_min':-np.deg2rad(10),
        'u_theta_max': np.deg2rad(10),
        'u_psi_min':  -np.deg2rad(45),
        'u_psi_max':   np.deg2rad(45),
        'v_cmd_min':   15,
        'v_cmd_max':   30
    }

    evader_state_constraints = {
        'x_min': -750,
        'x_max': 750,
        'y_min': -750,
        'y_max': 750,
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

    env = PursuerEvaderEnv(
        n_pursuers=1,
        n_evaders=1,
        pursuer_control_constraints=pursuer_control_constraints,
        evader_control_constraints=evader_control_constraints,
        pursuer_observation_constraints=pursuer_state_constraints,
        evader_observation_constraints=evader_state_constraints,
    )
    return env

from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

def env_creator(_):
    return PettingZooEnv(parallel_to_aec(create_env()))

from ray.tune.registry import register_env

register_env("pursuer_evader_env", env_creator)