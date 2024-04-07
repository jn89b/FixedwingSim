import casadi as ca
import numpy as np
from src.models.Plane import Plane
from opt_control.PlaneOptControl import PlaneOptControl
from jsbsim_backend.aircraft import Aircraft, x8

import gymnasium as gym
from src.sim_interface import OpenGymInterface
from src.conversions import meters_to_feet, mps_to_ktas


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

"""
Test the MPC imports
"""


def init_mpc_controller(mpc_control_constraints:dict,
                        state_constraints:dict,
                        mpc_params:dict, 
                        plane_model:dict) -> PlaneOptControl:

    plane_mpc = PlaneOptControl(
        control_constraints=mpc_control_constraints,
        state_constraints=state_constraints,
        mpc_params=mpc_params,
        casadi_model=plane_model)
    plane_mpc.init_optimization_problem()
    return plane_mpc




def make_env(init_state_dict, aircraft, control_constraints, state_constraints, mpc_params, mpc_control, rl_control_constraints):
    def _init():
        gym_adapter = OpenGymInterface(init_conditions=init_state_dict,
                                       aircraft=aircraft,
                                       use_mpc=True,
                                       flight_dynamics_sim_hz=50,
                                       mpc_controller=mpc_control)
        
        env = gym.make('MPCEnv',
                       backend_interface=gym_adapter,
                       rl_control_constraints=rl_control_constraints,
                       mpc_control_constraints=control_constraints,
                       state_constraints=state_constraints)
        env._max_episode_steps = 1000
        return env
    return _init


if __name__ == "__main__":
    LOAD_MODEL = False
    TOTAL_TIMESTEPS = 100000 #

    init_state_dict = {
        "ic/u-fps": mps_to_ktas(25),
        "ic/v-fps": 0.0,
        "ic/w-fps": 0.0,
        "ic/p-rad_sec": 0.0,
        "ic/q-rad_sec": 0.0,
        "ic/r-rad_sec": 0.0,
        "ic/h-sl-ft": meters_to_feet(50),
        "ic/long-gc-deg": 0.0,
        "ic/lat-gc-deg": 0.0,
        "ic/psi-true-deg": 20.0,
        "ic/theta-deg": 0.0,
        "ic/phi-deg": 0.0,
        "ic/alpha-deg": 0.0,
        "ic/beta-deg": 0.0,
        "ic/num_engines": 1,
    }

    mpc_params = {
        'N': 10,
        'Q': ca.diag([1.0, 1.0, 1.0, 0, 0, 0.0, 0.0]),
        'R': ca.diag([0.1, 0.1, 0.1, 0.1]),
        'dt': 0.1
    }

    rl_control_constraints = {
        'x_min': -10,
        'x_max': 10,
        'y_min': -10,
        'y_max': 10,
        'z_min': -1,
        'z_max': 1,
    }

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
        'x_min': -np.inf,
        'x_max': np.inf,
        'y_min': -np.inf,
        'y_max': np.inf,
        'z_min': -10,
        'z_max': 70,
        'phi_min':  -np.deg2rad(45),
        'phi_max':   np.deg2rad(45),
        'theta_min':-np.deg2rad(15),
        'theta_max': np.deg2rad(15),
        'psi_min':  -np.pi,
        'psi_max':   np.pi,
        'airspeed_min': 13,
        'airspeed_max': 30
    }


    plane = Plane()
    plane.set_state_space()
    mpc_control = init_mpc_controller(
        mpc_control_constraints=control_constraints,
        state_constraints=state_constraints,
        mpc_params=mpc_params,
        plane_model=plane)

    aircraft = x8

    #get number of cpus
    import os
    num_cpus = os.cpu_count()
    num_envs = num_cpus - 1

    # Use SubprocVecEnv to create multiple environments
    env = SubprocVecEnv([make_env(init_state_dict, aircraft, 
                                control_constraints, 
                                state_constraints, 
                                mpc_params,
                                mpc_control, 
                                rl_control_constraints) for _ in range(num_envs)],
                        start_method='spawn')

    # Your existing PPO model setup (unchanged parts omitted for brevity)...

    model = PPO("MultiInputPolicy", 
                env,
                learning_rate=0.001,
                n_epochs=10,
                ent_coef=0.001,
                verbose=1, tensorboard_log='tensorboard_logs/', 
                device='cuda')

    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
    model.save("simple_high_level_parallelized")
