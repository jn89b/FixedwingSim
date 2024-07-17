import ray
import os
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from pettingzoo.utils import parallel_to_aec

from src.fixedwing_sim.envs.ma_pursuer_evader_env_v2 import parallel_env
from pettingzoo.test import parallel_api_test

#https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/Ray/rllib_pistonball.py
"""
https://pettingzoo.farama.org/environments/mpe/simple_tag/
"""

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

    return parallel_env(
        n_pursuers=1,
        n_evaders=1,
        pursuer_control_constraints=pursuer_control_constraints,
        evader_control_constraints=evader_control_constraints,
        pursuer_observation_constraints=pursuer_state_constraints,
        evader_observation_constraints=evader_state_constraints,
    )

def env_creator(config):
    return ParallelPettingZooEnv(create_env())


if __name__ == "__main__":
    ray.init()
    
    env_name = "pursuer_evader_env"
    register_env(env_name, env_creator)

    env = create_env()
    #print("environment agents", env.agents)
    parallel_api_test(env, num_cycles=50)    
    print("Parallel API test passed.")        

    config = (
        PPOConfig()
        .environment(env=env_name)
        .env_runners(num_env_runners=4)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    cwd = os.getcwd()
    storage_path = os.path.join(cwd, "ray_results", env_name)
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path= storage_path,
        config=config.to_dict(),
    )

    # print("Training completed.")
