import numpy as np
import matplotlib.pyplot as plt
import ray
import os

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from pettingzoo.test import api_test
from src.fixedwing_sim.envs import ma_pursuer_evader_env_v2
from pettingzoo.test import parallel_api_test


# https://docs.ray.io/en/latest/rllib/rllib-env.html
class CustomCallbacks(DefaultCallbacks):
    # def on_train_result(self, *, trainer, result: dict, **kwargs):
    #     print(f"Training iteration: {result['training_iteration']}")
    #     print(f"Episode reward mean: {result['episode_reward_mean']}")
    #     print(f"Timesteps total: {result['timesteps_total']}")
    #     print(f"Time elapsed: {result['time_total_s']} seconds")
    
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        # This method is called at each step of the episode.
        # env = base_env.get_sub_environments()[0]  # Correct way to get the underlying environment
        # for agent_id in env.agents:
        #     obs = episode.last_raw_obs_for(agent_id)
        #     action = episode.last_action_for(agent_id)
        #     reward = episode.last_reward_for(agent_id)
        #     info = env.infos[agent_id]  # Access the self.infos dictionary
        #     print(f"Episode {episode.episode_id} step {episode.length}:")
        #     print(f"  Agent {agent_id} observations: {obs}")
        #     print(f"Episode {episode.episode_id} step {episode.length}:")
        #     print(f"  Agent {agent_id} observations: {obs}")
        #     print(f"  Agent {agent_id} actions: {action}")
        #     print(f"  Agent {agent_id} rewards: {reward}")
        #     print(f"  Agent {agent_id} infos: {info}")
        pass

def env_creator(args):
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

    env = ma_pursuer_evader_env_v2.parallel_env(
        pursuer_control_constraints=pursuer_control_constraints,
        pursuer_observation_constraints=pursuer_state_constraints,
        evader_control_constraints=evader_control_constraints,
        evader_observation_constraints=evader_state_constraints,
    )

    return env

if __name__ == '__main__':
    ray.init()
    
    # not sure what this does
    env_name = "ma_pursuer_evader_v2"
    # register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    #check if the environment is working
    # test_env = PettingZooEnv(env_creator())
    # parallel_api_test(env_creator(), num_cycles=1000)

    # obs_space = test_env.observation_space
    # act_space = test_env.action_space
    
    config = (
        PPOConfig()
        .environment(env=env_name)
        #set rollouts to a multiple of the environment time limit
        .rollouts(num_rollout_workers=1, 
                #   rollout_fragment_length=5,
                  num_envs_per_worker=1) 
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .debugging(log_level="DEBUG")
        .framework(framework="torch")
        #same for this as well 
        # .training() #
        # .callbacks(CustomCallbacks)
    )

    config_custmev = config.build()
    results = config_custmev.training_step()
            
    # tune.run(
    #     "PPO",
    #     name="ppo_ma_pursuer_evader_v2",
    #     stop={"timesteps_total": 10000},
    #     checkpoint_freq=10,
    #     config=config.to_dict(),
    # )