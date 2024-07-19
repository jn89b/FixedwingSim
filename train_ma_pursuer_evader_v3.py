import re
import os 
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.tune.registry import register_env
from src.fixedwing_sim.envs import ma_pursuer_evader_env_v3
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode import Episode

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from typing import Dict, Tuple

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
# Set the working directory in the Ray runtime environment
ray.init(runtime_env={"working_dir": "."})

class CustomCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ) -> None:
        # Initialize accumulators
        episode.user_data["pursuer_rewards"] = []
        episode.user_data["evader_rewards"] = []
        
        # Collect rewards
        for agent_id, agent_info in episode.agent_rewards.items():
            # print(f"Agent ID: {agent_id}")
            # print(f"Agent Info: {agent_info}")
            if "pursuer" in agent_id:
                episode.user_data["pursuer_rewards"].append(agent_info)
            elif "evader" in agent_id:
                episode.user_data["evader_rewards"].append(agent_info)
        
        # Calculate mean rewards
        sum_pursuer_rewards = sum(episode.user_data["pursuer_rewards"])
        sum_evader_rewards = sum(episode.user_data["evader_rewards"])
        
        if len(episode.user_data["pursuer_rewards"]) == 0:
            episode.custom_metrics["mean_pursuer_reward"] = 0
        else:
            episode.custom_metrics["mean_pursuer_reward"] = sum_pursuer_rewards / len(episode.user_data["pursuer_rewards"])
        
        if len(episode.user_data["evader_rewards"]) == 0:
            episode.custom_metrics["mean_evader_reward"] = 0
        else:
            episode.custom_metrics["mean_evader_reward"] = sum_evader_rewards / len(episode.user_data["evader_rewards"])
        
def policy_mapping_fn(agent_id, episode, **kwargs):
    if "pursuer" in agent_id:
        return "pursuer"
    else:
        return "evader"

env_name = "pursuer_evader_v3"  # Change this to the actual environment name
register_env(
    env_name,
    lambda _: ParallelPettingZooEnv(ma_pursuer_evader_env_v3.parallel_env()),
)

if __name__ == "__main__":
    base_config = (
        PPOConfig()
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .environment(env=env_name)
        .env_runners(
            num_env_runners=4,
            num_cpus_per_env_runner=2,
            env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        )
        .multi_agent(
            policies={"pursuer", "evader"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            vf_loss_coeff=0.005,
            train_batch_size=400,
        )
        .callbacks(CustomCallbacks)
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "pursuer": SingleAgentRLModuleSpec(),
                    "evader": SingleAgentRLModuleSpec(),
                }
            ),
        )
    )

    cwd = os.getcwd()
    storage_path = os.path.join(cwd, "ray_results", env_name)    
    exp_name = "tune_analyzing_results"
    tune.run(
        "PPO",
        name="PPO_pursuer_evader",
        stop={"timesteps_total": 50000},
        config=base_config.to_dict(),
        checkpoint_freq=10,
        storage_path=storage_path,
        # storage_path="ray_results",  # Directory to save the results
        # storage_path="ray_results/",  # Directory to save the results
        log_to_file=True,  # Ensure logs are written to files
    )
