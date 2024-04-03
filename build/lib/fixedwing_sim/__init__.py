import gymnasium
from gymnasium.envs.registration import register


register(
    id='UAMEnv-v1',
    entry_point='fixedwing_sim.envs.uav_env:UAMEnv',
    max_episode_steps=1000,
)

register(
    id='CustomEnv-v0',
    entry_point='fixedwing_sim.envs.custom_env:CustomEnv',
    max_episode_steps=1000,
)

