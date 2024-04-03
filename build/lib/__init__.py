from gymnasium.envs.registration import register

register(
    id='UAMEnv-v0',
    entry_point='src.envs:UAMEnv',
    max_episode_steps=1000,
)

register(
    id='CustomEnv-v0',
    entry_point='fixedwing_sim.envs.custom_env:CustomEnv',
    max_episode_steps=1000,
)

