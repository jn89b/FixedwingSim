from gymnasium.envs.registration import register

register(
    id='fixedwing_sim/UAMEnv-v0',
    entry_point='fixedwing_sim.src.envs:UAMEnv',
    max_episode_steps=1000,
)