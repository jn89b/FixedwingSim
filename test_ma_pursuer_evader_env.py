"""
Sanity check for the MA_PursuerEvaderEnv environment.
https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/
"""
import numpy as np
from src.fixedwing_sim.envs.ma_pursuer_evader_env import PursuerEvaderEnv

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
    'x_min': -750, #-np.inf,
    'x_max': 750, #np.inf,
    'y_min': -750, #-np.inf,
    'y_max': 750, #np.inf,
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
    'x_min': -750, #-np.inf,
    'x_max': 750, #np.inf,
    'y_min': -750, #-np.inf,
    'y_max': 750, #np.inf,
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

# Create an instance of the environment
env = PursuerEvaderEnv(
    n_pursuers=2,
    n_evaders=1,
    pursuer_control_constraints=pursuer_control_constraints,
    evader_control_constraints=evader_control_constraints,
    pursuer_observation_constraints=pursuer_state_constraints,
    evader_observation_constraints=evader_state_constraints,
)

# Reset the environment to start a new episode
observations = env.reset()

# Run a simple simulation loop
for step in range(10):  # Run for 10 steps as an example
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # actions = {}
    # for agent in env.agents:
    #     print(f"Agent: {agent}")
    #     actions[agent] = env.action_space(agent).sample()  # Random action
    #     observations[agent] = env.observation_space(agent).sample()
    #     print("Observation:", observations[agent])
    #     print("Action:", actions[agent])
    # Step through the environment with the sampled actions
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # Print the results of the step
    print(f"Step {step + 1}")
    print("Observations:", observations)
    print("Rewards:", rewards)
    print("Terminations:", terminations)
    print("Truncations:", truncations)
    print("Infos:", infos)

    # Check for termination condition
    if all(terminations.values()):
        break

# Close the environment
env.close()
