"""
Sanity check for the MA_PursuerEvaderEnv environment.
https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/
"""
import numpy as np
import matplotlib.pyplot as plt
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
    n_pursuers=1,
    n_evaders=1,
    pursuer_control_constraints=pursuer_control_constraints,
    evader_control_constraints=evader_control_constraints,
    pursuer_observation_constraints=pursuer_state_constraints,
    evader_observation_constraints=evader_state_constraints,
)

# Reset the environment to start a new episode
observations = env.reset()

N_steps = 20
# Run a simple simulation loop
for step in range(N_steps):  # Run for 10 steps as an example
    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # do an idiot check to make sure the environment is working as expected
    # let's check make them fly straight
    actions = {}
    for agent, action in env.agents.items():
        print(f"Agent: {agent}")
        #print("Action:", action)
        # this has to be normalized
        current_action = np.array([0.0, 0.0, 0.0, 0])
        actions[agent] = current_action
    
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

# 3D plot of the environment
fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
for name, agent in env.agents.items():
    x = agent.data_handler.x
    y = agent.data_handler.y
    z = agent.data_handler.z
    ax.scatter(x[0], y[0], z[0], label=name + ' start')
    ax.plot(x, y, z, label=name)
    
    
# for agent, obs in observations.items():
#     x, y, z = obs[:3]
#     ax.plot([x], [y], [z], 'o', label=agent)
ax.legend()
plt.show()