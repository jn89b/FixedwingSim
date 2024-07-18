import numpy as np
import matplotlib.pyplot as plt

from src.fixedwing_sim.envs import ma_pursuer_evader_env_v2

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

env = ma_pursuer_evader_env_v2.env(
    pursuer_control_constraints=pursuer_control_constraints,
    pursuer_observation_constraints=pursuer_state_constraints,
    evader_control_constraints=evader_control_constraints,
    evader_observation_constraints=evader_state_constraints,
)
env.reset()
N = 50
i = 0
set_random = False
for agent in env.agent_iter():    
    observation, reward, termination, truncation, info = env.last()
    print("iteration: ", i)
    print(f"Agent: {agent}")
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Termination: {termination}")
    print(f"Truncation: {truncation}")
    print(f"Info: {info}")
    print("\n")  
    
    if set_random:  
        action = env.action_space(agent).sample()
    else:
        #action = np.array([0, 0, 0, 0])
        action = [0, 0, 0, 0]
    
    print("action", action)
    i += 1
    if i == N:
        break
    
    env.step(action)

# 3D plot of the environment
fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
for name, agent in env.planes.items():
    x = agent.data_handler.x
    y = agent.data_handler.y
    z = agent.data_handler.z
    ax.scatter(x[0], y[0], z[0], label=name + ' start')
    ax.plot(x, y, z, label=name)
    
ax.legend()


#plot the rewards
fig, ax = plt.subplots()
for k,v in env.planes.items():
    ax.plot(v.data_handler.rewards, label=k)
plt.show()
