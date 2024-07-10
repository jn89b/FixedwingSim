import casadi as ca
import numpy as np
import gymnasium as gym
from src.models.Plane import Plane
from stable_baselines3.common.env_checker import check_env

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

goal_state = [50, 75, 65]
goal_state = np.array(goal_state)

plane = Plane()
plane.set_state_space()

start_state = [-50, -50, 45, 0, 0, 0, 15]
start_state = np.array(start_state)
env = gym.make('SimpleKinematicEnv',
               control_constraints=control_constraints,
               state_constraints=state_constraints,
               start_state = start_state,
               goal_state = goal_state,
               use_random_start = True,
               use_pursuers = True,
               num_pursuers = 2,
               ego_plane=plane)
check_env(env)

"""
More idiot checks to make sure the environment is working as expected
- Am i mapping the normal actions to the correct control inputs?
- Am i mapping the normal states to the correct states?
- Am i getting the correct reward?
- Am i getting the correct next state?
- Am i getting the correct done flag?
"""

#check if environment is working as expected
N = 50

obs, info = env.reset()
reward_history = []
time_history = []
for i in range(N):
    action = env.action_space.sample()
    # action = np.array([0.0, -0.5, 0, 0])
    obs, reward, done, _, info = env.step(action)
    #print(f"Action: {action}, Reward: {reward}, Done: {done}")
    reward_history.append(reward)
    
import matplotlib.pyplot as plt

history = env.data_handler
pursuers = env.pursuers
time_history = history.time_history
fig, ax = plt.subplots()
ax.plot(reward_history)

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(history.x, history.y, history.z, label='ego')
ax.scatter(history.x[0], history.y[0], history.z[0], 'bo')
ax.plot(goal_state[0], goal_state[1], goal_state[2], 'ro', label='goal')
for p in pursuers:
    ax.plot(p.data_handler.x, p.data_handler.y, p.data_handler.z, label='pursuer')
    ax.scatter(p.data_handler.x[0], p.data_handler.y[0], p.data_handler.z[0], 'go')

ax.legend()

fig,ax = plt.subplots(4,1 )
ax[0].plot(time_history, np.rad2deg(history.roll)[:-1], label='roll')
ax[1].plot(time_history, np.rad2deg(history.pitch)[:-1], label='pitch')
ax[2].plot(time_history, np.rad2deg(history.yaw)[:-1], label='yaw')
ax[3].plot(time_history, history.u[:-1], label='airspeed')

plt.show()

    
    
