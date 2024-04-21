import casadi as ca
import numpy as np
import time
from src.models.Plane import Plane
from opt_control.PlaneOptControl import PlaneOptControl
from jsbsim_backend.aircraft import Aircraft, x8

import gymnasium as gym
from src.sim_interface import OpenGymInterface
from src.conversions import meters_to_feet, mps_to_ktas
from stable_baselines3.common.env_checker import check_env

"""
Test the MPC imports
"""

def init_mpc_controller(mpc_control_constraints:dict,
                        state_constraints:dict,
                        mpc_params:dict, 
                        plane_model:dict) -> PlaneOptControl:

    plane_mpc = PlaneOptControl(
        control_constraints=mpc_control_constraints,
        state_constraints=state_constraints,
        mpc_params=mpc_params,
        casadi_model=plane_model)
    plane_mpc.init_optimization_problem()
    return plane_mpc


init_state_dict = {
    "ic/u-fps": mps_to_ktas(25),
    "ic/v-fps": 0.0,
    "ic/w-fps": 0.0,
    "ic/p-rad_sec": 0.0,
    "ic/q-rad_sec": 0.0,
    "ic/r-rad_sec": 0.0,
    "ic/h-sl-ft": meters_to_feet(50),
    "ic/long-gc-deg": 0.0,
    "ic/lat-gc-deg": 0.0,
    "ic/psi-true-deg": 90,
    "ic/theta-deg": 0.0,
    "ic/phi-deg": 0.0,
    "ic/alpha-deg": 0.0,
    "ic/beta-deg": 0.0,
    "ic/num_engines": 1,
}

mpc_params = {
    'N': 10,
    'Q': ca.diag([1.0, 1.0, 1.0, 0, 0, 0.0, 0.0]),
    'R': ca.diag([0.1, 0.1, 0.1, 0.1]),
    'dt': 0.1
}

rl_control_constraints = {
    'x_min': -5.0,
    'x_max': 5.0,
    'y_min': -5.0,
    'y_max': 5.0,
    'z_min': -1,
    'z_max': 1,
}

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(10),
    'u_theta_max': np.deg2rad(10),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   20,
    'v_cmd_max':   30
}

state_constraints = {
    'x_min': -np.inf,
    'x_max': np.inf,
    'y_min': -np.inf,
    'y_max': np.inf,
    'z_min': -10,
    'z_max': 70,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(15),
    'theta_max': np.deg2rad(15),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 20,
    'airspeed_max': 30
}

plane = Plane()
plane.set_state_space()
mpc_control = init_mpc_controller(
    mpc_control_constraints=control_constraints,
    state_constraints=state_constraints,
    mpc_params=mpc_params,
    plane_model=plane)

aircraft = x8

gym_adapter = OpenGymInterface(init_conditions=init_state_dict,
                                 aircraft=aircraft,
                                 use_mpc=True,
                                 flight_dynamics_sim_hz=200,
                                 mpc_controller=mpc_control)



#show all registered environments
# print(gym.envs.registry.keys())

#### This is the environment that will be used for training
env = gym.make('MPCEnv', 
               backend_interface=gym_adapter,
               rl_control_constraints=rl_control_constraints,
               mpc_control_constraints=control_constraints,
               state_constraints=state_constraints)
# check_env(env)
env.reset()
print("enviroment created")

action_test = [
    1.0, # move x direction
    0.0, # move y direction
    0.0  # move z direction
]

x_history = []
y_history = []
z_history = []
roll_history = []
pitch_history = []
yaw_history = []

t_history = []
distance_history = []
reward_history = []

x_ref = []
y_ref = []
z_ref = []


goal_position = env.goal_position

N = 500
for i in range(N):
    
    start_time = time.time()
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    end_time = time.time()
    # print(f"Time taken for step {i}: {end_time - start_time}")
    # print(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}")
    x = obs['ego'][0]
    y = obs['ego'][1]
    z = obs['ego'][2]
    roll = obs['ego'][3]
    pitch = obs['ego'][4]
    yaw = obs['ego'][5]
    # print("x: ", x, "y: ", y, "z: ", z)
    # print("\n")
    x_history.append(x)
    y_history.append(y)
    z_history.append(z)
    roll_history.append(np.rad2deg(roll))
    pitch_history.append(np.rad2deg(pitch))
    yaw_history.append(np.rad2deg(yaw))
    
    distance = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2 + (z - goal_position[2])**2)
    distance_history.append(distance)
    reward_history.append(reward)

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

for i in range(N):
    action = env.action_history[i]
    x_ref.append(action[0])
    y_ref.append(action[1])
    z_ref.append(action[2])


#plot 3d position
fig = plt.figure()
ax = plt.axes(projection='3d')
#plot start
ax.scatter(x_history[0], y_history[0], z_history[0], c='r', marker='o')
ax.plot3D(x_history, y_history, z_history, 'gray')  
ax.plot3D(x_ref, y_ref, z_ref, 'blue', alpha=0.5)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
# ax.set_xlim(-10, 100)
# ax.set_ylim(-10, 100)

fig,ax = plt.subplots(2,1)

ax[0].plot(range(N), distance_history)
ax[0].set_title('Distance to goal')
ax[0].set_xlabel('Time [s]') 

ax[1].plot(range(N), reward_history)
ax[1].set_title('Reward')
ax[1].set_xlabel('Time [s]')

fig, ax = plt.subplots(3,1)
ax[0].plot(range(N), roll_history)
ax[0].set_title('Roll')

ax[1].plot(range(N), pitch_history)
ax[1].set_title('Pitch')

ax[2].plot(range(N), yaw_history)
ax[2].set_title('Yaw')

print("Final time of simulation: ", env.backend_interface.sim.get_time())
plt.show()
