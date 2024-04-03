import fixedwing_sim
import gymnasium as gym
import numpy as np
from jsbsim_backend.aircraft import Aircraft, x8

from src.sim_interface import OpenGymInterface
from src.conversions import meters_to_feet, mps_to_ktas, ktas_to_mps

#this is used to interface with the stable baselines3 library
#we want to normalize the states and actions
# import stable_baselines
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.env_util import make_vec_env


## Need to define these parameters first before 
# running the test
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

aircraft = x8
gym_adapter = OpenGymInterface(init_conditions=init_state_dict,
                                 aircraft=aircraft,)

max_pitch = np.deg2rad(15)
min_pitch = np.deg2rad(-15)
max_roll = np.deg2rad(45)
min_roll = np.deg2rad(-45)
max_yaw = 180
min_yaw = -180

#these are the control constraints for the aircraft
aircraft_constraints = {
    'max_roll':  max_roll,
    'min_roll':  min_roll,
    'max_pitch': max_pitch,
    'min_pitch': min_pitch,
    'max_yaw':   max_yaw,
    'min_yaw':   min_yaw, 
    'max_throttle': 30, #this is bad, actually using airspeed 
    'min_throttle': 15, #this is bad, actually using airspeed
}

aircraft_state_constraints = {
    'min_x': -np.inf,
    'max_x': np.inf,
    'min_y': -np.inf,
    'max_y': np.inf,
    'min_z': -np.inf,
    'max_z': np.inf,
    'min_phi': min_roll,
    'max_phi': max_roll,
    'min_theta': min_pitch,
    'max_theta': max_pitch,
    'min_psi': -180,
    'max_psi': 180,
    'min_air_speed': 15, # m/s
    'max_air_speed': 30, # m/s
}


#### This is the environment that will be used for training
env = gym.make('UAMEnv-v1', 
               backend_interface=gym_adapter,
               control_constraints=aircraft_constraints,
               state_constraints=aircraft_state_constraints)

print("enviroment created")


# # we can use the DummyVecEnv to wrap the environment and make it 
# # compatible with stable baselines if we want to normalize the states
# new_env = DummyVecEnv([lambda: env])
# norm_env = VecNormalize(new_env, norm_obs=True, 
#                         norm_reward=True)

# print("environment wrapped and normalized")


#### Let's test the environment and see what happens
env.reset()

"""
Positive yaw is to the right
Positive roll is to the right

"""

#mapping normalized action to real action
action_test = [
    1, #roll
    0, #pitch
    1, #yaw 
    0.1, #throttle

]

x_history = []
y_history = []
z_history = []
phi_history = []
theta_history = []
psi_history = []


sim_end_time = 10

N = int(sim_end_time * env.backend_interface.sim.sim_frequency_hz)
for i in range(N):
    # action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action_test)
    if done:
        env.reset()
        
    observation = observation['ego']
    x_history.append(observation[0])
    y_history.append(observation[1])
    z_history.append(observation[2])
    phi_history.append(observation[3])
    theta_history.append(observation[4])
    psi_history.append(observation[5])
    
        
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.close('all')
data_results = env.backend_interface.graph
time_history = data_results.time 
# phi_history = data_results.roll
# theta_history = data_results.pitch
# psi_history = data_results.yaw

#plot 3d position
fig = plt.figure()
ax = plt.axes(projection='3d')
#plot start
ax.scatter(x_history[0], y_history[0], z_history[0], c='r', marker='o')
ax.plot3D(x_history, y_history, z_history, 'gray')  
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')

fig, ax = plt.subplots(3,1)
ax[0].plot(time_history, np.rad2deg(phi_history))
ax[1].plot(time_history, np.rad2deg(theta_history))
ax[2].plot(time_history, np.rad2deg(psi_history)) 


ax[0].set_ylabel('Roll [deg]')
ax[1].set_ylabel('Pitch [deg]')
ax[2].set_ylabel('Yaw [deg]')

fig,ax = plt.subplots()
ax.plot(time_history, data_results.airspeed)

plt.show()
        
