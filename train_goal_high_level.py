import os
import fixedwing_sim
import gymnasium as gym
import numpy as np
import torch

from jsbsim_backend.aircraft import Aircraft, x8
from src.sim_interface import OpenGymInterface
from src.conversions import meters_to_feet, mps_to_ktas, ktas_to_mps

#this is used to interface with the stable baselines3 library
#we want to normalize the states and actions
# import stable_baselines
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.env_util import make_vec_env
LOAD_MODEL = False
TOTAL_TIMESTEPS = 100000 #
USE_PARALLEL = False

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
    "ic/psi-true-deg": 45,
    "ic/theta-deg": 0.0,
    "ic/phi-deg": 0.0,
    "ic/alpha-deg": 0.0,
    "ic/beta-deg": 0.0,
    "ic/num_engines": 1,
}

aircraft = x8
gym_adapter = OpenGymInterface(init_conditions=init_state_dict,
                               flight_dynamics_sim_hz=60,
                                 aircraft=aircraft,)

max_pitch = np.deg2rad(15)
min_pitch = np.deg2rad(-15)
max_roll = np.deg2rad(45)
min_roll = np.deg2rad(-45)
max_yaw = np.deg2rad(180)
min_yaw = np.deg2rad(-180)

#these are the control constraints for the aircraft
aircraft_constraints = {
    'max_roll':  max_roll,
    'min_roll':  min_roll,
    'max_pitch': max_pitch,
    'min_pitch': min_pitch,
    'max_yaw':   max_yaw,
    'min_yaw':   min_yaw, 
    'max_throttle': 30, #this is bad, actually using airspeed 
    'min_throttle': 12, #this is bad, actually using airspeed
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
    'min_psi': np.deg2rad(-180),
    'max_psi': np.deg2rad(180),
    'min_air_speed': 10, # m/s
    'max_air_speed': 30, # m/s
}


#### This is the environment that will be used for training
env = gym.make('UAMEnv-v1', 
               backend_interface=gym_adapter,
               control_constraints=aircraft_constraints,
               state_constraints=aircraft_state_constraints,
               use_random_start=False)

env._max_episode_steps = 1E5
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
    0.1, #throttle, this is actually airspeed
]
print("running the environment")
print("torch check cuda", torch.cuda.is_available())

print()

## load the model
if LOAD_MODEL:
    # model = DQN.load("dqn_missiongym")
    model = PPO.load("simple_high_level")
else:
    if not USE_PARALLEL:
        check_env(env)
        # model = DQN('MultiInputPolicy', env, 
        #             verbose=1, tensorboard_log='tensorboard_logs/',
        #             device='cuda')
        model = PPO("MultiInputPolicy", 
                    env,
                    learning_rate=0.0001,
                    # clip_range=0.2,
                    # n_epochs=10,
                    ent_coef=0.01,
                    # seed=42, 
                    verbose=1, tensorboard_log='tensorboard_logs/', 
                    device='cuda')
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
        model.save("simple_high_level")
        print("model saved")
    else:
        check_env(env)
        # select number of parallel environments, the optimal choice is usually the number of vCPU.
        N_ENVS = os.cpu_count()
        vec_env = make_vec_env(
            lambda: env,
            n_envs=N_ENVS,
            # this can also be vec_env_cls=SubprocVecEnv, refer to the doc for more info.
            vec_env_cls=DummyVecEnv,
            # vec_env_kwargs=dict(start_method="fork"),
        )
        model = PPO("MultiInputPolicy", 
                    vec_env,
                    learning_rate=0.0001,
                    # clip_range=0.2,
                    # n_epochs=10,
                    ent_coef=0.0001,
                    # seed=42, 
                    verbose=1, tensorboard_log='tensorboard_logs/', 
                    device='cuda')
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
        model.save("simple_high_level")
        # don't forget to close the environment
        vec_env.close()
    
obs, info = env.reset()

N = 1000
x_history = []
y_history = []
z_history = []

for i in range(N):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if done:
        obs, info = env.reset()
    # print(obs)
    # print(rewards)
    # print(done)
    # print(info)
    x = obs['ego'][0]
    y = obs['ego'][1]
    z = obs['ego'][2]
    x_history.append(x)
    y_history.append(y)
    z_history.append(z)
    
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

#plot 3d position
fig = plt.figure()
ax = plt.axes(projection='3d')
#plot start
ax.scatter(x_history[0], y_history[0], z_history[0], c='r', marker='o')
ax.plot3D(x_history, y_history, z_history, 'gray')  
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
plt.show()