import casadi as ca
import numpy as np
from src.models.Plane import Plane
from opt_control.PlaneOptControl import PlaneOptControl
from jsbsim_backend.aircraft import Aircraft, x8

import gymnasium as gym
from src.sim_interface import OpenGymInterface
from src.conversions import meters_to_feet, mps_to_ktas

from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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


LOAD_MODEL = False
TOTAL_TIMESTEPS = 1000000#100000/2 #
CONTINUE_TRAINING = False

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
    "ic/psi-true-deg": 20.0,
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
    'x_min': -1,
    'x_max': 1,
    'y_min': -1,
    'y_max': 1,
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
    'v_cmd_min':   15,
    'v_cmd_max':   30
}

state_constraints = {
    'x_min': -np.inf,
    'x_max': np.inf,
    'y_min': -np.inf,
    'y_max': np.inf,
    'z_min': 30,
    'z_max': 100,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(15),
    'theta_max': np.deg2rad(15),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 15,
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
               use_random_start = True,
               backend_interface=gym_adapter,
               rl_control_constraints=rl_control_constraints,
               mpc_control_constraints=control_constraints,
               state_constraints=state_constraints)
# check_env(env)
env._max_episode_steps = 1000

obs, info = env.reset()
print("enviroment created")

# action_test = [
#     1.0, # move x direction
#     0.0, # move y direction
#     0.0  # move z direction
# ]

x_history = []
y_history = []
z_history = []
t_history = []

x_ref = []
y_ref = []
z_ref = []
distance_history = []

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/ppo_3/',
                                        name_prefix='rl_model')

if LOAD_MODEL and not CONTINUE_TRAINING:
    model = PPO.load("simple_high_level")
    print("model loaded")
elif LOAD_MODEL and CONTINUE_TRAINING:
    
    model = PPO.load("simple_high_level")
    model.set_env(env)
    print("model loaded and continuing training")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4,
                callback=checkpoint_callback)
    model.save("simple_high_level")
    print("model saved")
    
else:

    model = PPO("MultiInputPolicy", 
                env,
                learning_rate=0.001,
                gamma=0.9,
                # clip_range=0.2,
                n_epochs=10,
                ent_coef=0.001,
                seed=42, 
                verbose=1, tensorboard_log='tensorboard_logs/', 
                device='cuda')
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4, 
                callback=checkpoint_callback)
    model.save("simple_high_level")
    print("model saved")

# #use DDPG
# model = DDPG("MultiInputPolicy",
#              env,
#              learning_rate=0.001,
#              verbose=1,
#              tensorboard_log='tensorboard_logs/',
#              device='cuda')
# model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
# model.save("simple_high_level")
# print("model saved")

env.reset()
N = 150
goal_position = env.goal_position
for i in range(N):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if done:
        print("simulation done")
        obs, info = env.reset()
        break
    x = obs['ego'][0]
    y = obs['ego'][1]
    z = obs['ego'][2]
    x_history.append(x)
    y_history.append(y)
    z_history.append(z)
    distance = np.linalg.norm(np.array([x, y, z]) - goal_position)
    distance_history.append(distance)
    
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

for i in range(len(env.action_history)):
    action = env.action_history[i]
    x_ref.append(action[0])
    y_ref.append(action[1])
    z_ref.append(action[2])

#plot 3d position
fig = plt.figure()
ax = plt.axes(projection='3d')
#plot start
ax.scatter(x_history[0], y_history[0], z_history[0], c='r', 
           marker='o', label='Start')
ax.scatter(goal_position[0], goal_position[1], goal_position[2], c='g',
              marker='o', label='Goal')
ax.plot3D(x_history, y_history, z_history, 'gray')  
ax.plot3D(x_ref, y_ref, z_ref, 'blue')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_zlim(10, 70)
ax.legend()

fig, ax = plt.subplots()
ax.plot(distance_history)
plt.show()