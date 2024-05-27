import numpy as np
import gymnasium as gym
import random 
from src.models.Plane import Plane
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

LOAD_MODEL = True
TOTAL_TIMESTEPS = 2500000#
CONTINUE_TRAINING = False
COMPARE_MODELS = False

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(20),
    'u_theta_max': np.deg2rad(20),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   15,
    'v_cmd_max':   30
}

state_constraints = {
    'x_min': -500, #-np.inf,
    'x_max': 500, #np.inf,
    'y_min': -500, #-np.inf,
    'y_max': 500, #np.inf,
    'z_min': 30,
    'z_max': 80,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(20),
    'theta_max': np.deg2rad(20),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 12,
    'airspeed_max': 25
}

goal_state = [100, 100, 50]
goal_state = np.array(goal_state)

plane = Plane()
plane.set_state_space()

start_state = [0, 0, 50, 0, 0, 0, 18]
start_state = np.array(start_state)

env = gym.make('SimpleKinematicEnv',
               control_constraints=control_constraints,
               state_constraints=state_constraints,
               start_state = start_state,
               goal_state = goal_state,
               use_random_start = True,
               use_pursuers = True,
               num_pursuers = 2,
               ego_plane=plane,
               )
check_env(env)
num_envs = 10
vec_env = make_vec_env('SimpleKinematicEnv', n_envs=num_envs,
                        env_kwargs={
                            'control_constraints':control_constraints,
                            'state_constraints':state_constraints,
                            'start_state':start_state,
                            'goal_state':goal_state,
                            'use_random_start':True,
                            'use_pursuers':True,
                            'num_pursuers':2,
                            'ego_plane':plane
                        })

n_steps = 550 * 2 // num_envs
n_epochs = 10
batch_size = 100
#model_name = "kinematic_early_avoidance"
model_name = "kinematic_avoidance_ppo_4"
checkpoint_callback = CheckpointCallback(save_freq=10000, 
                                        save_path='./models/'+model_name+'_1/',
                                        name_prefix=model_name)

if LOAD_MODEL and not CONTINUE_TRAINING:
    print("loading model", model_name)
    model = PPO.load(model_name)    
    model.set_env(vec_env)
    print("model loaded")
elif LOAD_MODEL and CONTINUE_TRAINING:
    model = PPO.load(model_name)
    model.set_env(vec_env)
    print("model loaded and continuing training")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, 
                vec_env=vec_env,
                log_interval=4,
                callback=checkpoint_callback)
    model.save(model_name)
    print("model saved")
else:
    #check env 
    # check_env(env)
    model = PPO("MultiInputPolicy", 
                vec_env,
                n_epochs=n_epochs,
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=0.00003,
                seed=1, 
                verbose=1, tensorboard_log='tensorboard_logs/', 
                device='cuda')
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4, 
                callback=checkpoint_callback)
    model.save(model_name)
    print("model saved")
    
done = False
#obs = env.reset()

#choose a purely random seed    
# random_seed_num = 25
random.seed()
rand_num = random.randint(0, 100)
print("rand_num", rand_num)
for i in range(24):
    obs,info = env.reset()
print("start state", obs)

reward_history = []    
while done == False:
    action, _states = model.predict(obs)
    #print("action", action)
    obs, reward, done, _, info = env.step(action)
    reward_history.append(reward)
    # print("reward", reward)
    # print("done", done)
    #time_history.append(env.time)

import matplotlib.pyplot as plt
history = env.data_handler
fig, ax = plt.subplots()
ax.plot(reward_history)

pursuers = env.pursuers

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(history.x, history.y, history.z, label='ego')
ax.scatter(history.x[0], history.y[0], history.z[0], 'bo')
ax.plot(goal_state[0], goal_state[1], goal_state[2], 'ro', label='goal')
for i, pursuer in enumerate(pursuers):
    ax.scatter(pursuer.data_handler.x[0], pursuer.data_handler.y[0], pursuer.data_handler.z[0], label=f'pursuer start {i}')
    ax.plot(pursuer.data_handler.x, pursuer.data_handler.y, pursuer.data_handler.z, label=f'pursuer {i}')
ax.legend()

fig,ax = plt.subplots(4,1 )
ax[0].plot(np.rad2deg(history.roll)[:-1], label='roll')
ax[1].plot(np.rad2deg(history.pitch)[:-1], label='pitch')
ax[2].plot(np.rad2deg(history.yaw)[:-1], label='yaw')
ax[3].plot(history.u[:-1], label='airspeed')

plt.show()