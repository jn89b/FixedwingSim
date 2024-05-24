import numpy as np
import gymnasium as gym
from src.models.Plane import Plane
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

LOAD_MODEL = False
TOTAL_TIMESTEPS = 1500000#
CONTINUE_TRAINING = True
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
    'x_min': -750, #-np.inf,
    'x_max': 750, #np.inf,
    'y_min': -750, #-np.inf,
    'y_max': 750, #np.inf,
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

start_state = [-50, 0, 45, 0, 0, 0, 15]
start_state = np.array(start_state)

env = gym.make('SimpleKinematicEnv',
               control_constraints=control_constraints,
               state_constraints=state_constraints,
               start_state = start_state,
               goal_state = goal_state,
               use_random_start = True,
               ego_plane=plane)
check_env(env)

n_steps = 550 * 2
n_epochs = 10
batch_size = 100


model_name = "simple_kinematic_ppo_4"
checkpoint_callback = CheckpointCallback(save_freq=10000, 
                                        save_path='./models/'+model_name+'_1/',
                                        name_prefix=model_name)

if LOAD_MODEL and not CONTINUE_TRAINING:
    print("loading model", model_name)
    model = PPO.load(model_name)    
    model.set_env(env)
    print("model loaded")
elif LOAD_MODEL and CONTINUE_TRAINING:
    model = PPO.load(model_name)
    model.set_env(env)
    print("model loaded and continuing training")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4,
                callback=checkpoint_callback)
    model.save(model_name)
    print("model saved")
else:
    #check env 
    # check_env(env)
    model = PPO("MultiInputPolicy", 
                env,
                n_epochs=n_epochs,
                batch_size=batch_size,
                learning_rate=0.00003,
                seed=1, 
                verbose=1, tensorboard_log='tensorboard_logs/', 
                device='cuda')
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4, 
                callback=checkpoint_callback)
    model.save(model_name)
    print("model saved")
    
    