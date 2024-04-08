import math
# import airsim
import gymnasium
import numpy as np

from typing import Tuple, Dict, Any
from gymnasium import spaces, logger
from gymnasium.utils import seeding

from jsbsim_backend.aircraft import Aircraft, x8
from jsbsim_backend.simulator import FlightDynamics
from conversions import feet_to_meters, meters_to_feet, knots_to_mps, mps_to_knots
from sim_interface import CLSimInterface, OpenGymInterface

from guidance_control.autopilot import X8Autopilot
from opt_control.PlaneOptControl import PlaneOptControl
from models.Plane import Plane


from stable_baselines3.common.callbacks import BaseCallback

# class ActionHistoryCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(ActionHistoryCallback, self).__init__(verbose)
#         self.action_history = []

#     def _on_step(self) -> bool:
#         # Assuming `self.training_env` is a VecEnv, we take the last action from the buffer
#         # For a single environment, you might directly use `self.model.action` or modify accordingly
#         actions = self.model.action_buffer
#         self.action_history.extend(actions)
#         return True

class MPCEnv(gymnasium.Env):
    """
    Interfaces with MPC controller and JSBSim simulator
    requires mpc controller to be running in the background
    
    Action Space for the RL agent is:
        - x, y, z position commands
    
    The action space will then be masked to eliminate/penalize any 
    non-feasible commands (considers the aircraft is non-holonomic) it will
    then be passed to the MPC controller to generate the control commands  
    
    States for the RL agent is:
    - x, y, z position
    - roll, pitch, yaw
    
    """
    def __init__(self, 
                 backend_interface:OpenGymInterface=None,
                 rl_control_constraints:dict=None,
                 mpc_control_constraints:dict=None,
                 state_constraints:dict=None,
                 render_mode:str=None,
                 render_fps:int=7, 
                 use_random_start:bool=False) -> None:
        super(MPCEnv, self).__init__()
        
        #check to see if mpc_params is a dictionary
        if not isinstance(mpc_control_constraints, dict):
            raise ValueError("No MPC control constraints provided.")
        

        self.action_history = []
        self.backend_interface = backend_interface
        self.rl_control_constraints = rl_control_constraints
        self.mpc_control_constraints = mpc_control_constraints
        self.state_constraints = state_constraints
        self.action_space = self.init_attitude_action_space()
        print("Action Space: ", self.action_space)
        self.ego_obs_space = self.init_ego_observation()
        self.observation_space = spaces.Dict(
            {
                "ego": self.ego_obs_space
            }
        )

        self.use_random_start = use_random_start
        
        ## refactor this 
        self.goal_position = [50, 40, 50]
        self.distance_tolerance = 5
        self.time_step_constant = 300 #number of steps 
        self.time_limit = self.time_step_constant
        
        init_obs = self.__get_observation()
        
        self.old_distance_to_goal = self.compute_distance_to_goal(
            init_obs['ego'][0],
            init_obs['ego'][1],
            init_obs['ego'][2]
        )

        self.init_counter = 0
    
    def init_attitude_action_space(self) -> spaces.Box:
        """        
        NOTE for the yaw/heading command it must be in degrees for the 
        autopilot to understand it.
        
        Action space will be position commands that will be sent to the 
        MPC controller. 
        
        Action space is as follows:
        pitch, yaw, and airspeed
        """
        high_action = []
        low_action = []
        
        # high_action.append(self.rl_control_constraints['x_max'])
        # low_action.append(self.rl_control_constraints['x_min'])
        
        # high_action.append(self.rl_control_constraints['y_max'])
        # low_action.append(self.rl_control_constraints['y_min'])
        
        # high_action.append(self.rl_control_constraints['z_max'])
        # low_action.append(self.rl_control_constraints['z_min'])
        
        for k,v in self.rl_control_constraints.items():
            if 'max' in k:
                high_action.append(1)
            elif 'min' in k:
                low_action.append(-1)
                
        action_space = spaces.Box(low=np.array(low_action),
                                  high=np.array(high_action),
                                  dtype=np.float32)
        
        return action_space
    
    def init_ego_observation(self) -> spaces.Dict:
        """
        State orders are as follows:
        x, (east) (m)
        y, (north) (m)
        z, (up) (m)
        roll, (rad)
        pitch, (rad)
        yaw, (rad)
        airspeed (m/s)
        """
        high_obs = []
        low_obs = []
        
        high_obs.append(self.state_constraints['x_max'])
        low_obs.append(self.state_constraints['x_min'])
        
        high_obs.append(self.state_constraints['y_max'])
        low_obs.append(self.state_constraints['y_min'])
        
        high_obs.append(self.state_constraints['z_max'])
        low_obs.append(self.state_constraints['z_min'])
        
        high_obs.append(self.state_constraints['phi_max'])
        low_obs.append(self.state_constraints['phi_min'])
        
        high_obs.append(self.state_constraints['theta_max'])
        low_obs.append(self.state_constraints['theta_min'])
        
        high_obs.append(self.state_constraints['psi_max'])
        low_obs.append(self.state_constraints['psi_min'])
        
        high_obs.append(self.state_constraints['airspeed_max'])
        low_obs.append(self.state_constraints['airspeed_min'])
        
        obs_space = spaces.Box(low=np.array(low_obs),
                                            high=np.array(high_obs),
                                            dtype=np.float32)
            
        return obs_space
    
    def map_real_to_norm(self, norm_max:float, norm_min:float, real_val:float) -> float:
        return 2 * (real_val - norm_min) / (norm_max - norm_min) - 1
    
    def norm_map_to_real(self, norm_max:float, norm_min:float, norm_val:float) -> float:
        return norm_min + (norm_max - norm_min) * (norm_val + 1) / 2
    
    def map_normalized_action_to_real_action(self, action:np.ndarray) -> np.ndarray:
        """
        actions are normalized to [-1, 1] and must be mapped to the
        constraints of the aircraft
        Action order: roll, pitch, yaw, throttle
        """
        x_norm = action[0]
        y_norm = action[1]
        z_norm = action[2]
        
        x_cmd = self.norm_map_to_real(self.rl_control_constraints['x_max'],
                                            self.rl_control_constraints['x_min'],
                                            x_norm)
        
        y_cmd = self.norm_map_to_real(self.rl_control_constraints['y_max'],
                                            self.rl_control_constraints['y_min'],
                                            y_norm)
        
        z_cmd = self.norm_map_to_real(self.rl_control_constraints['z_max'],  
                                            self.rl_control_constraints['z_min'],
                                            z_norm)
        
        return np.array([x_cmd, y_cmd, z_cmd])
        
    def __get_observation(self) -> dict:
        # return self.backend_interface.get_observation()
        return {"ego": self.backend_interface.get_observation()}
        
    def __get_info(self) -> dict:
        return {}
    
    def compute_distance_to_goal(self, x, y, z) -> float:
        return np.sqrt((x - self.goal_position[0])**2 + \
            (y - self.goal_position[1])**2 +\
                (z - self.goal_position[2])**2)
    
    def get_reward(self) -> tuple:
        """
        This function will compute the reward for the agent
        based on the current state of the environment
        Very simple right now just get the distance to the goal
        """

        observation = self.__get_observation()
        x = observation['ego'][0]
        y = observation['ego'][1]
        z = observation['ego'][2]
        yaw = observation['ego'][5]
                        
        if z > 100:
            print("Crashed", x, y, z, yaw)
            return -100, True
        
        if z < 0:
            print("Crashed", x, y, z, yaw)
            return -100, True
    
        goal_x = self.goal_position[0]
        goal_y = self.goal_position[1]
        goal_z = self.goal_position[2]

        dz = goal_z - z
        dy = goal_y - y
        dx = goal_x - x
        
        los_goal = np.arctan2(dy, dx)
        #compute error between the current heading and the heading to the goal
        error_heading = abs(los_goal - yaw)        
        
        los_unit = np.array([np.cos(los_goal), np.sin(los_goal)])
        
        # print("obsevation ego yaw", observation['ego'][5])
        ego_unit = np.array([np.cos(yaw), np.sin(yaw)])
        
        dot_product = np.dot(los_unit, ego_unit)
        # print("Dot product", dot_product)
        
        distance = math.sqrt((x - goal_x)**2 + (y - goal_y)**2 + (z - goal_z)**2)
        
        # print("Distance", x, y, z, distance)
        # print("current heading", np.rad2deg(yaw), "desired heading", np.rad2deg(los_goal), "error heading", np.rad2deg(error_heading))
        if distance < self.distance_tolerance:
            print("Goal reached", x, y, z, yaw, distance)
            return 1000, True
    
        # reward = -distance - self.old_distance_to_goal
        # reward = dot_product - abs(dz)
        #print("Distance", distance, dot_product)
        #reward = np.exp(-0.5 * (distance**2))
        reward = (1 / (1 + distance)) #+ dot_product - abs(dz)
        # print("Reward", reward, distance)
        # reward = -error_heading        
        #we want to distance to the goal to decrease
        # reward = -distance #self.old_distance_to_goal - distance
        
        #update the old distance to goal
        # self.old_distance_to_goal = [x, y, z]
        self.old_distance_to_goal = distance 
        
        return reward, False
        
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Note action input is normalized to [-1, 1] and must be mapped to the
        constraints of the aircraft
        """
        reward = 0
        done = False
        real_action = self.map_normalized_action_to_real_action(action)
        observation = self.__get_observation()
        info = self.__get_info()
                
        self.time_limit -= 1
        self.init_counter += 1

        #check if the action is feasible 
        proj_x = observation['ego'][0] + real_action[0]
        proj_y = observation['ego'][1] + real_action[1]
        proj_z = observation['ego'][2] + real_action[2]
        proj_psi = np.arctan2(proj_y, proj_x)
        dz = proj_z - observation['ego'][2]
        dx = proj_x - observation['ego'][0]
        dy = proj_y - observation['ego'][1]
        proj_theta = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        delta_theta = abs(proj_theta - observation['ego'][4])
        delta_psi = abs(proj_psi - observation['ego'][5])
        
        #Wrap delta psi
        if delta_psi > np.pi:
            delta_psi = 2 * np.pi - delta_psi
        elif delta_psi < -np.pi:
            delta_psi = 2 * np.pi + delta_psi
        
        self.action_history.append((proj_x, proj_y, proj_z))
                
        # print("Projected position", proj_x, proj_y, proj_z)
        ## Right now penalize if it does illegal actions do action masking 
        # later on 
        position_action = np.array([proj_x, proj_y, proj_z])
        self.backend_interface.set_commands(position_action)
        time_penalty = -0.01

        # if delta_psi > self.mpc_control_constraints['u_psi_max']:
        #     # print("Delta psi", delta_psi)
        #     reward += 0 + time_penalty
        #     #done = True
        #     # return observation, reward, done, False, info
        
        # # # print("theta_max", np.rad2deg(self.mpc_control_constraints['u_theta_max']))
        # elif delta_theta > self.mpc_control_constraints['u_theta_max']:
        #     # print("observation ego", observation['ego'])
        #     # print("current theta", np.rad2deg(observation['ego'][4]), 
        #     #       "desired theta", np.rad2deg(proj_theta), 
        #     #       "delta theta", np.rad2deg(delta_theta))
        #     reward += 0 + time_penalty
        #     #done = True
        #     #return observation, reward, done, False, info

        # else:
        # self.backend_interface.run_backend()        
        step_reward,done = self.get_reward()
        reward   += step_reward + time_penalty 

        # if self.init_counter % 300 == 0:
        #     print("Step", self.init_counter, self.time_limit)
        #     print("x, y, z", observation['ego'][0], 
        #           observation['ego'][1], 
        #           observation['ego'][2])

        if self.time_limit <= 0:
            #print("Time limit reached", self.time_limit)
            done = True
        
        # if done:
        #     print("Episode done", reward, self.time_limit)
            
        #check if max episode steps reached
        return observation, reward, done, False, info
    
    def reset(self, seed=None) -> Any:
        
        if seed is not None or self.use_random_start:
            # self.np_random, seed = seeding.np_random(seed)
            #randomize the initial conditions
            #this needs to be refactored
            min_vel = self.state_constraints['airspeed_min']
            max_vel = self.state_constraints['airspeed_max']
            
            min_heading = -np.pi
            max_heading = np.pi
            
            random_vel = np.random.uniform(min_vel, max_vel)
            
            random_heading = np.random.uniform(min_heading, 
                                                max_heading)

            #wrap heading to [-pi, pi]
            if random_heading >= np.pi:
                print("Wrap heading")
                random_heading -= 2*np.pi
            elif random_heading <= -np.pi:
                print("Wrap heading")
                random_heading += 2*np.pi
                    
            random_roll = np.random.uniform(
                self.state_constraints['phi_min'],
                self.state_constraints['phi_max'])
            
            random_pitch = np.random.uniform(
                self.state_constraints['theta_min'],
                self.state_constraints['theta_max'])
            
            #move lat and lon to random position within a small radius
            # random_lat_dg = np.random.uniform(-0.0001, 0.0001)
            # random_lon_dg = np.random.uniform(-0.0001, 0.0001)
            random_lat_dg = 0.0
            random_lon_dg = 0.0
            random_alt_ft = meters_to_feet(np.random.uniform(40, 80))
            
            init_state_dict = {
                "ic/u-fps": meters_to_feet(random_vel),
                "ic/v-fps": 0.0,
                "ic/w-fps": 0.0,
                "ic/p-rad_sec": 0.0,
                "ic/q-rad_sec": 0.0,
                "ic/r-rad_sec": 0.0,
                "ic/h-sl-ft": random_alt_ft,
                "ic/long-gc-deg": random_lon_dg,
                "ic/lat-gc-deg": random_lat_dg,
                "ic/psi-true-deg": random_heading,
                "ic/theta-deg": random_pitch,
                "ic/phi-deg": random_roll,
                "ic/alpha-deg": 0.0,
                "ic/beta-deg": 0.0,
                "ic/num_engines": 1,
            }
            
            # goal_x = 100#np.random.uniform(, 100)
            # goal_y = 100#np.random.uniform(-100, 100)
            # goal_z = 50 #np.random.uniform(40, 80)
            # self.goal_position = [goal_x, goal_y, goal_z]
            
            self.backend_interface.init_conditions = init_state_dict
            self.backend_interface.reset_backend(
                init_conditions=init_state_dict)
            # print("Randomized initial conditions", init_state_dict)
                        
        else:    
            self.backend_interface.reset_backend()
        
        observation = self.__get_observation()
        info = self.__get_info()
        self.time_limit = self.time_step_constant

        return observation, info
    
    def render(self, mode:str='human') -> None:
        pass
    
