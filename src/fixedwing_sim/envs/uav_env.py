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

class UAMEnv(gymnasium.Env):
    def __init__(self, 
                 backend_interface:OpenGymInterface=None,
                 control_constraints:dict=None,
                 state_constraints:dict=None,
                 render_mode:str=None,
                 render_fps:int=7, 
                 use_random_start:bool=False) -> None:
        super(UAMEnv, self).__init__()
        
        self.backend_interface = backend_interface
        self.constraints = control_constraints
        self.state_constraints = state_constraints
        self.action_space = self.init_attitude_action_space()
        
        self.ego_obs_space = self.init_ego_observation()
        self.observation_space = spaces.Dict(
            {
                "ego": self.ego_obs_space
            }
        )
        # self.observation_space = self.ego_obs_space
        self.use_random_start = use_random_start
        
        ## refactor this 
        self.goal_position = [60, 60, 50]
        self.distance_tolerance = 10
        self.time_step_constant = 1000 #number of steps 
        self.time_limit = self.time_step_constant
        
        init_obs = self.__get_observation()
        
        self.old_distance_to_goal = self.compute_distance_to_goal(
            init_obs['ego'][0],
            init_obs['ego'][1],
            init_obs['ego'][2]
        )
        
    def init_attitude_action_space(self) -> spaces.Box:
        """
        For the first iteration, use continuous action space 
        control roll, pitch, yaw, throttle feed to low level controller
        
        NOTE for the yaw/heading command it must be in degrees for the 
        autopilot to understand it.
        """
        low_action = []
        high_action = []
        
        for k,v in self.constraints.items():
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
        
        high_obs.append(self.state_constraints['max_x'])
        low_obs.append(self.state_constraints['min_x'])
        
        high_obs.append(self.state_constraints['max_y'])
        low_obs.append(self.state_constraints['min_y'])
        
        high_obs.append(self.state_constraints['max_z'])
        low_obs.append(self.state_constraints['min_z'])
        
        high_obs.append(self.state_constraints['max_phi'])
        low_obs.append(self.state_constraints['min_phi'])
        
        high_obs.append(self.state_constraints['max_theta'])
        low_obs.append(self.state_constraints['min_theta'])
        
        high_obs.append(self.state_constraints['max_psi'])
        low_obs.append(self.state_constraints['min_psi'])
        
        high_obs.append(self.state_constraints['max_air_speed'])
        low_obs.append(self.state_constraints['min_air_speed'])
        
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
        roll_norm = action[0]
        pitch_norm = action[1]
        yaw_norm = action[2]
        throttle_norm = action[3]
        
        roll_cmd = self.norm_map_to_real(self.constraints['max_roll'],
                                            self.constraints['min_roll'],
                                            roll_norm)
        
        pitch_cmd = self.norm_map_to_real(self.constraints['max_pitch'],
                                            self.constraints['min_pitch'],
                                            pitch_norm)
        
        yaw_cmd = self.norm_map_to_real(self.constraints['max_yaw'],
                                            self.constraints['min_yaw'],
                                            yaw_norm)
        
        throttle_cmd = self.norm_map_to_real(self.constraints['max_throttle'],  
                                            self.constraints['min_throttle'],
                                            throttle_norm)
        
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd])
        
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
        
        # print("Current position", x, y, z)
                
        if z > 100:
            print("Crashed")
            return -1000, True
        
        if z < 0:
            print("Crashed")
            return -1000, True
        
        if self.time_limit <= 0:
            print("Time limit reached")
            return -1000, True
        
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
        print("current heading", np.rad2deg(yaw), "desired heading", np.rad2deg(los_goal), "error heading", np.rad2deg(error_heading))
        if distance < self.distance_tolerance:
            print("Goal reached")
            return 1000, True
    
        # reward = dot_product - abs(dz)
        #print("Distance", distance, dot_product)
        #reward = np.exp(-0.5 * (distance**2))
        # reward = (1 / (1 + distance)) + dot_product - abs(dz)
        # print("Reward", reward, distance)
        reward = -error_heading        
        #we want to distance to the goal to decrease
        # reward = -distance #self.old_distance_to_goal - distance
        
        #update the old distance to goal
        self.old_distance_to_goal = distance

        return reward, False
        
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Note action input is normalized to [-1, 1] and must be mapped to the
        constraints of the aircraft
        """
        reward = 0
        
        real_action = self.map_normalized_action_to_real_action(action)
        
        self.backend_interface.set_commands(real_action)
        self.backend_interface.run_backend()
        
        step_reward,done = self.get_reward()
        
        reward += step_reward
        
        observation = self.__get_observation()
        info = self.__get_info()
        
        self.time_limit -= 1
        
        if done:
            print("Episode done", reward)
            
        #check if max episode steps reached
        
        return observation, reward, done, False, info
    
    def reset(self, seed=None) -> Any:
        
        if seed is not None or self.use_random_start:
            # self.np_random, seed = seeding.np_random(seed)
            #randomize the initial conditions
            #this needs to be refactored
            min_vel = self.state_constraints['min_air_speed']
            max_vel = self.state_constraints['max_air_speed']
            
            min_heading = self.state_constraints['min_psi']
            max_heading = self.state_constraints['max_psi']
            
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
                self.state_constraints['min_phi'],
                self.state_constraints['max_phi'])
            
            random_pitch = np.random.uniform(
                self.state_constraints['min_theta'],
                self.state_constraints['max_theta'])
            
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
            
            goal_x = 100#np.random.uniform(, 100)
            goal_y = 100#np.random.uniform(-100, 100)
            goal_z = 50 #np.random.uniform(40, 80)
            self.goal_position = [goal_x, goal_y, goal_z]
            
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
    
    # def __render_frame(self) -> None:
    #     pass
    
