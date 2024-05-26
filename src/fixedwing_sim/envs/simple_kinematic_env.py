import math
import gymnasium
import numpy as np

from typing import Tuple, Dict, Any
from gymnasium import spaces, logger
from gymnasium.utils import seeding
from src.models.Plane import Plane

class DataHandler():
    def __init__(self) -> None:
        self.x = []
        self.y = []
        self.z = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.u = []
        
    def update_data(self,info_array:np.ndarray):
        self.x.append(info_array[0])
        self.y.append(info_array[1])
        self.z.append(info_array[2])
        self.roll.append(info_array[3])
        self.pitch.append(info_array[4])
        self.yaw.append(info_array[5])
        self.u.append(info_array[6])

class SimpleKinematicEnv(gymnasium.Env):
    def __init__(self,
                 control_constraints:dict=None,
                 state_constraints:dict=None,
                 ego_plane:Plane=None,
                 start_state:np.ndarray=None,
                 goal_state:np.ndarray=None,
                 distance_capture:float=10,
                 use_random_start:bool=False) -> None:
        """
        This is a environment that has the kinematic equations of motion for a fixed-wing aircraft.
        See if we can do some basic tasks such as goal following and then
        do some more complex tasks such as pursuer avoidance.
        The real test is to see if we can send it to an autopilot and have it fly.
        """
        super(SimpleKinematicEnv, self).__init__()
        self.control_constraints = control_constraints
        self.state_constraints = state_constraints
        self.ego_plane = ego_plane
        
        #this start state is the actual state of the aircraft, not the normalized state
        self.original_start_state = start_state
        self.start_state = start_state
        self.goal_state = goal_state
        self.data_handler = DataHandler()
                
        self.norm_start_state = self.map_real_observation_to_normalized_observation(
            self.start_state)
        #check if start state array is the correct size
        if self.start_state.size != self.ego_plane.n_states:
            raise ValueError("Start state must be the same size as the state space of the aircraft.", 
                             self.ego_plane.n_states, " != ", self.start_state.size)
        
        self.distance_capture = distance_capture
        self.use_random_start = use_random_start
        self.action_space = self.init_action_space()
        self.ego_obs_space = self.init_ego_observation()
        self.old_distance_from_goal = self.compute_distance(self.start_state[0:3], self.goal_state)
        
        self.observation_space = spaces.Dict({
            'ego': self.ego_obs_space,
            'actual_ego': self.actual_ego_observation()
        })        
        
        self.time_constant = 550 #the time constant of the system
        self.time_limit = self.time_constant
        self.dt = self.ego_plane.dt_val
        
    def compute_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute the distance between two points.
        """
        return math.dist(p1, p2)
    
    def init_action_space(self) -> spaces.Box:
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
        
        for k,v in self.control_constraints.items():
            if 'max' in k:
                high_action.append(1)
            elif 'min' in k:
                low_action.append(-1)
                
        action_space = spaces.Box(low=np.array(low_action),
                                  high=np.array(high_action),
                                  dtype=np.float32)
        
        return action_space
    
    def actual_ego_observation(self) -> spaces.Dict:
        """
        State orders are as follows:
        x, (east) (m)
        y, (north) (m)
        z, (down) (m)
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
        high_obs = [1, 1, 1, 1, 1, 1, 1]
        low_obs = [-1, -1, -1, -1, -1, -1, -1]


        obs_space = spaces.Box(low=np.array(low_obs),
                                            high=np.array(high_obs),
                                            dtype=np.float32)
        
        return obs_space
    
    def map_real_to_norm(self, norm_max:float, norm_min:float, real_val:float) -> float:
        return 2 * (real_val - norm_min) / (norm_max - norm_min) - 1
    
    def norm_map_to_real(self, norm_max:float, norm_min:float, norm_val:float) -> float:
        return norm_min + (norm_max - norm_min) * (norm_val + 1) / 2
    
    def map_normalized_observation_to_real_observation(self, observation:np.ndarray) -> np.ndarray:
        """
        observations are normalized to [-1, 1] and must be mapped to the
        constraints of the aircraft
        """
        x_norm = observation[0]
        y_norm = observation[1]
        z_norm = observation[2]
        roll_norm = observation[3]
        pitch_norm = observation[4]
        yaw_norm = observation[5]
        v_norm = observation[6]
        
        x = self.norm_map_to_real(self.state_constraints['x_max'],
                                  self.state_constraints['x_min'],
                                  x_norm)
        
        y = self.norm_map_to_real(self.state_constraints['y_max'],
                                  self.state_constraints['y_min'],
                                  y_norm)
        
        z = self.norm_map_to_real(self.state_constraints['z_max'],
                                  self.state_constraints['z_min'],
                                  z_norm)
        
        roll = self.norm_map_to_real(self.state_constraints['phi_max'],
                                     self.state_constraints['phi_min'],
                                     roll_norm)
        
        pitch = self.norm_map_to_real(self.state_constraints['theta_max'],
                                      self.state_constraints['theta_min'],
                                      pitch_norm)
        
        yaw = self.norm_map_to_real(self.state_constraints['psi_max'],
                                    self.state_constraints['psi_min'],
                                    yaw_norm)
        
        v = self.norm_map_to_real(self.state_constraints['airspeed_max'],
                                  self.state_constraints['airspeed_min'],
                                  v_norm)
        
        return np.array([x, y, z, roll, pitch, yaw, v])
    
    def map_real_observation_to_normalized_observation(self, observation:np.ndarray) -> np.ndarray:
        """
        observations are normalized to [-1, 1] and must be mapped to the
        constraints of the aircraft
        """
        x = observation[0]
        y = observation[1]
        z = observation[2]
        roll = observation[3]
        pitch = observation[4]
        yaw = observation[5]
        v = observation[6]
        
        x_norm = self.map_real_to_norm(self.state_constraints['x_max'],
                                       self.state_constraints['x_min'],
                                       x)
        
        y_norm = self.map_real_to_norm(self.state_constraints['y_max'],
                                       self.state_constraints['y_min'],
                                       y)
        
        z_norm = self.map_real_to_norm(self.state_constraints['z_max'],
                                       self.state_constraints['z_min'],
                                       z)
        
        roll_norm = self.map_real_to_norm(self.state_constraints['phi_max'],
                                          self.state_constraints['phi_min'],
                                          roll)
        
        pitch_norm = self.map_real_to_norm(self.state_constraints['theta_max'],
                                           self.state_constraints['theta_min'],
                                           pitch)
        
        yaw_norm = self.map_real_to_norm(self.state_constraints['psi_max'],
                                         self.state_constraints['psi_min'],
                                         yaw)
        
        v_norm = self.map_real_to_norm(self.state_constraints['airspeed_max'],
                                       self.state_constraints['airspeed_min'],
                                       v)
        
        return np.array([x_norm, y_norm, z_norm, roll_norm, pitch_norm, yaw_norm, v_norm])
    
    
    def map_normalized_action_to_real_action(self, action:np.ndarray) -> np.ndarray:
        
        roll_cmd = self.norm_map_to_real(self.control_constraints['u_phi_max'],
                                     self.control_constraints['u_phi_min'],
                                     action[0])
        
        pitch_cmd = self.norm_map_to_real(self.control_constraints['u_theta_max'],
                                      self.control_constraints['u_theta_min'],
                                      action[1])
        
        yaw_cmd = self.norm_map_to_real(self.control_constraints['u_psi_max'],
                                    self.control_constraints['u_psi_min'],
                                    action[2])
        
        airspeed_cmd = self.norm_map_to_real(self.control_constraints['v_cmd_max'],
                                            self.control_constraints['v_cmd_min'],
                                            action[3])
        
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, airspeed_cmd])
    
    
    def __get_observation(self) -> dict:
        """
        Get the observation of the environment
        """
        obs = {
            "ego": self.norm_start_state,
            "actual_ego": self.start_state
        }
        
        return obs
    
    def __get_info(self) -> dict:
        """
        Get the info of the environment
        """
        info = {
            "time_limit": self.time_limit,
            "distance_from_goal": self.compute_distance(self.start_state[0:3], self.goal_state)
        }
        
        return info
    
    def __get_reward(self, current_state:np.ndarray) -> tuple:
        """
        Get the reward and check if done
        """
        current_position = current_state[0:3]
        #distance_from_goal = self.compute_distance(current_position, self.goal_state)
        time_step_penalty = -1
        
        #compute normalized unit vector from current position to goal
        los_angle = math.atan2(self.goal_state[1] - current_position[1],
                                 self.goal_state[0] - current_position[0])
        los_unit_vector = np.array([math.cos(los_angle), 
                                    math.sin(los_angle)])
        
        ego_heading = current_state[5]
        ego_unit_vector = np.array([math.cos(ego_heading), 
                                    math.sin(ego_heading)])
        
        dot_product = np.dot(los_unit_vector, ego_unit_vector)
        
        heading_diff = abs(los_angle - ego_heading)
        distance_from_goal = self.compute_distance(current_position, self.goal_state)
        altitude_diff = abs(self.goal_state[2] - current_position[2])

        if current_position[0] < self.state_constraints['x_min']:
            reward = -10
            done = True
            # print("Crashed! Too far left!", current_position)
            return reward, done
        elif current_position[0] > self.state_constraints['x_max']:
            reward = -10
            done = True
            # print("Crashed! Too far right!", current_position)
            return reward, done
        
        if current_position[1] < self.state_constraints['y_min']:
            reward = -10
            done = True
            # print("Crashed! Too far back!", current_position)
            return reward, done
        elif current_position[1] > self.state_constraints['y_max']:
            reward = -10
            done = True
            # print("Crashed! Too far forward!", current_position)
            return reward, done

        if current_position[2] < self.state_constraints['z_min']:
            reward = -10
            done = True
            #print("Crashed! Too low!", current_position)
            return reward, done
        
        if current_position[2] > self.state_constraints['z_max']:
            reward = -10
            done = True
            # print("Crashed! Too high!", current_position)
            return reward, done
        
        if distance_from_goal <= self.distance_capture:
            print("Goal Reached!")
            done = True
            reward = 1000
            return reward, done 
        
        if self.time_limit <= 0:
            reward = time_step_penalty - (distance_from_goal) #- (altitude_diff)
            done = True
            return reward , done
        
        done = False
        #we want to get closer to the goal 
        reward = dot_product #time_step_penalty - (distance_from_goal)
        self.old_distance_from_goal = distance_from_goal
        return reward, done 
    
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Note action is a normalized action that must be mapped to the real
        """
        
        real_action = self.map_normalized_action_to_real_action(action)
        next_state = self.ego_plane.rk45(self.start_state, 
                                         real_action, 
                                         self.dt)
        reward, done = self.__get_reward(next_state)
        
        self.norm_start_state = self.map_real_observation_to_normalized_observation(
            next_state)
        self.norm_start_state = self.norm_start_state.astype(np.float32)
        # self.start_state = next_state
        #make sure its dtype is float32
        self.start_state = next_state.astype(np.float32)
        observation = self.__get_observation()
        info = self.__get_info()
        
        self.time_limit -= 1
        
        self.data_handler.update_data(self.start_state) 
            
        return observation, reward, done, False, info
        
    def reset(self, seed=None) -> Any:
        super().reset(seed=seed)
        
        #right now see if this works we will randomize the goal location later on
        correct_spawn = False
        if self.use_random_start:
            while correct_spawn == False:
                self.goal_state = np.random.uniform(
                    low=[self.state_constraints['x_min'],
                        self.state_constraints['y_min'],
                        self.state_constraints['z_min']],
                    high=[self.state_constraints['x_max'],
                        self.state_constraints['y_max'],
                        self.state_constraints['z_max']])
                
                self.start_state = self.original_start_state
                
                #randomize the heading of the aircraft
                self.start_state[5] = np.random.uniform(
                    low=self.state_constraints['psi_min'],
                    high=self.state_constraints['psi_max'])

                #randomize the airspeed of the aircraft
                self.start_state[6] = np.random.uniform(
                    low=self.state_constraints['airspeed_min'],
                    high=self.state_constraints['airspeed_max'])
                
                distance_from_goal = self.compute_distance(
                    self.start_state[0:3], self.goal_state)
                
                if distance_from_goal > 50:
                    correct_spawn = True
                                    
            #     self.start_state = np.random.uniform(
            #         low=[self.state_constraints['x_min'],
            #             self.state_constraints['y_min'],
            #             self.state_constraints['z_min'],
            #             0,
            #             0,
            #             #  self.state_constraints['phi_min'],
            #             #  self.state_constraints['theta_min'],
            #             self.state_constraints['psi_min'],
            #             self.state_constraints['airspeed_min']],
                    
            #         high=[self.state_constraints['x_max'],
            #             self.state_constraints['y_max'],
            #             self.state_constraints['z_max'],
            #             0,
            #             0,
            #             #   self.state_constraints['phi_max'],
            #             #   self.state_constraints['theta_max'],
            #             self.state_constraints['psi_max'],
            #             self.state_constraints['airspeed_max']])
            #     distance_from_goal = self.compute_distance(
            #         self.start_state[0:3], self.goal_state)
                
            #     if distance_from_goal > 50:
            #         correct_spawn = True
                
            self.start_state = self.start_state.astype(np.float32)
            
        else:
            self.start_state = self.original_start_state
            self.start_state = self.start_state.astype(np.float32)
            
        self.norm_start_state = self.map_real_observation_to_normalized_observation(
            self.start_state)
        self.norm_start_state = self.norm_start_state.astype(np.float32)
        
        self.time_limit = self.time_constant
        
        observation = self.__get_observation()
        info = self.__get_info()
        
        #reupdate the history log 
        self.data_handler = DataHandler()
        self.data_handler.update_data(self.start_state)
        return observation, info
    
    def render(self, mode='human') -> None:
        pass
    