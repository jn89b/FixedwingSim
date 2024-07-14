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
        self.time_history = []
        
    def update_data(self,info_array:np.ndarray):
        self.x.append(info_array[0])
        self.y.append(info_array[1])
        self.z.append(info_array[2])
        self.roll.append(info_array[3])
        self.pitch.append(info_array[4])
        self.yaw.append(info_array[5])
        self.u.append(info_array[6])

    def update_time(self, time:float) -> None:
        self.time_history.append(time)

class SimpleKinematicEnv(gymnasium.Env):
    def __init__(self,
                 control_constraints:dict=None,
                 state_constraints:dict=None,
                 ego_plane:Plane=None,
                 start_state:np.ndarray=None,
                 goal_state:np.ndarray=None,
                 distance_capture:float=10,
                 use_random_start:bool=False,
                 use_pursuers:bool=False,
                 num_pursuers:int=0,
                 pursuer_capture_dist:float=20,
                 pursuer_spawn_dist:float=75) -> None:
        """
        This is a environment that has the kinematic equations of 
        motion for a fixed-wing aircraft.
        See if we can do some basic tasks such as goal following and then
        do some more complex tasks such as pursuer avoidance.
        The real test is to see if we can send it to an 
        autopilot and have it fly.
        """
        super(SimpleKinematicEnv, self).__init__()
        self.control_constraints = control_constraints
        self.state_constraints = state_constraints
        print("state_constraints", state_constraints)
        self.ego_plane = ego_plane
        
        #this start state is the actual state of the aircraft, not the normalized state
        self.original_start_state = start_state
        self.start_state = start_state
        self.goal_state = goal_state
        self.use_pursuers = use_pursuers
        self.num_pursuers = num_pursuers
        self.pursuer_capture_dist = pursuer_capture_dist
        self.pursuer_spawn_dist = pursuer_spawn_dist
        
        self.data_handler = DataHandler()
        
        if self.use_pursuers:
            self.pursuers = self.init_pursuers()
        
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
        self.old_distance_from_goal = self.compute_distance(self.start_state[0:3], 
                                                            self.goal_state)
                
        self.time_constant = 50 #the time constant of the system
        self.time_limit = self.time_constant
        self.dt = self.ego_plane.dt_val
               
        self.observation_space = spaces.Dict({
            'ego': self.ego_obs_space,
            'actual_ego': self.actual_ego_observation()
        })
        

     
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
    
    def init_pursuers(self) -> list[Plane]:
        pursuers = []
        init_count = 0 
        ego_position = self.start_state[0:3]
        ego_heading = self.start_state[5]
        while init_count < self.num_pursuers:
            
            pursuer = Plane()
            pursuer.set_state_space()
            rand_x = np.random.uniform(-self.pursuer_spawn_dist, 
                                       self.pursuer_spawn_dist)
            
            rand_y = np.random.uniform(-self.pursuer_spawn_dist,
                                        self.pursuer_spawn_dist)
            
            rand_z = np.random.uniform(self.state_constraints['z_min'],
                                        self.state_constraints['z_max'])

            rand_x = ego_position[0] + rand_x
            rand_y = ego_position[1] + rand_y
            
            # rand_x = np.random.uniform(
            #     low=self.state_constraints['x_min'],
            #     high=self.state_constraints['x_max'])
            
            # rand_y = np.random.uniform(
            #     low=self.state_constraints['y_min'],
            #     high=self.state_constraints['y_max'])
            
            # rand_z = np.random.uniform(
            #     low=self.state_constraints['z_min'],
            #     high=self.state_constraints['z_max'])
            
            rand_heading = ego_heading + \
                np.random.uniform(-np.deg2rad(45), np.deg2rad(45))
            
            # rand_heading = np.random.uniform(
            #     low=self.state_constraints['psi_min'],
            #     high=self.state_constraints['psi_max'])
                
            rand_airspeed = np.random.uniform(
                low=self.state_constraints['airspeed_min'],
                high=self.state_constraints['airspeed_max'])
                
            pursuer_state = np.array([rand_x, rand_y, rand_z, 
                                      rand_heading, 
                                      0, 
                                      0, 
                                      rand_airspeed])
            
            dist_from_ego = self.compute_distance(
                self.start_state[0:3], pursuer_state[0:3])
            
            if dist_from_ego < self.pursuer_spawn_dist:
                continue
            
            pursuer.set_info(pursuer_state)
            pursuers.append(pursuer)
            init_count += 1
        
        return pursuers
    
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
        
        if self.use_pursuers:
            num_pursuers = self.num_pursuers
            for i in range(num_pursuers):
                #this is for actual distance from pursuer
                high_obs.append(np.inf)
                low_obs.append(-np.inf)
                
                #this is for actual heading difference from pursuer
                high_obs.append(np.inf)
                low_obs.append(-np.inf)
                
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

        if self.use_pursuers:
            for p in self.pursuers:
                #this is the normalized distance from the pursuer
                high_obs.append(2)
                low_obs.append(-1)
                
                #this is the normalized heading difference from the pursuer               
                high_obs.append(1)
                low_obs.append(-1)
        
        obs_space = spaces.Box(low=np.array(low_obs),
                                            high=np.array(high_obs),
                                            dtype=np.float32)
                
        return obs_space
    
    def map_real_to_norm(self, norm_max:float, norm_min:float, real_val:float) -> float:
        """I can probably abstract this out to a utility function"""
        return 2 * (real_val - norm_min) / (norm_max - norm_min) - 1
    
    def norm_map_to_real(self, norm_max:float, norm_min:float, norm_val:float) -> float:
        """
        I can probably abstract this out to a utility function
        """
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
        Get the observation of the environment if we have pursuers
        append the information of the pursuers to the observation space
        """

        # need to append this to the observation space
        # we will need to store the relative position of the pursuers
        # and the heading of the pursuers
        if self.use_pursuers:
            ego_obs = self.norm_start_state
            actual_ego_obs = self.start_state
            for p in self.pursuers:
                pursuer_state = p.get_info()
                
                distance_from_pursuer = self.compute_distance(
                    self.start_state[0:3], pursuer_state[0:3])
                heading_pursuer = pursuer_state[5]
                
                norm_pursuer = self.map_real_observation_to_normalized_observation(
                    pursuer_state)
                
                norm_dist_from_pursuer = self.compute_distance(
                    self.norm_start_state[0:3], norm_pursuer[0:3])

                
                pursuer_norm_heading = self.map_real_to_norm(
                    self.state_constraints['psi_max'],
                    self.state_constraints['psi_min'],
                    heading_pursuer)
                norm_heading_diff = abs(self.norm_start_state[5] - pursuer_norm_heading)
                                    
                ego_obs = np.append(ego_obs, norm_dist_from_pursuer)
                ego_obs = np.append(ego_obs, norm_heading_diff)
                
                actual_ego_obs = np.append(actual_ego_obs, 
                                           distance_from_pursuer)
                
                actual_ego_obs = np.append(actual_ego_obs,
                                             heading_pursuer)
                #make sure type is float32
                ego_obs = ego_obs.astype(np.float32)
                actual_ego_obs = actual_ego_obs.astype(np.float32)
                
            obs = {
                "ego": ego_obs,
                "actual_ego": actual_ego_obs
                }            
        else:
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

        goal_dz_norm = self.map_real_to_norm(self.state_constraints['z_max'],
                                                self.state_constraints['z_min'],
                                                self.goal_state[2])

        z_norm = self.map_real_to_norm(self.state_constraints['z_max'],
                                        self.state_constraints['z_min'],
                                        current_position[2])
        dz = abs(goal_dz_norm - z_norm)
        dz = np.clip(dz, 0, 1)
        
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
            reward = -10 #- (altitude_diff)
            done = True
            return reward , done
        
        done = False
        #we want to get closer to the goal 
        reward = dot_product - dz #time_step_penalty - (distance_from_goal)
        self.old_distance_from_goal = distance_from_goal
        return reward, done 
    
    def __get_evader_reward(self, current_state:np.ndarray) -> tuple:
        reward = 0
        
        current_position = current_state[0:3]        
        if self.time_limit <= 0:
            # print("You Survived!")
            reward = 1000
            done = True
            return reward, done
        
        out_of_bounds_penalty = -1000
        if current_position[0] < self.state_constraints['x_min']:
            reward = out_of_bounds_penalty
            done = True
            return reward, done
        elif current_position[0] > self.state_constraints['x_max']:
            reward = out_of_bounds_penalty
            done = True
            # print("Crashed! Too far right!", current_position)
            return reward, done
        
        if current_position[1] < self.state_constraints['y_min']:
            reward = out_of_bounds_penalty
            done = True
            # print("Crashed! Too far back!", current_position)
            return reward, done
        elif current_position[1] > self.state_constraints['y_max']:
            reward = out_of_bounds_penalty
            done = True
            # print("Crashed! Too far forward!", current_position)
            return reward, done

        if current_position[2] < self.state_constraints['z_min']:
            reward = out_of_bounds_penalty
            done = True
            print("Crashed! Too low!", current_position)
            return reward, done
        
        if current_position[2] > self.state_constraints['z_max']:
            reward = out_of_bounds_penalty
            done = True
            print("Crashed! Too high!", current_position)
            return reward, done
        
        #compute costs from pursuers
        caught = False
        sum_reward = 0
        # we will take the average of the distance and the dot product
        # between the heading of the pursuer and the evader
        unit_vector_ego = np.array([math.cos(current_state[5]),
                                    math.sin(current_state[5])])
        
        #map this as probability of capture?
        dot_products = []
        norm_distances = []
        for p in self.pursuers:
            pursuer_state = p.get_info()
            norm_pursuer = self.map_real_observation_to_normalized_observation(
                pursuer_state)
            
            #clip the pursuer state to the constraints of the environment
            pursuer_state[0] = np.clip(pursuer_state[0],
                                        self.state_constraints['x_min'],
                                        self.state_constraints['x_max'])
            pursuer_state[1] = np.clip(pursuer_state[1],
                                        self.state_constraints['y_min'],
                                        self.state_constraints['y_max'])
            pursuer_state[2] = np.clip(pursuer_state[2],
                                        self.state_constraints['z_min'],
                                        self.state_constraints['z_max'])

            distance_from_pursuer = self.compute_distance(
                current_state[0:3], pursuer_state[0:3])

            norm_dist_from_pursuer = self.compute_distance(
                self.norm_start_state[0:3], norm_pursuer[0:3]) 
                
            heading_pursuer = pursuer_state[5]
            unit_vector_pursuer = np.array([math.cos(heading_pursuer),
                                             math.sin(heading_pursuer)])
            dot_product = np.dot(unit_vector_ego, unit_vector_pursuer)    
            
            norm_distances.append(norm_dist_from_pursuer)
            dot_products.append(dot_product)
            
            #check if within heading
            heading_error = abs(current_state[5] - heading_pursuer)
            
            #wrap the heading error
            if heading_error > np.pi:
                heading_error = 2*np.pi - heading_error
            elif heading_error < -np.pi:
                heading_error = 2*np.pi + heading_error
                
            if distance_from_pursuer < self.pursuer_capture_dist and \
                abs(heading_error) <= np.deg2rad(45):
                #sum_reward += -10 
                caught = True
                #sum_reward += -dot_product + norm_dist_from_pursuer
        
        min_norm_distance = min(norm_distances)
        min_dot_product = min(dot_products) 
        #should I penalize the worst case award? 
        if caught:
            done = True
            reward = -100 #+ min_norm_distance + min_dot_product#(sum_reward/len(self.pursuers))
            return reward, done
        else:
            done = False
            reward = 1 + min_dot_product + min_norm_distance#sum_reward/len(self.pursuers) + 1
            return reward, done
             
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Note action is a normalized action that must be mapped to the real
        Eventually abstract this??
        """
        real_action = self.map_normalized_action_to_real_action(action)
        #set pitch to 0 for now to see what happens
        # real_action[1] = 0
        # next_state = self.ego_plane.rk45(self.start_state, 
        #                                  real_action, 
        #                                  self.dt)
        # to make the control more smooth we will update the decision input
        # every 1 second
        every_one_second = int(1/self.dt)
        current_time_step = self.time_constant - self.time_limit
        next_state = self.start_state
        for i in range(every_one_second):
            next_state = self.ego_plane.rk45(next_state, 
                                             real_action, 
                                             self.dt)
            self.data_handler.update_data(self.start_state)
            actual_sim_time = current_time_step + (i*self.dt)
            self.data_handler.update_time(actual_sim_time)            
             
            # step through and update the pursuers
            if self.use_pursuers:
                #loop through pursuers and run PN guidance for each pursuer
                for p in self.pursuers:
                    pursuer_state = p.get_info()
                    
                    #this is the actual pro nav algorithm
                    dx = next_state[0] - pursuer_state[0]
                    dy = next_state[1] - pursuer_state[1]
                    dz = next_state[2] - pursuer_state[2]
                
                    los = math.atan2(dy, dx)
                    pitch_los = math.atan2(dz, math.sqrt(dx**2 + dy**2))
                    error_pitch = pitch_los - pursuer_state[4]
                    error_los = los - pursuer_state[5]
                                    
                    if error_los > np.deg2rad(20):
                        vel_cmd = 30
                    else:
                        vel_cmd = 15
                    
                    #remember that this is in NED frame
                    pitch_cmd = np.clip(error_pitch, -np.deg2rad(20), np.deg2rad(20))
                    roll_cmd = np.clip(error_los, -np.deg2rad(30), np.deg2rad(30))
                    los = np.clip(los, -np.pi, np.pi)
                    pursuer_action = [
                        roll_cmd*0.2,
                        -pitch_cmd*0.2,
                        error_los,
                        vel_cmd 
                    ]
                    
                    pursuer_next_state = p.rk45(pursuer_state, pursuer_action, self.dt)
                    p.set_info(pursuer_next_state)
                    
                reward, done = self.__get_evader_reward(next_state)
                #use other reward function to avoid pursuers
                # need to also step the pursuers
            else:
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
                
                self.start_state[2] = np.random.uniform(
                    low=self.state_constraints['z_min']+10,
                    high=self.state_constraints['z_max']-10)
                
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
                                
            self.start_state = self.start_state.astype(np.float32)
            
        else:
            self.start_state = self.original_start_state
            self.start_state = self.start_state.astype(np.float32)
            
        if self.use_pursuers:
            self.pursuers = self.init_pursuers()
        
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
    