import math
# import airsim
import gymnasium
import numpy as np

from typing import Tuple, Dict, Any
from gymnasium import spaces, logger
from gymnasium.utils import seeding

from jsbsim_backend.aircraft import Aircraft, x8
from jsbsim_backend.simulator import FlightDynamics
from conversions import feet_to_meters, meters_to_feet, knots_to_mps, mps_to_knots, mps_to_ktas
from sim_interface import CLSimInterface, OpenGymInterface, PursuerInterface
from conversions import local_to_global_position


from guidance_control.autopilot import X8Autopilot
from opt_control.PlaneOptControl import PlaneOptControl
from models.Plane import Plane


from stable_baselines3.common.callbacks import BaseCallback


class PursuerEnv(gymnasium.Env):
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
                 num_pursuers:int=3,
                 start_distance_from_ego:float=50,
                 distance_capture:float=10,
                 pursuer_velocities:np.ndarray=np.ndarray([15, 30]),
                 render_mode:str=None,
                 render_fps:int=7, 
                 use_random_start:bool=False) -> None:
        super(PursuerEnv, self).__init__()
        
        #check to see if mpc_params is a dictionary
        if not isinstance(mpc_control_constraints, dict):
            raise ValueError("No MPC control constraints provided.")
        
        self.action_history = []
        self.backend_interface = backend_interface
        self.rl_control_constraints = rl_control_constraints
        self.mpc_control_constraints = mpc_control_constraints
        self.state_constraints = state_constraints
        
        self.num_pursuers = num_pursuers
        self.start_distance_from_ego = start_distance_from_ego
        self.distance_tolerance = distance_capture
        self.pursuer_velocities = pursuer_velocities
        
        self.action_space = self.init_attitude_action_space()
        self.ego_obs_space = self.init_ego_observation()
        
        self.observation_space = spaces.Dict(
            {
                "ego": self.ego_obs_space,
                "actual_ego": self.actual_ego_observation()
            }
        )
        
        self.use_random_start = use_random_start        
        self.distance_tolerance = 15
        self.time_step_constant = 550    #number of steps
        #need to figure out the relationship between time and steps 
        self.time_limit = self.time_step_constant 
        self.init_counter = 0
        self.distance_history = []
        self.pursuers = self.init_pursuers(num_pursuers)
        
    def compute_distance(self, p1:np.ndarray, p2:np.ndarray) -> float:
        return np.linalg.norm(p1-p2)
        
    def init_pursuers(self, num_pursuers:int) -> list[PursuerInterface]:
        """
        Generate pursuers that will be used to track the ego aircraft
        """
        pursuers  = []
        init_count = 0
        ego_obs = self.backend_interface.get_observation()
        ego_pos = ego_obs[:3]
        ego_heading = ego_obs[5]

        while init_count < num_pursuers:
            rand_x = np.random.uniform(ego_pos[0]-self.start_distance_from_ego, 
                                       ego_pos[0]+self.start_distance_from_ego)
            rand_y = np.random.uniform(ego_pos[1]-self.start_distance_from_ego,
                                        ego_pos[1]+self.start_distance_from_ego)

            rand_z = np.random.uniform(ego_pos[2]-self.start_distance_from_ego/2,
                                        ego_pos[2]+self.start_distance_from_ego/2)
            
            rand_z = 50
            
            pursuer_vel = np.random.uniform(15, 40)
            random_heading = np.random.uniform(-np.pi, np.pi)
            pursuer_geo_pos = local_to_global_position([rand_x, rand_y, rand_z])
            
            random_heading = ego_obs[5]
            pursuer_init_conditions = {
                "ic/u-fps": meters_to_feet(pursuer_vel),
                "ic/v-fps": 0.0,
                "ic/w-fps": 0.0,
                "ic/p-rad_sec": 0.0,
                "ic/q-rad_sec": 0.0,
                "ic/r-rad_sec": 0.0,
                "ic/h-sl-ft": meters_to_feet(pursuer_geo_pos[2]),
                "ic/long-gc-deg": pursuer_geo_pos[0],
                "ic/lat-gc-deg": pursuer_geo_pos[1],
                "ic/psi-true-deg": np.rad2deg(random_heading),
                "ic/theta-deg": 0.0,
                "ic/phi-deg": 0.0,
                "ic/alpha-deg": 0.0,
                "ic/beta-deg": 0.0,
                "ic/num_engines": 1,
            }
            
            #this is bad I don't need to use this 
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
            
            pursuer = PursuerInterface(
                init_conditions=pursuer_init_conditions,
                evader_position=[0, 0, 0],
                control_constraints=control_constraints,
                min_max_vels=[15,30],
                id_number=init_count,
                flight_dynamics_sim_hz=self.backend_interface.flight_dynamics_sim_hz,
            )
            
            pursuer_obs = pursuer.get_observation()
            pursuer_pos = pursuer_obs[:3]
            distance = self.compute_distance(ego_pos, pursuer_pos)
            if (distance > self.start_distance_from_ego) and \
                (distance > self.distance_tolerance):   
                self.distance_history.append(distance)
                pursuers.append(pursuer)
                init_count += 1
            else:
                continue

        #send initial commands to pursuers
        evader_observation = self.backend_interface.get_observation()
        for pursuer in pursuers:
            pursuer_height = pursuer.get_observation()[2]
            turn_cmd, v_cmd = pursuer.pursuit_nav(evader_observation)
            dz = evader_observation[2] - pursuer_height
            pursuer.set_command(np.deg2rad(turn_cmd), v_cmd, pursuer_height+dz)
    
        return pursuers

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
        
        for k,v in self.rl_control_constraints.items():
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
        
        num_pursuer_obs = 2
        for n in range(num_pursuer_obs):
            for i in range(self.num_pursuers):
                low_obs.append(-np.inf)
                high_obs.append(np.inf)
        
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
        
        # high_obs.append(self.state_constraints['x_max'])
        # low_obs.append(self.state_constraints['x_min'])
        
        # high_obs.append(self.state_constraints['y_max'])
        # low_obs.append(self.state_constraints['y_min'])
        
        # high_obs.append(self.state_constraints['z_max'])
        # low_obs.append(self.state_constraints['z_min'])
        
        # high_obs.append(self.state_constraints['phi_max'])
        # low_obs.append(self.state_constraints['phi_min'])
        
        # high_obs.append(self.state_constraints['theta_max'])
        # low_obs.append(self.state_constraints['theta_min'])
        
        # high_obs.append(self.state_constraints['psi_max'])
        # low_obs.append(self.state_constraints['psi_min'])
        
        # high_obs.append(self.state_constraints['airspeed_max'])
        # low_obs.append(self.state_constraints['airspeed_min'])
        
        #add pursuers to the observation space this will be 
        #the distance to the pursuers
        num_pursuer_obs = 2
        for n in range(num_pursuer_obs):
            for i in range(self.num_pursuers):
                #low_obs.append(-np.inf)
                #high_obs.append(np.inf)
                low_obs.append(-1)
                high_obs.append(1)
            
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
        """
        actions are normalized to [-1, 1] and must be mapped to the
        constraints of the aircraft
        Action order: roll, pitch, yaw, throttle
        
        Let's change this to heading command, z cmd, and airspeed
        
        """
        # x_norm = action[0]
        # y_norm = action[1]
        # z_norm = action[2]
        # v_norm = action[3]
        
        # x_cmd = self.norm_map_to_real(self.rl_control_constraints['x_max'],
        #                                     self.rl_control_constraints['x_min'],
        #                                     x_norm)
        
        # y_cmd = self.norm_map_to_real(self.rl_control_constraints['y_max'],
        #                                     self.rl_control_constraints['y_min'],
        #                                     y_norm)
        
        # z_cmd = self.norm_map_to_real(self.rl_control_constraints['z_max'],  
        #                                     self.rl_control_constraints['z_min'],
        #                                     z_norm)
        
        # v_cmd = self.norm_map_to_real(self.rl_control_constraints['v_cmd_min'],
        #                               self.rl_control_constraints['v_cmd_max'],
        #                               v_norm)
            
        heading_cmd = self.norm_map_to_real(self.rl_control_constraints['heading_cmd_max'],
                                            self.rl_control_constraints['heading_cmd_min'],
                                            action[0])
        
        z_cmd = self.norm_map_to_real(self.rl_control_constraints['z_max'],
                                      self.rl_control_constraints['z_min'],
                                      action[1])
        
        v_cmd = self.norm_map_to_real(self.rl_control_constraints['v_cmd_max'],
                                      self.rl_control_constraints['v_cmd_min'],
                                      action[2])        
        
        return np.array([heading_cmd, z_cmd, v_cmd])
        
    def __get_observation(self) -> dict:
        ego_obs = self.backend_interface.get_observation()
        norm_ego_obs = self.map_real_observation_to_normalized_observation(ego_obs)
        #update observation space with pursuer distance
        for pursuer in self.pursuers:
            pursuer_obs = pursuer.get_observation()
            norm_pursuer_obs = self.map_real_observation_to_normalized_observation(pursuer_obs)
            norm_pursuer_pos = norm_pursuer_obs[:3]
            
            norm_distance = self.compute_distance(norm_ego_obs[:3], norm_pursuer_pos)
            norm_ego_obs = np.append(norm_ego_obs, norm_distance)
            
            pursuer_heading = norm_pursuer_obs[5]  
            heading_diff = abs(norm_ego_obs[5] - pursuer_heading)
            
            norm_ego_obs = np.append(norm_ego_obs, heading_diff)
            
            
            ego_obs = np.append(ego_obs, norm_distance)
            ego_obs = np.append(ego_obs, heading_diff)
            
        
        #make sure array is type float32
        norm_ego_obs = np.array(norm_ego_obs, dtype=np.float32)
        ego_obs = np.array(ego_obs, dtype=np.float32)
            
        obs = {"ego": norm_ego_obs,
               "actual_ego": ego_obs}    
        
        return obs
    
    def __get_info(self) -> dict:
        info = {}
        for pursuer in self.pursuers:
            info["pursuer_{}".format(pursuer.id)] = pursuer.get_observation()
        
        return info
        
    def get_reward(self) -> tuple:
        """
        This function will compute the reward for the agent
        based on the current state of the environment
        Very simple right now just get the distance to the goal
        """
        reward = 0          
        ego_obs = self.__get_observation()
        ego_pos = ego_obs['ego'][:3]
        ego_actual_pos = ego_obs['actual_ego'][:3]
        #check time limit
        if self.time_limit <= 0:
            reward += 1000
            # print("Time limit reached you survived!")
            return reward, True

        if ego_pos[2] < -1:
            #print("crashed into ground")
            reward += -100
            return reward, True
        
        if ego_pos[2] > 1:
            #print("Flew too high sky")
            reward += -100
            return reward, True
            #print("Flew too high sky")

        # if ego_pos[2] < self.state_constraints['z_min']:
        #     reward += -1E3
        #     print("Crashed into the ground")
        #     return reward, True
        
        # if ego_pos[2] > self.state_constraints['z_max']:
        #     reward += -1E3
        #     print("Flew too high sky")
        #     return reward, True

        caught = False
        # this is redundant but I will keep it for now
        # should put this as an average reward based on the number of pursuers
        for i, pursuer in enumerate(self.pursuers):
            # pursuer_obs = pursuer.get_observation()
            # pursuer_pos = pursuer_obs[:3]
            # old_distance = self.distance_history[i]    
            # distance = self.compute_distance(ego_pos, pursuer_pos)
            # heading_error = abs(ego_obs['ego'][5] - pursuer_obs[5])             
            # dz = ego_pos[2] - pursuer_pos[2]
            pursuer_obs = pursuer.get_observation()
            norm_pursuer_obs = pursuer_obs#self.map_real_observation_to_normalized_observation(pursuer_obs)
            norm_pursuer_pos = norm_pursuer_obs[:3]
            old_distance = self.distance_history[i]
            distance = self.compute_distance(ego_actual_pos, norm_pursuer_pos)
            
            #normalize distance to old distance
            distance_cost = distance - old_distance 
            
            #normalize ratio of distance to old distance
            # heading_error = abs(ego_obs['actual_ego'][5] - norm_pursuer_obs[5])
            #negative because we want to reward being as far away from pursuers
            #get 2d unit vectors
            ego_unit_vector = ego_obs['ego'][:2] / np.linalg.norm(ego_obs['ego'][:2])
            pursuer_unit_vector = norm_pursuer_obs[:2] / np.linalg.norm(norm_pursuer_obs[:2])
            dot_product = - np.dot(ego_unit_vector, pursuer_unit_vector)/2
        
            #print("Distance: ", distance)
            #we want to be as far away from the pursuers as possible
            if distance_cost < 0:
                reward += -1 + dot_product
                # reward += distance_cost  
                # reward += dot_product  + (1*0.001)
            elif distance <= self.distance_tolerance:
                #reward += distance - old_distance
                # reward += self.time_step_constant - self.time_limit  
                reward += -100 
                print("Caught by pursuer")
                caught = True
            else:
                reward += 1 + dot_product
                # reward += dot_product + (1*0.001)        
            self.distance_history[i] = distance
        
        # get the average reward
        reward = (reward / self.num_pursuers) + (1*0.1)
        
        if caught:
            return reward, True
        else:
            return reward, False
        
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Note action input is normalized to [-1, 1] and must be mapped to the
        constraints of the aircraft
        """
        # reward = 0
        done = False
        real_action = self.map_normalized_action_to_real_action(action)
        observation = self.__get_observation()
        info = self.__get_info()
                
        self.time_limit -= 1
        self.init_counter += 1

        #check if the action is feasible 
        # proj_x = observation['ego'][0] + real_action[0]
        # proj_y = observation['ego'][1] + real_action[1]
        # proj_z = observation['ego'][2] + real_action[2]
        # proj_psi = np.arctan2(proj_y, proj_x)
        # dz = proj_z - observation['ego'][2]
        # dx = proj_x - observation['ego'][0]
        # dy = proj_y - observation['ego'][1]
        # proj_theta = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        # delta_psi = abs(proj_psi - observation['ego'][5])
        
        #Wrap delta psi
        # if delta_psi > np.pi:
        #     delta_psi = 2 * np.pi - delta_psi
        # elif delta_psi < -np.pi:
        #     delta_psi = 2 * np.pi + delta_psi

        #self.action_history.append((proj_x, proj_y, proj_z))
        # position_action = np.array([proj_x, proj_y, proj_z, real_action[3]])
        #self.backend_interface.set_commands(position_action)
        
        # self.backend_interface.set_commands_w_pursuers(position_action, 
        #                                                self.pursuers)
        # real_action[1] = proj_z 
        self.backend_interface.set_commands_w_pursuers(real_action,
                                                         self.pursuers)

        step_reward,done = self.get_reward()
        time_step_reward = 1
        reward   = step_reward #+ time_step_reward 
        
        #check if max episode steps reached
        return observation, reward, done, False, info
    
    def reset(self, seed=None) -> Any:
        super().reset(seed=seed)
        
        #if seed is not None or self.use_random_start:
        if self.use_random_start:
            min_vel = self.state_constraints['airspeed_min'] + 3
            max_vel = self.state_constraints['airspeed_max'] - 3
            
            min_heading = -np.pi
            max_heading = np.pi
            
            random_vel = np.random.uniform(min_vel, max_vel)
            
            random_heading = np.random.uniform(min_heading+3, 
                                                max_heading-3)
            

            #wrap heading to [-pi, pi]
            if random_heading >= np.pi:
                print("Wrap heading")
                random_heading -= 2*np.pi
            elif random_heading <= -np.pi:
                print("Wrap heading")
                random_heading += 2*np.pi
                
            random_heading = np.rad2deg(random_heading)
                    
            random_roll = np.random.uniform(
                self.state_constraints['phi_min'],
                self.state_constraints['phi_max'])
            
            random_roll = 0.0 
            # random_pitch = np.random.uniform(
            #     self.state_constraints['theta_min'],
            #     self.state_constraints['theta_max'])
            
            random_pitch = 0.0
            
            random_x = np.random.uniform(-100, 100)
            random_y = np.random.uniform(-100, 100)
            random_z = np.random.uniform(40, 60)
            random_position = np.array([random_x, random_y, random_z])
            geo_location = local_to_global_position(random_position)
            random_lat_dg = geo_location[1]
            random_lon_dg = geo_location[0]
            # random_alt_ft = geo_location[2]
            init_state_dict = {
                "ic/u-fps": meters_to_feet(random_vel),
                "ic/v-fps": 0.0,
                "ic/w-fps": 0.0,
                "ic/p-rad_sec": 0.0,
                "ic/q-rad_sec": 0.0,
                "ic/r-rad_sec": 0.0,
                "ic/h-sl-ft": meters_to_feet(random_z),
                "ic/long-gc-deg": random_lon_dg,
                "ic/lat-gc-deg": random_lat_dg,
                "ic/psi-true-deg": random_heading,
                "ic/theta-deg": np.rad2deg(random_pitch),
                "ic/phi-deg": np.rad2deg(random_roll),
                "ic/alpha-deg": 0.0,
                "ic/beta-deg": 0.0,
                "ic/num_engines": 1,
            }
        
            self.backend_interface.init_conditions = init_state_dict
            self.backend_interface.reset_backend(
                init_conditions=init_state_dict)
            self.backend_interface = self.backend_interface
            self.pursuers = self.init_pursuers(self.num_pursuers)
                        
        else:    
            self.backend_interface.reset_backend()       
            for pursuer in self.pursuers:
                pursuer.reset_backend()
        
        observation = self.__get_observation()
        info = self.__get_info()

        self.time_limit = self.time_step_constant

        return observation, info
    
    def render(self, mode:str='human') -> None:
        pass
    
