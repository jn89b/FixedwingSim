"""
This utilizes the Petting Zoo api to create a multi-agent environment for the pursuer-evader scenario.
https://clementbm.github.io/project/2023/03/29/reinforcement-learning-connect-four-rllib.html
"""

import functools
import numpy as np

from pettingzoo import AECEnv
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from gymnasium import spaces, logger
from src.models.Plane import Plane
from copy import copy


def raw_env(render_mode=None, n_pursuers:int=1,
                 n_evaders:int=1,
                 pursuer_control_constraints:dict=None,
                 evader_control_constraints:dict=None,
                 pursuer_observation_constraints:dict=None,
                 evader_observation_constraints:dict=None,
                 pursuer_capture_distance:float=20.0,
                 pursuer_min_spawning_distance:float=100.0,
                 pursuers_start_positions:list=None,
                 evader_start_positions:list=None,
                 dt:float=0.1,
                 rl_time_limit:int=100):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    #set kwargs to the PursuerEvaderEnv
    env = PursuerEvaderEnv(
        n_pursuers=n_pursuers,
        n_evaders=n_evaders,
        pursuer_control_constraints=pursuer_control_constraints,
        evader_control_constraints=evader_control_constraints,
        pursuer_observation_constraints=pursuer_observation_constraints,
        evader_observation_constraints=evader_observation_constraints,
        pursuer_capture_distance=pursuer_capture_distance,
        pursuer_min_spawning_distance=pursuer_min_spawning_distance,
        pursuers_start_positions=pursuers_start_positions,
        evader_start_positions=evader_start_positions,
        dt=dt,
        rl_time_limit=rl_time_limit
    )
    
    # env = PursuerEvaderEnv(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class PursuerEvaderEnv(ParallelEnv):
    """
    Creates a multi-agent environment where there can be n pursuers and 
    for now 1 evader.
    
    Reward function will be the reverse of one so if one is positive
    the other will be negative (like mini-max)
    """
    metadata = {"render.modes": ["human", "rgb_array"],
                "name": "pursuer_evader_env"}

    def __init__(self,
                 n_pursuers:int=1,
                 n_evaders:int=1,
                 pursuer_control_constraints:dict=None,
                 evader_control_constraints:dict=None,
                 pursuer_observation_constraints:dict=None,
                 evader_observation_constraints:dict=None,
                 pursuer_capture_distance:float=20.0,
                 pursuer_min_spawning_distance:float=100.0,
                 pursuers_start_positions:list=None,
                 evader_start_positions:list=None,
                 dt:float=0.1,
                 rl_time_limit:int=100):
        super(PursuerEvaderEnv).__init__()
        
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.pursuer_control_constraints = pursuer_control_constraints
        self.evader_control_constraints = evader_control_constraints
        self.pursuer_observation_constraints = pursuer_observation_constraints
        self.evader_observation_constraints = evader_observation_constraints
        self.pursuer_capture_distance = pursuer_capture_distance
        self.pursuer_min_spawning_distance = pursuer_min_spawning_distance
        self.pursuers_start_positions = pursuers_start_positions
        self.evader_start_positions = evader_start_positions
        
        self.state_size = 7
        self.action_size = 4
        
        self.dt = dt
        #used to determine the number of steps in a second for input frequency of the agents
        self.every_one_second = int(1/dt)
        self.rl_time_limit = rl_time_limit 
        self.rl_time_constant = rl_time_limit
        self.init_agents()
        self.init_agents_action_space()
        self.init_agents_observation_space()
    
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]
    
    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]
    
    def map_real_to_norm(self, norm_max:float, norm_min:float, real_val:float) -> float:
        """I can probably abstract this out to a utility function"""
        return 2 * (real_val - norm_min) / (norm_max - norm_min) - 1
    
    def norm_map_to_real(self, norm_max:float, norm_min:float, norm_val:float) -> float:
        """
        I can probably abstract this out to a utility function
        """
        return norm_min + (norm_max - norm_min) * (norm_val + 1) / 2
    
    def map_normalized_observation_to_real_observation(self, 
            observation:np.ndarray, state_constraints:dict) -> np.ndarray:
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
        
        x = self.norm_map_to_real(state_constraints['x_max'],
                                  state_constraints['x_min'],
                                  x_norm)
        
        y = self.norm_map_to_real(state_constraints['y_max'],
                                  state_constraints['y_min'],
                                  y_norm)
        
        z = self.norm_map_to_real(state_constraints['z_max'],
                                  state_constraints['z_min'],
                                  z_norm)
        
        roll = self.norm_map_to_real(state_constraints['phi_max'],
                                     state_constraints['phi_min'],
                                     roll_norm)
        
        pitch = self.norm_map_to_real(state_constraints['theta_max'],
                                      state_constraints['theta_min'],
                                      pitch_norm)
        
        yaw = self.norm_map_to_real(state_constraints['psi_max'],
                                    state_constraints['psi_min'],
                                    yaw_norm)
        
        v = self.norm_map_to_real(state_constraints['airspeed_max'],
                                  state_constraints['airspeed_min'],
                                  v_norm)
        real_obs = np.array([x, y, z, roll, pitch, yaw, v])
        return real_obs.astype(np.float32)
    
    
    def map_real_observation_to_normalized_observation(self, 
            observation:np.ndarray, state_constraints:dict) -> np.ndarray:
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
        
        x_norm = self.map_real_to_norm(state_constraints['x_max'],
                                       state_constraints['x_min'],
                                       x)
        
        y_norm = self.map_real_to_norm(state_constraints['y_max'],
                                       state_constraints['y_min'],
                                       y)
        
        z_norm = self.map_real_to_norm(state_constraints['z_max'],
                                       state_constraints['z_min'],
                                       z)
        
        roll_norm = self.map_real_to_norm(state_constraints['phi_max'],
                                          state_constraints['phi_min'],
                                          roll)
        
        pitch_norm = self.map_real_to_norm(state_constraints['theta_max'],
                                           state_constraints['theta_min'],
                                           pitch)
        
        yaw_norm = self.map_real_to_norm(state_constraints['psi_max'],
                                         state_constraints['psi_min'],
                                         yaw)
        
        v_norm = self.map_real_to_norm(state_constraints['airspeed_max'],
                                       state_constraints['airspeed_min'],
                                       v)
        
        norm_obs = np.array([x_norm, y_norm, z_norm, roll_norm, pitch_norm, yaw_norm, v_norm])
        #make sure to return as float32
        return norm_obs.astype(np.float32)
        
    
    def map_normalized_action_to_real_action(self, 
            action:np.ndarray, control_constraints:dict) -> np.ndarray:
        
        roll_cmd = self.norm_map_to_real(control_constraints['u_phi_max'],
                                     control_constraints['u_phi_min'],
                                     action[0])
        
        pitch_cmd = self.norm_map_to_real(control_constraints['u_theta_max'],
                                      control_constraints['u_theta_min'],
                                      action[1])
        
        yaw_cmd = self.norm_map_to_real(control_constraints['u_psi_max'],
                                    control_constraints['u_psi_min'],
                                    action[2])
        
        airspeed_cmd = self.norm_map_to_real(control_constraints['v_cmd_max'],
                                            control_constraints['v_cmd_min'],
                                            action[3])
        
        real_action = np.array([roll_cmd, pitch_cmd, yaw_cmd, airspeed_cmd])
        #return as type float32
        return real_action.astype(np.float32)
    
    
    def compute_relative_distance(self, ego:Plane, other:Plane) -> float:
        """
        Computes the relative distance between the ego and another plane
        """
        distance = np.linalg.norm(ego.get_info()[:3] - other.get_info()[:3])
        return distance
    
    def compute_relative_velocity(self, ego:Plane, other:Plane) -> float:
        """
        Computes the relative velocity between the ego and another plane
        """
        relative_velocity = np.linalg.norm(ego.get_info()[6] - other.get_info()[6])
        return relative_velocity
    
    def compute_relative_heading(self, ego:Plane, other:Plane) -> float:
        """
        Computes the relative heading between the ego and another plane
        this will be used to compute the dot product
        """
        # relative_heading = np.arctan2(ego.get_info()[1] - other.get_info()[1],
        #                               ego.get_info()[0] - other.get_info()[0])
        ego_heading = ego.get_info()[5]
        other_heading = other.get_info()[5]
        ego_unit_vector = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        other_unit_vector = np.array([np.cos(other_heading), np.sin(other_heading)])
        
        relative_heading = np.dot(ego_unit_vector, other_unit_vector)
        return relative_heading
    
    def set_start_positions(self, start_positions:list, index:int, 
                            num_required:int, obs_constraints:dict,
                            use_spawn_constraint:bool=False,
                            ref_position:np.ndarray=None) -> np.ndarray:
        """
        Spawns the pursuers and evaders in the environment
        If use_spawn_constraint is set to True then the pursuers and evaders
        will spawn at a minimum distance from the reference position
        """
        if start_positions:
            if len(start_positions) != num_required:
                raise ValueError("Num start positions must equal to num_required", 
                                 num_required, len(start_positions))
            
            start = start_positions[index]
                
            if len(start) != self.state_size:
                raise ValueError("Start position must be of size 7 the set state space",
                                 len(start))
            
            actual_states = np.array(start)
            actual_states = actual_states.astype(np.float32)
            print("Actual States:", actual_states)
            return actual_states
        else:
            x = np.random.uniform(obs_constraints['x_min'],
                                    obs_constraints['x_max'])
            y = np.random.uniform(obs_constraints['y_min'],
                                    obs_constraints['y_max'])
            z = np.random.uniform(obs_constraints['z_min'],
                                    obs_constraints['z_max'])
            roll = np.random.uniform(obs_constraints['phi_min'],
                                    obs_constraints['phi_max'])
            pitch = np.random.uniform(obs_constraints['theta_min'],
                                    obs_constraints['theta_max'])
            yaw = np.random.uniform(obs_constraints['psi_min'],
                                    obs_constraints['psi_max'])
            v = np.random.uniform(obs_constraints['airspeed_min'],
                                    obs_constraints['airspeed_max'])
            actual_states = np.array([x, y, z, roll, pitch, yaw, v])
            #return as float32
            actual_states = actual_states.astype(np.float32)
            return actual_states
    
    def init_agents(self) -> None:
        """
        Creates a dictionary of agents in the environment for pursuers and evaders
        """
        self.agents_dict = {}
        self.possible_agents = []
        self.agents = []
        #TODO: Need to refactor this code, this is code duplication for pursuers and evaders
        for i in range(self.n_pursuers):
            plane = Plane()
            plane.set_state_space()
            name = 'pursuer_{}'.format(i)
            actual_states = self.set_start_positions(self.pursuers_start_positions, 
                                                     i, self.n_pursuers, 
                                                     self.pursuer_observation_constraints)
            
            plane.set_info(actual_states)
            self.agents_dict[name] = plane    
            self.possible_agents.append(name)
            
        for i in range(self.n_evaders):
            plane = Plane()
            plane.set_state_space()
            name = 'evader_{}'.format(i)
            actual_states = self.set_start_positions(self.evader_start_positions,
                                                        i, self.n_evaders,
                                                        self.evader_observation_constraints)
            plane.set_info(actual_states)
            self.agents_dict[name] = plane
            self.possible_agents.append(name)
        
        self.action_spaces = {agent: None for agent in self.agents_dict}
        self.observation_spaces = {agent: None for agent in self.agents_dict}
        self.infos = {agent: None for agent in self.agents_dict}
        self.observations = {agent: None for agent in self.agents_dict}
        self.rewards = {agent: 0 for agent in self.agents_dict}
        
    def init_agents_action_space(self) -> None:
        """
        This function will initialize the pursuers and evaders in the environment.
        based on the their respective control constraints
        """
        for i in range(self.n_pursuers):
            name = 'pursuer_{}'.format(i)
            self.action_spaces[name] = self.set_action_space(
                self.pursuer_control_constraints)
            
        for i in range(self.n_evaders):
            name = 'evader_{}'.format(i)
            self.action_spaces[name] = self.set_action_space(
                self.evader_control_constraints)
        
    def is_out_of_bounds(self, ego:Plane, state_constraints:dict) -> bool:
        """
        """
        ego_position = ego.get_info()[:3]
        if ego_position[0] < state_constraints['x_min'] or ego_position[0] > state_constraints['x_max']:
            return True
        elif ego_position[1] < state_constraints['y_min'] or ego_position[1] > state_constraints['y_max']:
            return True
        elif ego_position[2] < state_constraints['z_min'] or ego_position[2] > state_constraints['z_max']:
            return True
        
        return False
        
    def compute_evader_reward(self, pursuer:Plane, evader:Plane) -> float:
        """
        This function will compute the reward for the evader
        """
        
    def set_action_space(self, control_constraints:dict) -> spaces.Box:
        """
        This function will set the action space for the agents based on the control constraints
        These values will be normalized between a value of 1 and -1 since 
        we want a zero mean and unit variance for the actions.
        """
        
        high_action = []
        low_action = []
        
        for k,v in control_constraints.items():
            if 'max' in k:
                high_action.append(1)
            elif 'min' in k:
                low_action.append(-1)
                
        action_space = spaces.Box(low=np.array(low_action),
                                  high=np.array(high_action),
                                  dtype=np.float32)
        
        return action_space
    
    def init_agents_observation_space(self) -> None:
        """
        This function will initialize the pursuers and evaders in the environment
        based on their respective observation constraints.
        """
        for i in range(self.n_pursuers):
            self.observation_spaces['pursuer_{}'.format(i)] = self.set_observation_space(
                self.pursuer_observation_constraints)
            
        for i in range(self.n_evaders):
            self.observation_spaces['evader_{}'.format(i)] = self.set_observation_space(
                self.evader_observation_constraints)
               
    def set_observation_space(self, observation_constraints:dict,
                              set_for_evader:bool=False) -> spaces.Box:
        """
        This function will set the observation space for the agents based 
        on the observation constraints provided, the values will be normalized
        between a value of 1 and -1 since we want a zero mean and unit variance
        """
        high_observation = []
        low_observation = []
        
        for k,v in observation_constraints.items():
            if 'max' in k:
                high_observation.append(1)
            elif 'min' in k:
                low_observation.append(-1)
                
        #include an additional 3 values for the relative distance, 
        # relative velocity, and relative heading this will be used to
        # to help the agent understand the relationship between the pursuers and evaders
        #TODO: For future work need to set a flag for evader to include
        n_relative_values = 3 
        for i in range(n_relative_values):
            high_observation.append(1)
            low_observation.append(-1) 
        
        if set_for_evader:
            n_relative_values = 3 * self.n_pursuers
            for i in range(n_relative_values):
                high_observation.append(1)
                low_observation.append(-1)
        
        observation_space = spaces.Box(low=np.array(low_observation),
                                       high=np.array(high_observation),
                                       dtype=np.float32)
        
        return observation_space
    
    def get_relative_distance_obs(self, agent_name:str,
                                  get_norm_obs:bool=False) -> np.ndarray:
        """
        This function will return the relative observations of the agent
        """
        if get_norm_obs:
            observations = self.observations[agent_name]
            #since we have x,y,z,roll,pitch,yaw,airspeed
            relative_observations = observations[7:]
            #get the 0 and every other 3rd value
            relative_position = relative_observations[0::3]
            return relative_position
        
        observations = self.infos[agent_name]
        #since we have x,y,z,roll,pitch,yaw,airspeed
        relative_observations = observations[7:]
        #get the 0 and every other 3rd value
        relative_position = relative_observations[0::3]
        return relative_position
    
    def get_relative_velocity_obs(self, agent_name:str) -> np.ndarray:
        """
        This function will return the relative velocity of the agent
        """
        observations = self.infos[agent_name]
        #since we have x,y,z,roll,pitch,yaw,airspeed
        relative_observations = observations[7:]
        #get the 1 and every other 3rd value
        relative_velocity = relative_observations[1::3]
        return relative_velocity
    
    def get_relative_heading_obs(self, agent_name:str) -> np.ndarray:
        """
        This function will return the relative heading of the agent
        """
        observations = self.infos[agent_name]
        #since we have x,y,z,roll,pitch,yaw,airspeed
        relative_observations = observations[7:]
        #get the 2 and every other 3rd value
        relative_heading = relative_observations[2::3]
        return relative_heading
    
    def step(self, actions:dict) -> tuple:
        """
        This function will take in the action from the agents and return the 
        observations, rewards, dones, and infos for each agent.
        
        Need to check if any of our pursuers have captured the evader
        if so then we need to terminate the episode.
        
        Or if the evader has lived for a certain amount of time then we need to
        terminate the episode.
        
        Action input is dict with key as agent name and value as the action
        """
        if not actions:
            raise ValueError("Actions cannot be None")
        
        self.rl_time_limit -= 1
        # make pursuer move and then evader move
        for agent_name, action in actions.items():
            if 'pursuer' in agent_name:
                real_action = self.map_normalized_action_to_real_action(
                    action, self.pursuer_control_constraints)
            else:
                'evader' in agent_name
                real_action = self.map_normalized_action_to_real_action(
                    action, self.evader_control_constraints)  
            
            current_time_step = self.rl_time_limit
            current_agent = self.agents_dict[agent_name]

            #this is a working variable going to be used to store the next state
            next_state = current_agent.get_info()
            for i in range(self.every_one_second):
                next_state = current_agent.rk45(next_state, real_action, self.dt)
                #this will cache the next state
                current_agent.set_info(next_state)
                actual_sim_time = current_time_step + i * self.dt
                current_agent.set_time(actual_sim_time)
                    
        # set observations
        # self.observations = {a: None for a in self.agents_dict} 
        # self.infos = {a:None for a in self.agents_dict}
        for agent_name, agent in self.agents_dict.items():
            if 'pursuer' in agent_name:
                norm_observation = self.map_real_observation_to_normalized_observation(
                    agent.get_info(), self.pursuer_observation_constraints)
                self.observations[agent_name] = norm_observation
                # infos[agent_name] = 
                # print("Info before:", infos[agent_name])
                
                #include the relative distance, relative velocity, and relative heading of the evader
                #TODO: Need to refactor this code 
                for k,v in self.agents_dict.items():
                    if 'evader' in k:
                        evader_norm = self.map_real_observation_to_normalized_observation(
                            v.get_info(), self.evader_observation_constraints)
                        
                        evader_norm_positons = evader_norm[:3]
                        evader_norm_velocities = evader_norm[6]
                        evader_norm_heading = evader_norm[5]
                        
                        relative_norm_distance = np.linalg.norm(norm_observation[:3] - evader_norm_positons)
                        relative_norm_velocity = np.linalg.norm(norm_observation[6] - evader_norm_velocities)
                        relative_norm_heading = np.arctan2(norm_observation[1] - evader_norm_positons[1],
                                                              norm_observation[0] - evader_norm_positons[0]) 

                        relative_distance = self.compute_relative_distance(agent, v)
                        relative_velocity = self.compute_relative_velocity(agent, v)
                        relative_heading = self.compute_relative_heading(agent, v)
                        
                        self.observations[agent_name] = np.append(self.observations[agent_name],
                                                                    np.array([relative_norm_distance,
                                                                            relative_norm_velocity,
                                                                            relative_norm_heading]))
                        self.infos[agent_name] = np.append(agent.get_info(),
                                                              np.array([relative_distance,
                                                                        relative_velocity,
                                                                        relative_heading]))
            else:
                'evader' in agent_name
                norm_observation = self.map_real_observation_to_normalized_observation(
                    agent.get_info(), self.evader_observation_constraints)
                self.observations[agent_name] = norm_observation
                
                for k,v in self.agents_dict.items():
                    if 'pursuer' in k:
                        pursuer_norm = self.map_real_observation_to_normalized_observation(
                            v.get_info(), self.pursuer_observation_constraints)
                        
                        pursuer_norm_positons = pursuer_norm[:3]
                        pursuer_norm_velocities = pursuer_norm[6]
                        pursuer_norm_heading = pursuer_norm[5]
                        
                        relative_norm_distance = np.linalg.norm(norm_observation[:3] - pursuer_norm_positons)
                        relative_norm_velocity = np.linalg.norm(norm_observation[6] - pursuer_norm_velocities)
                        relative_norm_heading = np.arctan2(norm_observation[1] - pursuer_norm_positons[1],
                                                          norm_observation[0] - pursuer_norm_positons[0]) 

                        relative_distance = self.compute_relative_distance(agent, v)
                        relative_velocity = self.compute_relative_velocity(agent, v)
                        relative_heading = self.compute_relative_heading(agent, v)
                        
                        self.observations[agent_name] = np.append(self.observations[agent_name],
                                                                    np.array([relative_norm_distance,
                                                                            relative_norm_velocity,
                                                                            relative_norm_heading]))
                                                
                        self.infos[agent_name] = np.append(agent.get_info(),
                                                              np.array([relative_distance,
                                                                        relative_velocity,
                                                                        relative_heading]))    
        # compute rewards for pursuer and evaders
        ## do if and elif here to check for termination
        # check if pursuer has captured evader
        # if so we will end the episode and reward the pursuer 
        # check if evader has lived for a certain amount of time
        # if so we will end the episode and reward the evader
        # set truncations -> reached maximum number of steps
        
        # for this terminations and truncations will be the same
        """
        We want to end the episode if:
            - The time limit has been reached -> evader survived
            - The pursuer has captured the evader -> evader did not survive
            - If the pursuer is out of bounds -> evader gets a huge reward
            - If the evader is out of bounds -> pursuer gets a huge reward
        """
        if self.rl_time_limit <= 0:
            ROUND_END = True
            evader_survived_round = True
        else:
            ROUND_END = False
            evader_survived_round = False
            
        # rewards = {}
        # truncations = {}
        # terminations = {}
        rewards = {a: 0 for a in self.agents_dict}
        truncations = {a: ROUND_END for a in self.agents_dict}
        terminations = {a: ROUND_END for a in self.agents_dict}
        
        out_of_bounds_penalty = np.array([-100])
        huge_payout = np.array([100])
        
        for agent_name, agent in self.agents_dict.items():
            #pursuer is negative
            if 'pursuer' in agent_name:
                pursuer = agent
                relative_distance_observation = self.get_relative_distance_obs(agent_name)
                if self.is_out_of_bounds(pursuer, self.pursuer_observation_constraints):
                    rewards[agent_name] = out_of_bounds_penalty
                    ROUND_END = True
                    print("Out of bounds")
                elif evader_survived_round:
                    rewards[agent_name] = -huge_payout
                elif np.any(relative_distance_observation <= self.pursuer_capture_distance):
                    rewards[agent_name] = huge_payout
                    ROUND_END = True
                else:  
                    # compute a reward function for the pursuer based on relative distance and heading
                    rel_norm_distance = self.get_relative_distance_obs(agent_name, get_norm_obs=True)
                    rel_norm_heading = self.get_relative_heading_obs(agent_name)
                    #for the pursuer we want to minimize the distance and get the maximum dot product
                    rewards[agent_name] = -rel_norm_distance + rel_norm_heading
                    
                self.rewards[agent_name] = rewards[agent_name]
                pursuer.update_reward(rewards[agent_name])
                #truncations[agent_name]  = ROUND_END
                terminations[agent_name] = ROUND_END
                
            else:
                #evader is positive
                'evader' in agent_name
                evader = agent
                # rewards[agent_name] = -1
                relative_distance_observation = self.get_relative_distance_obs(agent_name)
                if self.is_out_of_bounds(evader, self.evader_observation_constraints):
                    rewards[agent_name] = out_of_bounds_penalty
                    ROUND_END = True
                    print("Out of bounds")
                elif evader_survived_round:
                    rewards[agent_name] = huge_payout
                    ROUND_END = True
                #means the pursuer has captured the evader
                elif np.any(relative_distance_observation <= self.pursuer_capture_distance):
                    rewards[agent_name] = -huge_payout
                    ROUND_END = True
                else:
                    # compute a reward function for the evader based on relative distance and heading
                    rel_norm_distance = self.get_relative_distance_obs(agent_name, get_norm_obs=True)
                    rel_norm_heading = self.get_relative_heading_obs(agent_name)
                    
                    #for the evader we want to maximize the distance and get the minimum dot product
                    rewards[agent_name] = rel_norm_distance - rel_norm_heading
                    self.rewards[agent_name] = rewards[agent_name]
                    
                evader.update_reward(rewards[agent_name])    
                #truncations[agent_name] = ROUND_END
                terminations[agent_name] = ROUND_END

        #check if any of the agents have terminated the episode
        if any(terminations.values()):
            ROUND_END = True
            self.agents = []
            
        terminations = {a: ROUND_END for a in self.agents_dict}
        
        for agent_name, obs in self.observations.items():
            assert obs is not None, "Observations cannot be None"

        return self.observations, rewards, terminations, truncations, self.infos
        
    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.init_agents()
        self.init_agents_action_space()
        self.init_agents_observation_space()
        self.agents = copy(self.possible_agents)
        self.rl_time_limit = self.rl_time_constant
        self.rewards = {agent: 0 for agent in self.agents_dict}
        self._cumulative_rewards = {agent: 0 for agent in self.agents_dict}
        self.terminations = {agent: False for agent in self.agents_dict}
        self.truncations = {agent: False for agent in self.agents_dict}
        # self.infos = {agent: {} for agent in self.agents_dict}
        # self.state = {agent: None for agent in self.agents_dict}
        # self.observations = {agent: None for agent in self.agents_dict}
        # print("Agents:", self.agents)
        
        return self.observations, self.infos
        
    def render(self, mode='human'):
        pass  # Implement rendering logic

    def close(self):
        pass  # Implement any cleanup