import os 
import numpy as np
import functools
from pettingzoo import AECEnv, ParallelEnv
from gymnasium import spaces
from src.models.Plane import Plane
from copy import copy
from pettingzoo.utils import wrappers
from pettingzoo.utils import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

def env(render_mode=None, 
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
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recomend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    TODO:refactor this to use kwargs for the environment parameters
    '''
    env = raw_env(
        render_mode=render_mode,
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
    # env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {'render.modes': ['human'], 
                "name": "pursuer_evader_v2",
                "is_parallelizable": True,
                }
    def __init__(self,
                 render_mode=None,
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
        super().__init__()
        
        #TODO: need to refactor this to be more general and take in number of pursuers and evaders
        self.agents = ['pursuer', 'evader']
        self.possible_agents = copy(self.agents)
    
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        if pursuer_control_constraints is None:
            raise ValueError("pursuer_control_constraints must be specified")
        if evader_control_constraints is None:
            raise ValueError("evader_control_constraints must be specified")
        if pursuer_observation_constraints is None:
            raise ValueError("pursuer_observation_constraints must be specified")
        if evader_observation_constraints is None:
            raise ValueError("evader_observation_constraints must be specified")
        
        self.pursuer_control_constraints = pursuer_control_constraints
        self.evader_control_constraints = evader_control_constraints
        self.pursuer_observation_constraints = pursuer_observation_constraints
        self.evader_observation_constraints = evader_observation_constraints
        self.pursuer_capture_distance = pursuer_capture_distance
        self.pursuer_min_spawning_distance = pursuer_min_spawning_distance
        self.pursuers_start_positions = pursuers_start_positions
        self.evader_start_positions = evader_start_positions
        
        self.dt = dt
        #used to determine the number of steps in a second for input frequency of the agents
        self.every_one_second = int(1/dt)
        rl_time_limit = 50
        self.rl_time_limit = rl_time_limit 
        self.rl_time_constant = rl_time_limit

        #self.action_space
        self.render_mode = render_mode
        self.planes = self.init_planes()
        self.observations = {}
        self.infos = {}
        for agent,v in self.planes.items():
            self.observations[agent] = self.observe(agent)
            self.infos[agent] = {}
        #this is a flag to determine if the round has ended
        #based on the pursuer capturing the evader
        #or the time limit has been reached
        self.ROUND_END = False
        self.out_of_bounds_penalty = -100
        self.terminal_reward = 100 

    ########### METHODS ADDED BY ME ################
    def init_planes(self) -> dict:
        plane_states = {}
        
        #TODO: need to refactor this to be more general and take in number of pursuers and evaders
        
        # we want to spawn the evaders first
        for agent in self.agents:
            if 'evader' in agent:
                plane_states[agent] = self.set_plane(agent, self.evader_start_positions)
        
        for agent in self.agents:
            if 'pursuer' in agent:
                plane_states[agent] = self.set_plane(agent, self.pursuers_start_positions)
        return plane_states
    
    def set_plane(self, agent:str, locations:list) -> Plane:
        plane = Plane()
        plane.set_state_space()
        actual_states = self.set_spawn_states(agent, locations)
        plane.set_info(actual_states)
        
        return plane

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

    def map_real_to_norm(self, norm_max:float, norm_min:float, real_val:float) -> float:
        """I can probably abstract this out to a utility function"""
        #TODO: put this in a utility function
        return 2 * (real_val - norm_min) / (norm_max - norm_min) - 1
    
    def norm_map_to_real(self, norm_max:float, norm_min:float, norm_val:float) -> float:
        """
        I can probably abstract this out to a utility function
        """
        #TODO: put this in a utility function
        return norm_min + (norm_max - norm_min) * (norm_val + 1) / 2

    def set_spawn_states(self, agent:str, locations:list) -> np.array:
        """
        Spawn the agent at the specified location 
        if the location is not specified, spawn the agent at a random location
        if the agent is a pursuer make sure that it is spawned at a distance greater 
        than the minimum spawning distance
        """
        #TODO: need to refactor this to be more general and take in number of pursuers and evaders
        if 'evader' in agent:
            agent_constraints = self.evader_observation_constraints
            if locations is None:
                x = 0
                y = 0
                z = np.random.randint(agent_constraints['z_min'], agent_constraints['z_max'])
                phi = np.random.uniform(agent_constraints['phi_min'], agent_constraints['phi_max'])
                theta = np.random.uniform(agent_constraints['theta_min'], agent_constraints['theta_max'])
                psi = np.random.uniform(agent_constraints['psi_min'], agent_constraints['psi_max'])
                airspeed = np.random.uniform(agent_constraints['airspeed_min'], agent_constraints['airspeed_max'])
                states = np.array([x, y, z, phi, theta, psi, airspeed])
            else:
                states = np.array(locations)
        else:
            agent_constraints = self.pursuer_observation_constraints
            correct_spawn = False
            min_spawn_distance = self.pursuer_min_spawning_distance
            if locations is None:
                #TODO: this needs to change right now assuming evader spawns at 0,0
                x = np.random.uniform(-min_spawn_distance, min_spawn_distance)
                y = np.random.uniform(-min_spawn_distance, min_spawn_distance)
                z = np.random.uniform(agent_constraints['z_min'], agent_constraints['z_max'])
                phi = np.random.uniform(agent_constraints['phi_min'], agent_constraints['phi_max'])
                theta = np.random.uniform(agent_constraints['theta_min'], agent_constraints['theta_max'])
                psi = np.random.uniform(agent_constraints['psi_min'], agent_constraints['psi_max'])
                airspeed = np.random.uniform(agent_constraints['airspeed_min'], agent_constraints['airspeed_max'])
                states = np.array([x, y, z, phi, theta, psi, airspeed])
            else:
                states = np.array(locations)
            
        #make sure its float 32
        return states.astype(np.float32)
        
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
    
    ########### METHODS REQUIRED FOR PETTINGZOO ################
    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        """
        We'll use rays to normalize the observation space for each agent
        """
        high_obs = []
        low_obs = []
        
        #TODO: need to refactor this to be more general and take in number of pursuers and evaders
        if 'pursuer' in agent:
            observation_constraints = self.pursuer_observation_constraints
        else:
            observation_constraints = self.evader_observation_constraints
            
        for k,v in observation_constraints.items():
            if 'max' in k:
                high_obs.append(v)
            elif 'min' in k:
                low_obs.append(v)    
        
        # Add the constraints for relative observations
        relative_max_params = {
            'max_relative_distance': 1000,
            'max_dot_product': 1,
            'max_relative_speed': 50,
        }
        relative_min_params = {
            'min_relative_distance': 0,
            'min_dot_product': -1,
            'min_relative_speed': 0,
        }

        for k, v in relative_max_params.items():
            high_obs.append(v)

        for k, v in relative_min_params.items():
            low_obs.append(v)

        observation_space = spaces.Box(low=np.array(low_obs),
                                    high=np.array(high_obs),
                                    dtype=np.float32)

        return observation_space
    
    @functools.lru_cache(maxsize=None)    
    def action_space(self, agent: str) -> spaces.Box:
        high_action = []
        low_action = []

        #TODO: need to refactor this to be more general and take in number of pursuers and evaders
        if 'pursuer' in agent:
            control_constraints = self.pursuer_control_constraints
        else:
            control_constraints = self.evader_control_constraints
            
        for k,v in control_constraints.items():
            if 'max' in k:
                high_action.append(v)
            elif 'min' in k:
                low_action.append(v)
                
        action_space = spaces.Box(low=np.array(low_action),
                                  high=np.array(high_action),
                                  dtype=np.float32)
        
        return action_space
        
    def get_relative_observations(
        self, ego_plane:Plane, get_evader:bool=True) -> np.ndarray:
        """
        Get the relative observations of the other guys 
        If you flag get_evader to true returns all the other evaders 
        If you flag to false returns all the all the other pursuers
        """
        if get_evader:
            search_string = 'evader'
        else:
            search_string = 'pursuer'

        relative_obs = np.array([])
        ego_state = ego_plane.get_info()
                
        for k, other_plane in self.planes.items():
            
            if search_string in k:
                other_state = other_plane.get_info()
                distance = np.linalg.norm(ego_state[:3] - other_state[:3])
                dot_product = self.compute_relative_heading(ego_plane,other_plane)
                #get the last index since that is the velocity of the vehicle
                relative_velocities = np.abs(ego_state[-1] - other_state[-1])
                current_rel_obs =  np.array([distance, dot_product,relative_velocities])
                # observation = np.append(observation, relative_obs)
            else:
                continue 
            
            relative_obs = np.append(relative_obs, current_rel_obs)
    
        return relative_obs
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        #plane = self.planes[agent]
        #return plane.get_info()
        # code duplication right here but its fine for now
        observation = self.planes[agent].get_info()
        ego_plane = self.planes[agent]
        # we need to include the relative observation of the other agent
        # so if we have a pursuer we want to know the relative position of it
        # to the evader and vice versa
        #TODO: refactor this a lot of code duplication?
        if 'pursuer' in agent:
            relative_obs = self.get_relative_observations(ego_plane=ego_plane,
                                                          get_evader=True)
        else:
            relative_obs = self.get_relative_observations(ego_plane=ego_plane,
                                                          get_evader=False)
        observation = np.append(observation, relative_obs)
         
        return observation
        # return self.observations[agent].get_info()
        #return np.array(self.observations[agent])
    
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass
    
    def reset(self, seed=None, options=None) -> tuple:
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
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        # self.state = self.init_planes()
        # self.state = {agent: None for agent in self.agents}
        self.planes = self.init_planes()
        self.observations = {}
        for k,v in self.planes.items():
            self.observations[k] = self.observe(k)
            self.infos[k] = {}
        #TODO: refactor this a lot of duplication

        #self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.rl_time_limit = self.rl_time_constant
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.ROUND_END = False

        # return self.observations, self.infos
        
    def get_relative_distance_obs(self, agent:str) -> np.ndarray:
        """
        Get the relative distance between the pursuer and the evader
        """
        # if 'pursuer' in agent:
        #     for agent_names, planes in self.planes
        
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
        
    def get_pursuer_reward(self, pursuer:Plane, evader:Plane,
                           state_constraints:dict) -> tuple:
        """
        For the pursuer we want to MINIMIZE the distance 
        between the pursuer and the evader, as well as MAXIMIZE
        the dot product
        
        Returns a tuple of the reward and whether the simulation is done
        
        """ 
        sim_done = False
        if self.is_out_of_bounds(pursuer, state_constraints):
            sim_done = True
            return self.out_of_bounds_penalty, sim_done

        #TODO: refactor this to consider more pursuers and evaders
        pursuer_state = pursuer.get_info()
        evader_state = evader.get_info()
        
        #check if we have captured the evader
        distance = np.linalg.norm(pursuer_state[:3] - evader_state[:3])
        if distance < self.pursuer_capture_distance:
            sim_done = True
            return self.terminal_reward, sim_done
        
        if self.rl_time_limit == 0:
            sim_done = True
            return -self.terminal_reward, sim_done
        
        
        #get the normalized observation
        norm_obs = self.map_real_observation_to_normalized_observation(
            pursuer_state, state_constraints)
        
        other_norm_obs = self.map_real_observation_to_normalized_observation(
            evader_state, state_constraints)
        
        norm_distance = np.linalg.norm(norm_obs[:3] - other_norm_obs[:3])
        dot_product = self.compute_relative_heading(pursuer, evader)
        
        reward = -norm_distance + dot_product
        
        return reward, sim_done
        
    def get_evader_reward(self, evader:Plane, pursuer:Plane,
                          state_constraints:dict) -> tuple:
        """
        For the pursuer we want to MAXIMIZE the distance between
        the pursuer and the evader, as well as MINIMIZE the dot product
        """
        sim_done = False
        if self.is_out_of_bounds(evader, state_constraints):
            sim_done = True
            return self.out_of_bounds_penalty, sim_done
        
        pursuer_state = pursuer.get_info()
        evader_state = evader.get_info()
        
        #check if evader has been captured
        distance = np.linalg.norm(pursuer_state[:3] - evader_state[:3])
        if distance < self.pursuer_capture_distance:
            sim_done = True
            #minus because this is really a penalty
            return -self.terminal_reward, sim_done
        
        if self.rl_time_limit == 0:
            sim_done = True
            return self.terminal_reward, sim_done
        
        norm_obs = self.map_real_observation_to_normalized_observation(
            evader_state, state_constraints)
        
        other_norm_obs = self.map_real_observation_to_normalized_observation(
            pursuer_state, state_constraints)
        
        norm_distance = np.linalg.norm(norm_obs[:3] - other_norm_obs[:3])
        dot_product = self.compute_relative_heading(evader, pursuer)
        
        reward = norm_distance - dot_product
        
        return reward, sim_done
        
    def step(self, action:np.ndarray):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        
        #TODO: need to refactor this to be more general and take in number of pursuers and evaders
        # if action is None:
        #     return 
        
        # if any of the terminations are True, set true for all agents
        # and end the round
        agent = self.agent_selection
        
        if (self.truncations[agent] or self.terminations[agent]):
            return 
        
        # if (self.ROUND_END):
        #     for agent in self.agents:
        #         self.terminations[agent] = True
        #         self.truncations[agent] = True
        #     return self.observations, self.rewards, self.terminations, self.truncations, self.infos
        
        # if action is None:
        #     # self._was_dead_step(agent)
        #     return 
        
        print("action is: ", action)
            
        if 'pursuer' in agent:
            control_constraints = self.pursuer_control_constraints
            state_constraints = self.pursuer_observation_constraints
        else:
            control_constraints = self.evader_control_constraints
            state_constraints = self.evader_observation_constraints
        
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0
        
        # get the current state of the agent
        # self.state[agent] = action
        denorm_action = action
        # denorm_action = self.map_normalized_action_to_real_action(action, 
        #     control_constraints)
                
        # move the agent based on the action 
        current_time_step = self.rl_time_limit 
        # did this because I'm tired of calling the dictionary
        current_plane = self.planes[agent]
        # this is going to be our working variable
        next_state = current_plane.get_info()
        for i in range(self.every_one_second):
            next_step = current_plane.rk45(next_state, denorm_action, self.dt)
            current_plane.set_info(next_step)
            #wrap yaw from -pi to pi
            if next_step[5] > np.pi:
                next_step[5] -= 2*np.pi
            elif next_step[5] < -np.pi:
                next_step[5] += 2*np.pi
                
            actual_sim_time = current_time_step + i * self.dt
            current_plane.set_time(actual_sim_time)
            #TODO: include a conditional check here to break out of loop 
        
        # update the observations dictionary
        self.observations[agent] = self.observe(agent)
        #TODO: need to remove the planes and make it just infos
        self.infos[agent] = {}
        reward = 0
        
        # decrease the time limit
        self.rl_time_limit -= 1        
        
        # check condition to see if the round has ended
        if self.rl_time_limit == 0:
            self.ROUND_END = True
            
        # update the rewards dictionary
        if 'pursuer' in agent:
            reward, is_done = self.get_pursuer_reward(current_plane, 
                                                          self.planes['evader'],
                                                          state_constraints)    
        else:
            reward, is_done = self.get_evader_reward(current_plane, 
                                                         self.planes['pursuer'],
                                                         state_constraints)
        # reward = 1
        self.rewards[agent] += reward
        self.planes[agent].update_reward(reward)    
        # select the next agent
        if is_done:
            print("The round has ended", self.observations)
            self.ROUND_END = True
            next_agent = self._agent_selector.next()
            # since this is a 0 sum game we want to set the reward for the other
            # agent to be the opposite of the reward of the other agent 
            # if the one agent wins the other agent has to lose no draw
            self.rewards[next_agent] += -reward
            self.planes[next_agent].update_reward(-reward)

        self.agent_selection = self._agent_selector.next()
                        
    
        # add rewards to the cumulative rewards
        self._accumulate_rewards()
        
        if self.ROUND_END:
            self.terminations = {agent: True for agent in self.agents}
        
        if self.render_mode == 'human':
            self.render()

        # return self.observations, self.rewards, self.terminations, self.truncations, self.infos
