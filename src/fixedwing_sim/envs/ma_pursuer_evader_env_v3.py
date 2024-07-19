# noqa: D212, D415
"""
# Rock Paper Scissors

```{figure} classic_rps.gif
:width: 140px
:name: rps
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import rps_v2` |
|--------------------|-----------------------------------------|
| Actions            | Discrete                                |
| Parallel API       | Yes                                     |
| Manual Control     | No                                      |
| Agents             | `agents= ['player_0', 'player_1']`      |
| Agents             | 2                                       |
| Action Shape       | Discrete(3)                             |
| Action Values      | Discrete(3)                             |
| Observation Shape  | Discrete(4)                             |
| Observation Values | Discrete(4)                             |


Rock, Paper, Scissors is a 2-player hand game where each player chooses either rock, paper or scissors and reveals their choices simultaneously. If both players make the same choice, then it is a draw. However, if their choices are different, the winner is determined as follows: rock beats
scissors, scissors beat paper, and paper beats rock.

The game can be expanded to have extra actions by adding new action pairs. Adding the new actions in pairs allows for a more balanced game. This means that the final game will have an odd number of actions and each action wins over exactly half of the other actions while being defeated by the
other half. The most common expansion of this game is [Rock, Paper, Scissors, Lizard, Spock](http://www.samkass.com/theories/RPSSL.html), in which only one extra action pair is added.

### Arguments

``` python
rps_v2.env(num_actions=3, max_cycles=15)
```

`num_actions`:  number of actions applicable in the game. The default value is 3 for the game of Rock, Paper, Scissors. This argument must be an integer greater than 3 and with odd parity. If the value given is 5, the game is expanded to Rock, Paper, Scissors, Lizard, Spock.

`max_cycles`:  after max_cycles steps all agents will return done.

### Observation Space

#### Pursuer Evader 

If 3 actions are required, the game played is the standard Rock, Paper, Scissors. The observation is the last opponent action and its space is a scalar value with 4 possible values. Since both players reveal their choices at the same time, the observation is None until both players have acted.
Therefore, 3 represents no action taken yet. Rock is represented with 0, paper with 1 and scissors with 2.

| Value  |  Observation |
| :----: | :---------:  |
| 0      | Rock         |
| 1      | Paper        |
| 2      | Scissors     |
| 3      | None         |

#### Expanded Game

If the number of actions required in the game is greater than 3, the observation is still the last opponent action and its space is a scalar with 1 + n possible values, where n is the number of actions. The observation will as well be None until both players have acted and the largest possible
scalar value for the space, 1 + n, represents no action taken yet. The additional actions are encoded in increasing order starting from the 0 Rock action. If 5 actions are required the game is expanded to Rock, Paper, Scissors, Lizard, Spock. The following table shows an example of an observation
space with 7 possible actions.

| Value  |  Observation |
| :----: | :---------:  |
| 0      | Rock         |
| 1      | Paper        |
| 2      | Scissors     |
| 3      | Lizard       |
| 4      | Spock        |
| 5      | Action_6     |
| 6      | Action_7     |
| 7      | None         |

### Action Space

#### Rock, Paper, Scissors

The action space is a scalar value with 3 possible values. The values are encoded as follows: Rock is 0, paper is 1 and scissors is 2.

| Value  |  Action |
| :----: | :---------:  |
| 0      | Rock         |
| 1      | Paper        |
| 2      | Scissors     |

#### Expanded Game

The action space is a scalar value with n possible values, where n is the number of additional action pairs. The values for 7 possible actions are encoded as in the following table.

| Value  |  Action |
| :----: | :---------:  |
| 0      | Rock         |
| 1      | Paper        |
| 2      | Scissors     |
| 3      | Lizard       |
| 4      | Spock        |
| 5      | Action_6     |
| 6      | Action_7     |

### Rewards

| Winner | Loser |
| :----: | :---: |
| +1     | -1    |

If the game ends in a draw, both players will receive a reward of 0.

### Version History

* v2: Merge RPS and rock paper lizard scissors spock environments, add num_actions and max_cycles arguments (1.9.0)
* v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)

"""
from __future__ import annotations

import os

import gymnasium
import numpy as np
# import pygame
# from gymnasium.spaces import Discrete, Tuple, Box
from gymnasium import spaces

from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from src.models.Plane import Plane


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
    
    pursuer_control_constraints = {
        'u_phi_min':  -np.deg2rad(45),
        'u_phi_max':   np.deg2rad(45),
        'u_theta_min':-np.deg2rad(5),
        'u_theta_max': np.deg2rad(5),
        'u_psi_min':  -np.deg2rad(45),
        'u_psi_max':   np.deg2rad(45),
        'v_cmd_min':   15,
        'v_cmd_max':   30
    }

    pursuer_observation_constraints = {
        'x_min': -750, 
        'x_max': 750,
        'y_min': -750,
        'y_max': 750,
        'z_min': 30,
        'z_max': 100,
        'phi_min':  -np.deg2rad(45),
        'phi_max':   np.deg2rad(45),
        'theta_min':-np.deg2rad(20),
        'theta_max': np.deg2rad(20),
        'psi_min':  -np.pi,
        'psi_max':   np.pi,
        'airspeed_min': 15,
        'airspeed_max': 30
    }

    evader_control_constraints = {
        'u_phi_min':  -np.deg2rad(45),
        'u_phi_max':   np.deg2rad(45),
        'u_theta_min':-np.deg2rad(5),
        'u_theta_max': np.deg2rad(5),
        'u_psi_min':  -np.deg2rad(45),
        'u_psi_max':   np.deg2rad(45),
        'v_cmd_min':   15,
        'v_cmd_max':   25
    }

    evader_observation_constraints = {
        'x_min': -750,
        'x_max': 750,
        'y_min': -750,
        'y_max': 750,
        'z_min': 30,
        'z_max': 100,
        'phi_min':  -np.deg2rad(45),
        'phi_max':   np.deg2rad(45),
        'theta_min':-np.deg2rad(20),
        'theta_max': np.deg2rad(20),
        'psi_min':  -np.pi,
        'psi_max':   np.pi,
        'airspeed_min': 15,
        'airspeed_max': 25
    }

    # env = raw_env(**kwargs)
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
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv):
    """Two-player environment for rock paper scissors.

    Expandable environment to rock paper scissors lizard spock action_6 action_7 ...
    The observation is simply the last opponent action.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pursuer_evader_v3",
        "is_parallelizable": True,
    }

    def __init__(
        self,
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
        #EzPickle.__init__(self, num_actions, max_cycles, render_mode, screen_height)
        super().__init__()
        #self.max_cycles = max_cycles
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
        
        self.dt = dt
        #used to determine the number of steps in a second for input frequency of the agents
        self.every_one_second = int(1/dt)
        rl_time_limit = 50
        self.rl_time_limit = rl_time_limit 
        self.rl_time_constant = rl_time_limit
        
        self.pursuer_control_constraints = pursuer_control_constraints
        self.evader_control_constraints = evader_control_constraints
        self.pursuer_observation_constraints = pursuer_observation_constraints
        self.evader_observation_constraints = evader_observation_constraints
        self.pursuer_capture_distance = pursuer_capture_distance
        self.pursuer_min_spawning_distance = pursuer_min_spawning_distance
        self.pursuers_start_positions = pursuers_start_positions
        self.evader_start_positions = evader_start_positions

        self.ROUND_END = False
        self.out_of_bounds_penalty = -100
        self.terminal_reward = 100 
        
        # Add the constraints for relative observations
        self.relative_max_params = {
            'max_relative_distance': 1000,
            'max_dot_product': 1,
            'max_relative_speed': 50,
        }
        self.relative_min_params = {
            'min_relative_distance': 0,
            'min_dot_product': -1,
            'min_relative_speed': 0,
        }

        self.agents = ['pursuer' , 'evader']
        self.possible_agents = self.agents[:]
        self.planes = self.init_planes()
        self.action_spaces = self.init_action_spaces()
        self.observation_spaces = self.init_observation_spaces()

        self.render_mode = render_mode

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
            #if locations is None:
            x = 0
            y = 0
            z = np.random.randint(agent_constraints['z_min'], agent_constraints['z_max'])
            phi = np.random.uniform(agent_constraints['phi_min'], agent_constraints['phi_max'])
            theta = np.random.uniform(agent_constraints['theta_min'], agent_constraints['theta_max'])
            psi = np.random.uniform(agent_constraints['psi_min'], agent_constraints['psi_max'])
            airspeed = np.random.uniform(agent_constraints['airspeed_min'], agent_constraints['airspeed_max'])
            states = np.array([x, y, z, phi, theta, psi, airspeed])
            # else:
            #     states = np.array(locations)
        else:
            agent_constraints = self.pursuer_observation_constraints
            correct_spawn = False
            min_spawn_distance = self.pursuer_min_spawning_distance
            # if locations is None:
            #TODO: this needs to change right now assuming evader spawns at 0,0
            x = np.random.uniform(-min_spawn_distance, min_spawn_distance)
            y = np.random.uniform(-min_spawn_distance, min_spawn_distance)
            z = np.random.uniform(agent_constraints['z_min'], agent_constraints['z_max'])
            phi = np.random.uniform(agent_constraints['phi_min'], agent_constraints['phi_max'])
            theta = np.random.uniform(agent_constraints['theta_min'], agent_constraints['theta_max'])
            psi = np.random.uniform(agent_constraints['psi_min'], agent_constraints['psi_max'])
            airspeed = np.random.uniform(agent_constraints['airspeed_min'], agent_constraints['airspeed_max'])
            states = np.array([x, y, z, phi, theta, psi, airspeed])
            # else:
            #     states = np.array(locations)
            
        #make sure its float 32
        return states.astype(np.float32)
        
    def is_out_of_bounds(self, ego:Plane, state_constraints:dict) -> bool:
        """
        """
        ego_position = ego.get_info()[:3]
        buffer = 1
        if ego_position[0] < state_constraints['x_min']+buffer or ego_position[0] > state_constraints['x_max']-buffer:
            return True
        elif ego_position[1] < state_constraints['y_min']+buffer or ego_position[1] > state_constraints['y_max'] - buffer:
            return True
        elif ego_position[2] < state_constraints['z_min']+buffer or ego_position[2] > state_constraints['z_max'] - buffer:
            return True
        
        return False
        
    def init_action_spaces(self) -> None:
        action_spaces = {}
        for agent in self.agents:
            action_spaces[agent] = self.get_agent_action_space(agent)
        return action_spaces
    
    def get_agent_action_space(self, agent:str) -> gymnasium.Space:
        """
        We'll use rays to normalize the action space for each agent
        """
        high_action = []
        low_action = []
        
        if 'pursuer' in agent:
            action_constraints = self.pursuer_control_constraints
        else:
            action_constraints = self.evader_control_constraints
            
        for k,v in action_constraints.items():
            if 'max' in k:
                high_action.append(v)
            elif 'min' in k:
                low_action.append(v)
        
        action_space = spaces.Box(low=np.array(low_action),
                                    high=np.array(high_action),
                                    dtype=np.float32)
        
        return action_space
        
    def init_observation_spaces(self) -> None:
        observation_spaces = {}
        for agent in self.agents:
            observation_spaces[agent] = self.get_agent_obs_space(agent)
        return observation_spaces

    def get_agent_obs_space(self, agent:str) -> gymnasium.Space:
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

        for k, v in self.relative_max_params.items():
            high_obs.append(v)

        for k, v in self.relative_min_params.items():
            low_obs.append(v)

        observation_space = spaces.Box(low=np.array(low_obs),
                                    high=np.array(high_obs),
                                    dtype=np.float32)
        
        return observation_space

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
                
                #clip the relative observations
                distance = np.clip(distance, self.relative_min_params['min_relative_distance'], 
                                   self.relative_max_params['max_relative_distance'])
                
                dot_product = np.clip(dot_product, self.relative_min_params['min_dot_product'],
                                      self.relative_max_params['max_dot_product'])
                
                current_rel_obs = np.clip(current_rel_obs, self.relative_min_params['min_relative_speed'],
                                          self.relative_max_params['max_relative_speed'])
                
            else:
                continue 
            
            relative_obs = np.append(relative_obs, current_rel_obs)
    
        return relative_obs

    def get_observation(self, agent:str) -> np.array:
        """
        Besides the state of the aircraft we want to return
        the relative position, relative speed and relative heading
        of the other aircraft
        Ego means the agent that we are observing
        """
        ego_states = self.planes[agent].get_info()
        ego_plane = self.planes[agent]
        if 'pursuer' in agent:
            relative_obs = self.get_relative_observations(ego_plane, get_evader=True)
        else:
            relative_obs = self.get_relative_observations(ego_plane, get_evader=False)
            
        return np.append(ego_states, relative_obs)

    def observation_space(self, agent) -> gymnasium.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent) -> gymnasium.Space:
        return self.action_spaces[agent]

    def observe(self, agent:str) -> np.array:
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self) -> None:
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # self.state = {agent: self._none for agent in self.agents}
        # self.observations = {agent: self._none for agent in self.agents}
        self.planes = self.init_planes()
        self.observations = {agent: self.get_observation(agent) for agent in self.agents}
        self.rl_time_limit = self.rl_time_constant
        self.ROUND_END = False
        
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
            print("Pursuer is out of bounds")
            sim_done = True
            return self.out_of_bounds_penalty, sim_done

        #TODO: refactor this to consider more pursuers and evaders
        pursuer_state = pursuer.get_info()
        evader_state = evader.get_info()
        
        #check if we have captured the evader
        distance = np.linalg.norm(pursuer_state[:3] - evader_state[:3])
        if distance <= self.pursuer_capture_distance:
            print("Pursuer has captured the evader")
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
        
        
    def get_evader_reward(self, evader:Plane, pursuer:Plane,
                          state_constraints:dict) -> tuple:
        """
        For the pursuer we want to MAXIMIZE the distance between
        the pursuer and the evader, as well as MINIMIZE the dot product
        """
        sim_done = False
        if self.is_out_of_bounds(evader, state_constraints):
            print("Evader is out of bounds")
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

    def step(self, action):

        #check if termination is true
        for k,v in self.terminations.items():
            if v:
                self.ROUND_END = True
                action = None
                self._was_dead_step(action)
                return 

        current_agent = self.agent_selection
        self._cumulative_rewards[current_agent] = 0
        
        if 'pursuer' in current_agent:
            control_constraints = self.pursuer_control_constraints
            state_constraints = self.pursuer_observation_constraints
        else:
            control_constraints = self.evader_control_constraints
            state_constraints = self.evader_observation_constraints
        
        # self.state[self.agent_selection] = action

        #move the agent 
        current_time_step = self.rl_time_limit
        current_plane = self.planes[current_agent]
        next_state = current_plane.get_info()
        for i in range(self.every_one_second):
            next_step = current_plane.rk45(next_state, action, self.dt)
            current_plane.set_info(next_step)
            #wrap yaw from -pi to pi
            if next_step[5] > np.pi:
                next_step[5] -= 2*np.pi
            elif next_step[5] < -np.pi:
                next_step[5] += 2*np.pi
                
            actual_sim_time = current_time_step + i * self.dt
            current_plane.set_time(actual_sim_time)
            #TODO: include a conditional check here to break out of loop 

        #update the observation
        current_obs = self.get_observation(current_agent)
        self.observations[current_agent] = current_obs

        reward = 0 
        self.rl_time_limit -= 1
        
        # this is the dot product and distance between the two planes
        if 'pursuer' in current_agent:
            reward, is_done = self.get_pursuer_reward(current_plane, 
                                                          self.planes['evader'],
                                                          state_constraints)    
        else:
            reward, is_done = self.get_evader_reward(current_plane, 
                                                         self.planes['pursuer'],
                                                         state_constraints)

        # Time limit has been reached
        if self.rl_time_limit == 0:
            self.ROUND_END = True
            print("Time limit reached evader wins")
            # self.terminations = {agent: True for agent in self.agents}
            reward = self.terminal_reward
            for agent in self.agents:
                if 'pursuer' in agent:
                    self.rewards[agent] = -reward
                else:
                    self.rewards[agent] = reward
        # out of bounds penalty
        elif self.is_out_of_bounds(current_plane, state_constraints):
            self.ROUND_END = True
            reward = self.out_of_bounds_penalty 
            for agent in self.agents:
                if agent == current_agent:
                    self.rewards[agent] = reward
                else:
                    self.rewards[agent] = 0 
        # the other agent has been captured
        elif is_done:
            self.ROUND_END = True
            self.rewards[current_agent] = reward
            for agent in self.agents:
                if agent == current_agent:
                    self.rewards[agent] = reward
                else:
                    self.rewards[agent] = -reward
        
        self._cumulative_rewards[current_agent] = reward
        
        if self.ROUND_END:
            print("Round Ended", self.rewards)
            self.terminations = {agent: True for agent in self.agents}
            self.truncations = {agent: True for agent in self.agents}
            # self.infos = {agent: {} for agent in self.agents}
        
        if self.render_mode == "human":
            self.render()
            
        self.agent_selection = self._agent_selector.next()
            
        # add rewards to the cumulative rewards
        self._accumulate_rewards()
        
        self.infos[current_agent] = {
            'state': next_state,
            'observation': current_obs,
            'action': action,
            'time': actual_sim_time,
            'reward': reward
        }
        