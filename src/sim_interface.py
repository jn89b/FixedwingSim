from abc import ABC
import numpy as np
import time 
from jsbsim_backend.simulator import FlightDynamics
from jsbsim_backend.aircraft import Aircraft, x8
from debug_utils import *
# import jsbsim_properties as prp
from simple_pid import PID
from guidance_control.autopilot import X8Autopilot
from guidance_control.navigation import WindEstimation
from opt_control.PlaneOptControl import PlaneOptControl
from conversions import feet_to_meters, meters_to_feet, ktas_to_mps, mps_to_ktas
from opt_control.PlaneMPC import PlaneMPC
from src.report_diagrams import ReportGraphs
from typing import Type, Tuple, Dict


"""
Adapter class to interface with the JSBSim API and other simulators
"""

class CLSimInterface():
    def __init__(self, 
                 init_conditions: dict,
                 aircraft: Aircraft = x8,
                 flight_dynamics_sim_hz: float = 240,
                 autopilot: X8Autopilot = None,
                 debug_level: int = 0):
        
        self.init_conditions = init_conditions
        self.aircraft = aircraft
        self.flight_dynamics_sim_hz = flight_dynamics_sim_hz
                
        #this will initialize the simulator with the aircraft and initial conditions
        self.sim = FlightDynamics(aircraft=aircraft, 
                                  init_conditions=init_conditions, 
                                  debug_level=debug_level,
                                  sim_frequency_hz=flight_dynamics_sim_hz)
        self.sim.start_engines()
        self.sim.set_throttle_mixture_controls(0.3, 0)
        
        self.autopilot = autopilot
        if self.autopilot is None:
            self.autopilot = X8Autopilot(self.sim)
            
        self.wind = WindEstimation(self.sim)
        self.report = ReportGraphs(self.sim)
        self.over = False
        self.graph = DebugGraphs(self.sim)
        
    def run_backend(self, track_data:bool=True) -> None:
        self.sim.run()
        if track_data:
            self.graph.get_pos_data()
            self.graph.get_time_data()
            self.graph.get_angle_data()
            self.graph.get_airspeed()
    
    def reset_backend(self) -> None:
        """
        Require a method to reset the simulator
        """
        raise NotImplementedError("Method reset_sim not implemented")
    
    def get_states(self) -> dict:
        return self.sim.get_states()
    
class OpenGymInterface(CLSimInterface):
    def __init__(self, 
                 init_conditions: dict,
                 aircraft: Aircraft = x8,
                 flight_dynamics_sim_hz: float = 100,
                 autopilot: X8Autopilot = None,
                 use_mpc: bool = False,
                 mpc_controller: PlaneOptControl = None,
                 debug_level: int = 0):
        
        super().__init__(init_conditions=init_conditions,
                         aircraft=aircraft,
                         flight_dynamics_sim_hz=flight_dynamics_sim_hz,
                         autopilot=autopilot,
                         debug_level=debug_level)
        
        # Additional initialization for OpenGymInterface
        self.use_mpc = use_mpc
        self.mpc_controller = mpc_controller
        if self.use_mpc and self.mpc_controller is None:
            raise ValueError("MPC Controller not provided")

    def set_commands(self, action:np.ndarray, 
                     init_states_mpc:np.ndarray=None) -> None:
        """
        This sets the roll, pitch, yaw, and throttle commands 
        for the aircraft using the autopilot, simulating 
        our flight controller        
        """
        # self.autopilot.set_commands(action)
        if not self.use_mpc:
            print("Using Autopilot")
            roll_cmd = action[0]
            pitch_cmd = action[1]
            yaw_cmd = action[2]
            throttle_cmd = action[3] #this is 
            # print("yaw cmd deg", np.rad2deg(yaw_cmd))
            # self.autopilot.roll_hold(roll_cmd)
            # self.autopilot.pitch_hold(pitch_cmd)
            self.autopilot.altitude_hold(meters_to_feet(50))
            self.autopilot.heading_hold(np.rad2deg(yaw_cmd)) 
            self.autopilot.airspeed_hold_w_throttle(mps_to_ktas(20))
        else:
            final_states = [
                action[0],
                action[1],
                action[2],
                0,
                0,
                0,
                20
            ]
            #feed it to the mpc controller 
            init_states = self.get_observation()
            # print("init_states", init_states)
            init_control = [
                init_states[3],
                init_states[4],
                init_states[5],
                init_states[6]
            ]
            # print("init_states", init_states[2])
            # print("pitch", np.rad2deg(init_states[4]))
            solution_results, end_time = self.mpc_controller.get_solution(
                init_states, final_states, init_control)

            #set the commands
            idx_step = 1
            z_cmd = solution_results['z'][idx_step]
            roll_cmd = solution_results['phi'][idx_step]
            pitch_cmd = solution_results['theta'][idx_step]
            heading_cmd = solution_results['psi'][idx_step]
            airspeed_cmd = solution_results['v_cmd'][idx_step]
            
            self.autopilot.altitude_hold(meters_to_feet(60))
            self.autopilot.airspeed_hold_w_throttle(mps_to_ktas(airspeed_cmd))
            sim_hz = self.flight_dynamics_sim_hz
            control_hz = 5
            
            for i in range(sim_hz):
                if i % control_hz == 0 and i != 0:
                    return
                else:
                    self.run_backend()
                    
    def set_commands_w_pursuers(self, action:np.ndarray,
                                pursuer_list: list) -> None:
        final_states = [
            action[0],
            action[1],
            action[2],
            0,
            0,
            0,
            20
        ]
        #feed it to the mpc controller 
        init_states = self.get_observation()
        init_control = [
            init_states[3],
            init_states[4],
            init_states[5],
            init_states[6]
        ]
        
        solution_results, end_time = self.mpc_controller.get_solution(
            init_states, final_states, init_control)

        #set the commands
        idx_step = 1
        z_cmd = solution_results['z'][idx_step]
        roll_cmd = solution_results['phi'][idx_step]
        pitch_cmd = solution_results['theta'][idx_step]
        heading_cmd = solution_results['psi'][idx_step]
        airspeed_cmd = solution_results['v_cmd'][idx_step]
                
        self.autopilot.pitch_hold(pitch_cmd)
        self.autopilot.roll_hold(roll_cmd)
        self.autopilot.altitude_hold(meters_to_feet(heading_cmd))
        sim_hz = self.flight_dynamics_sim_hz
        control_hz = 10

        current_time = self.sim.get_time()            
        for i in range(sim_hz):
            if i % control_hz == 0 and i != 0:
                return
            else:
                self.run_backend()
                evader_observation = self.get_observation()
                for pursuer in pursuer_list:
                    pursuer_height = pursuer.get_observation()[2]
                    turn_cmd, v_cmd = pursuer.pursuit_nav(evader_observation)
                    dz = evader_observation[2] - pursuer_height
                    pursuer.set_command(turn_cmd, 
                                        v_cmd, 
                                        pursuer_height+dz)
                
                    
    def reset_backend(self, init_conditions:dict=None) -> None:
        if init_conditions is not None:
            self.init_conditions = init_conditions
            self.sim.reinitialise(init_conditions)
        else:
            self.sim.reinitialise(self.init_conditions)

    def get_info(self) -> dict:
        return self.sim.get_states()

    def get_observation(self) -> np.ndarray:
        """
        Returns the observation space for the environment as 
        a numpy array 
        """
        state_dict = self.sim.get_states()
        states = [
            state_dict['x'],
            state_dict['y'],
            state_dict['z'],
            state_dict['phi'],
            state_dict['theta'],
            state_dict['psi'],
            state_dict['airspeed'],
        ]

        return np.array(states, dtype=np.float32)
    
class PursuerInterface(CLSimInterface):
    """
    This class is an interface between the pursuer drone and the
    JSBSim flight dynamics. It is used to control the pursuer drone
    where the user can utilize a simple proportional navigation 
    or a pursuit guidance law to track the evader drone.
    """
    def __init__(self, 
                 init_conditions: dict,
                 evader_position: np.ndarray,
                 control_constraints: dict,
                 id_number: int,
                 nav_constant: float = 5,
                 aircraft: Aircraft = x8,
                 flight_dynamics_sim_hz: float = 100,
                 autopilot: X8Autopilot = None,
                 min_max_vels:np.ndarray = np.array([15, 30]),
                 debug_level: int = 0):
        
        super().__init__(init_conditions=init_conditions,
                         aircraft=aircraft,
                         flight_dynamics_sim_hz=flight_dynamics_sim_hz,
                         autopilot=autopilot,
                         debug_level=debug_level)
        
        self.control_constraints = control_constraints
        self.nav_constant = nav_constant
        self.old_states = self.get_observation()
        self.min_max_vels = min_max_vels
        self.previous_heading_rad = self.old_states[5]
        self.old_evader_position = evader_position
        self.id = id_number
        # self.old_los = np.arctan
        

    def track_evader(self, evader_position: np.ndarray) -> None:
        current_states = self.get_observation()

    def vector_magnitude(self,v):
        """Calculate the magnitude of a vector."""
        return math.sqrt(v[0]**2 + v[1]**2)


    def cartesian_to_navigation_radians(self, 
            cartesian_angle_radians:float) -> float:
        """
        Converts a Cartesian angle in radians to a navigation 
        system angle in radians.
        North is 0 radians, East is π/2 radians, 
        South is π radians, and West is 3π/2 radians.
        
        Parameters:
        - cartesian_angle_radians: Angle in radians in the Cartesian coordinate system.
        
        Returns:
        - A navigation system angle in radians.
        """
        new_yaw = math.pi/2 - cartesian_angle_radians
        return new_yaw

    def pro_nav(self, target_states:np.ndarray) -> tuple:
        """
        Returns the control commands for the pursuer drone
        Keep in mind in the simulator the angle orientation is 
        defined as follows:
        
        North = 0 degrees
        East = 90 degrees
        South = 180 degrees
        West = 270 degrees
        
        If we are using cartesian coordinates we will have to map it 
        to the above orientation to get the correct angle 
        """
        current_states = self.get_observation()
        pursuer_vel_mag = current_states[-1]
        
        dx = target_states[0] - current_states[0]
        dy = target_states[1] - current_states[1]
        
        los = np.arctan2(dy, dx)
          
        target_vel = target_states[-1]
        target_vx = target_vel * np.cos(target_states[5])
        target_vy = target_vel * np.sin(target_states[5])
        
        dt = 1/self.flight_dynamics_sim_hz
        target_next_x = target_states[0] + (target_vx*dt)
        target_next_y = target_states[1] + (target_vy*dt)
        
        los_next = np.arctan2(target_next_y - current_states[1], 
                              target_next_x - current_states[0])
                
        pursuer_vx = pursuer_vel_mag * np.cos(current_states[5])
        pursuer_vy = pursuer_vel_mag * np.sin(current_states[5])
        
        los_rate = np.array([target_vx - pursuer_vx, 
                             target_vy - pursuer_vy])
        los_rate = np.arctan2(los_rate[1], los_rate[0])
        
        los_mag_vel = -np.linalg.norm(los_rate)
                
        los = self.cartesian_to_navigation_radians(los)

        los_rate = self.cartesian_to_navigation_radians(los_rate)
        
        los_rate = los_rate * dt
        
        return los, los_rate

    def pursuit_nav(self, target_states:np.ndarray) -> tuple:
        """
        Returns the control commands for the pursuer drone
        Keep in mind in the simulator the angle orientation is 
        defined as follows:
        
        North = 0 degrees
        East = 90 degrees
        South = 180 degrees
        West = 270 degrees
        
        If we are using cartesian coordinates we will have to map it 
        to the above orientation to get the correct angle 
        
        """
        current_states = self.get_observation()
        
        dx = target_states[0] - current_states[0]
        dy = target_states[1] - current_states[1]
        los = np.arctan2(dy, dx)        
                        
        los = self.cartesian_to_navigation_radians(los)
        yaw = current_states[5]
        error_los = abs(los - yaw)

        if error_los > np.deg2rad(20):
            vel_cmd = self.min_max_vels[0]
        else:
            vel_cmd = self.min_max_vels[1]
            
        vel_cmd = np.clip(vel_cmd, self.min_max_vels[0], 
                          self.min_max_vels[1])
        return los, vel_cmd
    
    def set_command(self, heading_cmd_rad:float,
                    v_cmd_ms:float, 
                    height_cmd_m:float,
                    control_hz:int=30) -> None:
        """
        Set the control commands from pro_nav to the aircraft
        """
        v_max = self.control_constraints['v_cmd_max']
        v_min = self.control_constraints['v_cmd_min']
        
        if v_cmd_ms >= v_max:
            v_cmd_ms = v_max
        elif v_cmd_ms <= v_min:
            v_cmd_ms = v_min
        
        if heading_cmd_rad > np.pi:
            heading_cmd_rad = heading_cmd_rad - 2*np.pi
        elif heading_cmd_rad < -np.pi:
            heading_cmd_rad = heading_cmd_rad + 2*np.pi
                    
        self.autopilot.heading_hold(np.rad2deg(heading_cmd_rad))    
        self.autopilot.altitude_hold(meters_to_feet(height_cmd_m))
        self.autopilot.airspeed_hold_w_throttle(mps_to_ktas(v_cmd_ms))
        self.run_backend()

    def reset_backend(self, init_conditions:dict=None) -> None:
        if init_conditions is not None:
            self.init_conditions = init_conditions
            self.sim.reinitialise(init_conditions)
            # self.sim = FlightDynamics(aircraft=self.aircraft, 
            #                           init_conditions=init_conditions)
        else:
            self.sim.reinitialise(self.init_conditions)

    def get_info(self) -> dict:
        """
        Might belong to parent class
        """
        return self.sim.get_states()

    def get_observation(self) -> np.ndarray:
        """
        Returns the observation space for the environment as 
        a numpy array, might belong to parent class
        """
        state_dict = self.sim.get_states()
        states = [
            state_dict['x'],
            state_dict['y'],
            state_dict['z'],
            state_dict['phi'],
            state_dict['theta'],
            state_dict['psi'],
            state_dict['airspeed'],
        ]

        return np.array(states, dtype=np.float32)
