from abc import ABC
import numpy as np

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
                                  debug_level=debug_level)
        self.sim.start_engines()
        self.sim.set_throttle_mixture_controls(0.5, 0)
        
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
                 flight_dynamics_sim_hz: float = 50,
                 autopilot: X8Autopilot = None,
                 use_mpc: bool = False,
                 mpc_controller: PlaneOptControl = None,
                 debug_level: int = 0):
        
        # Remove the call to super().__init__()
        
        self.init_conditions = init_conditions
        self.aircraft = aircraft
        self.flight_dynamics_sim_hz = flight_dynamics_sim_hz
        self.use_mpc = use_mpc
        self.mpc_controller = mpc_controller
        if self.use_mpc and self.mpc_controller is None:
            raise ValueError("MPC Controller not provided")

        #this will initialize the simulator with the aircraft and initial conditions
        self.sim = FlightDynamics(aircraft=aircraft, 
                                  init_conditions=init_conditions, 
                                  debug_level=debug_level,
                                  sim_frequency_hz=flight_dynamics_sim_hz)
        
        self.autopilot = autopilot
        if self.autopilot is None:
            self.autopilot = X8Autopilot(self.sim)
            
        self.wind = WindEstimation(self.sim)
        self.report = ReportGraphs(self.sim)
        self.over = False
        self.graph = DebugGraphs(self.sim)

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
            print("yaw cmd deg", np.rad2deg(yaw_cmd))
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
            #print("action", action)
            #feed it to the mpc controller 
            init_states = self.get_observation()
            # print("init_states", init_states)
            init_control = [
                init_states[3],
                init_states[4],
                init_states[5],
                init_states[6]
            ]
            solution_results, end_time = self.mpc_controller.get_solution(
                init_states, final_states, init_control)
            
            
            for k in solution_results.keys():
                # print(k, solution_results[k])
                v_cmd = solution_results[k]
                z_cmd = solution_results['z']
                roll_cmd = solution_results['phi']
                pitch_cmd = solution_results['theta']
                heading_cmd = solution_results['psi']
                airspeed_cmd = solution_results['v_cmd']
                
                #for i in range(len(v_cmd)):
                # for i in range(5):
                #     self.autopilot.altitude_hold(z_cmd[i])  
                #     self.autopilot.heading_hold(np.rad2deg(heading_cmd[i]))
                #     self.autopilot.airspeed_hold_w_throttle(mps_to_ktas(airspeed_cmd[i]))
                    #self.run_backend()
                    
            #set the commands
            #idx_step = int((end_time - init_states['time']) * self.flight_dynamics_sim_hz)
            idx_step = 1
            z_cmd = solution_results['z'][idx_step]
            roll_cmd = solution_results['phi'][idx_step]
            pitch_cmd = solution_results['theta'][idx_step]
            heading_cmd = solution_results['psi'][idx_step]
            airspeed_cmd = solution_results['v_cmd'][idx_step]

            # self.autopilot.pitch_hold(pitch_cmd)
            # self.autopilot.roll_hold(roll_cmd)
            self.autopilot.heading_hold(np.rad2deg(heading_cmd))
            self.autopilot.altitude_hold(meters_to_feet(50))
            self.autopilot.airspeed_hold_w_throttle(mps_to_ktas(airspeed_cmd))
        
    def reset_backend(self, init_conditions:dict=None) -> None:
        if init_conditions is not None:
            self.init_conditions = init_conditions
            self.sim = FlightDynamics(aircraft=self.aircraft, 
                                      init_conditions=init_conditions)
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
