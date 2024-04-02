from abc import ABC
import numpy as np

from jsbim_backend.simulator import FlightDynamics
from jsbim_backend.aircraft import Aircraft, x8
from debug_utils import *
# import jsbsim_properties as prp
from simple_pid import PID
from guidance_control.autopilot import X8Autopilot
from guidance_control.navigation import WindEstimation
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
        
        self.autopilot = autopilot
        if self.autopilot is None:
            self.autopilot = X8Autopilot(self.sim)
            
        self.wind = WindEstimation(self.sim)
        self.report = ReportGraphs(self.sim)
        self.over = False
        self.graph = DebugGraphs(self.sim)
        
    def run_backend(self) -> None:
        raise NotImplementedError("Method run_sim not implemented")
    
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
                 flight_dynamics_sim_hz: float = 240,
                 autopilot: X8Autopilot = None,
                 debug_level: int = 0):
        
        # Remove the call to super().__init__()
        
        self.init_conditions = init_conditions
        self.aircraft = aircraft
        self.flight_dynamics_sim_hz = flight_dynamics_sim_hz
        
        #this will initialize the simulator with the aircraft and initial conditions
        self.sim = FlightDynamics(aircraft=aircraft, 
                                  init_conditions=init_conditions, 
                                  debug_level=debug_level)
        
        self.autopilot = autopilot
        if self.autopilot is None:
            self.autopilot = X8Autopilot(self.sim)
            
        self.wind = WindEstimation(self.sim)
        self.report = ReportGraphs(self.sim)
        self.over = False
        self.graph = DebugGraphs(self.sim)
        
    def run_backend(self) -> None:
        self.sim.run()
        
    def reset_backend(self, init_conditions:dict=None) -> None:
        if init_conditions is not None:
            self.init_conditions = init_conditions
            self.sim = FlightDynamics(aircraft=self.aircraft, 
                                      init_conditions=init_conditions)
        else:
            self.sim.reinitialise(self.init_conditions)
        
    def get_info(self) -> dict:
        return self.sim.get_states()

    # def render(self, mode: str = 'human') -> None:
    #     self.graph.plot()
