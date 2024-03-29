from abc import ABC
import numpy as np
#import airsim
# import gym
# from tasks import Shaping
from src.jsbsim_simulator import FlightDynamics
from src.jsbsim_aircraft import Aircraft, cessna172P, ball, x8
from src.debug_utils import *
# import jsbsim_properties as prp
from simple_pid import PID
from src.autopilot import X8Autopilot
from src.navigation import WindEstimation
from src.report_diagrams import ReportGraphs

from typing import Type, Tuple, Dict


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
        
    def run_sim(self) -> None:
        raise NotImplementedError("Method run_sim not implemented")

    

