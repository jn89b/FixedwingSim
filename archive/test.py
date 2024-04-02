import jsbsim
import numpy as np
from guidance_control.autopilot import X8Autopilot
from src.jsbsim_simulator import FlightDynamics
from src.jsbsim_aircraft import x8


def feet_to_meters(feet:float) -> float:
    return feet * 0.3048

def meters_to_feet(meters:float) -> float:
    return meters / 0.3048

def knots_to_mps(knots:float) -> float:
    return knots * 0.514444

def mps_to_knots(mps:float) -> float:
    return mps / 0.514444

sim_hz = 200
dt = 1 / sim_hz


aircraft_name = "x8"

fdm = jsbsim.FGFDMExec(None)
fdm.set_debug_level(0)
fdm.load_model(aircraft_name)

# https://jsbsim-team.github.io/jsbsim/classJSBSim_1_1FGInitialCondition.html
init_state_dict = {
    "ic/u-fps": mps_to_knots(25),
    "ic/v-fps": 0.0,
    "ic/w-fps": 0.0,
    "ic/p-rad_sec": 0.0,
    "ic/q-rad_sec": 0.0,
    "ic/r-rad_sec": 0.0,
    "ic/h-sl-ft": meters_to_feet(50),
    "ic/long-gc-deg": 0.0,
    "ic/lat-gc-deg": 0.0,
    "ic/psi-true-deg": 45,
    "ic/theta-deg": 0.0,
    "ic/phi-deg": 0.0,
    "ic/alpha-deg": 0.0,
    "ic/beta-deg": 0.0,
    "ic/num_engines": 1,
}

autopilot = X8Autopilot(fdm)

for k, v in init_state_dict.items():
    fdm[k] = v
fdm.run_ic()
result = fdm.run()

#start engine
fdm["fcs/throttle-cmd-norm"] = 0.2

altitude_history = []
heading_history = []
roll_history = []
time_history = []
start_time = fdm.get_property_value("simulation/sim-time-sec")

t_sim_end = 10
goal_heading = 80
n_steps = int(t_sim_end / dt)
current_heading_dg = fdm.get_property_value("attitude/heading-true-rad")

for i in range(0, n_steps):
    #do a level hold 
    #autopilot.level_hold()
    fdm.run()
    error_heading_dg = goal_heading - current_heading_dg
    # autopilot.altitude_hold(meters_to_feet(65))
    autopilot.roll_hold(np.deg2rad(10))
    autopilot.heading_hold(error_heading_dg)
    autopilot.pitch_hold(np.deg2rad(2))
    altitude_m = feet_to_meters(fdm.get_property_value("position/h-sl-ft"))
    # print(altitude_m)
    current_time = fdm.get_property_value("simulation/sim-time-sec")
    # print(current_time - start_time)
    # print("\n")
    current_heading_dg = np.rad2deg(fdm.get_property_value("attitude/heading-true-rad"))
    altitude_history.append(altitude_m)
    roll_history.append(fdm.get_property_value("attitude/phi-rad"))
    heading_history.append(fdm.get_property_value("attitude/heading-true-rad"))
    time_history.append(current_time - start_time)
    

import matplotlib.pyplot as plt

fig,ax = plt.subplots(4,1)
ax[0].plot(time_history, altitude_history)
ax[1].plot(time_history, np.rad2deg(roll_history))
ax[2].plot(time_history, np.rad2deg(heading_history))

# plt.plot(time_history, altitude_history)
plt.show()