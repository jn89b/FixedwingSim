import jsbsim

from src.autopilot import X8Autopilot
from src.jsbsim_simulator import Simulation
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
    "ic/psi-true-deg": 0.0,
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
fdm["fcs/throttle-cmd-norm"] = -1.0

for i in range(0, 1000):
    fdm.run()
    print(fdm.get_property_value("velocities/v-down-fps"))