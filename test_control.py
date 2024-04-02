import jsbsim
import numpy as np
import matplotlib.pyplot as plt
from guidance_control.autopilot import X8Autopilot
from jsbim_backend.simulator import FlightDynamics
from conversions import feet_to_meters, meters_to_feet, knots_to_mps, mps_to_knots

# https://jsbsim-team.github.io/jsbsim/classJSBSim_1_1FGInitialCondition.html
"""
Trying to figure out how to interface with the JSBSim API
"""
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
    "ic/psi-true-deg": 15,
    "ic/theta-deg": 0.0,
    "ic/phi-deg": 0.0,
    "ic/alpha-deg": 0.0,
    "ic/beta-deg": 0.0,
    "ic/num_engines": 1,
}

sim = FlightDynamics(init_conditions=init_state_dict)
autopilot = X8Autopilot(sim)

current_pos = sim.get_local_position()
current_orientation = sim.get_local_orientation()
states = sim.get_states()

# start engine
sim.start_engines()
sim.set_throttle_mixture_controls(0.2, 0)
sim_end_time = 10

sim_freq = sim.sim_frequency_hz
N = int(sim_end_time * sim_freq)

x_history = []
y_history = []
z_history = []

phi_history = []
theta_history = []
psi_history = []

time_history = []
start_time = sim.get_time()

for i in range(N):
    current_pos = sim.get_local_position()
    current_orientation = sim.get_local_orientation()
    states = sim.get_states()
    
    x_history.append(current_pos[0])
    y_history.append(current_pos[1])
    z_history.append(current_pos[2])
    phi_history.append(current_orientation[0])
    theta_history.append(current_orientation[1])
    psi_history.append(current_orientation[2])
    time_history.append(sim.get_time() - start_time)
    autopilot.heading_hold(90)
    sim.run()

#3D plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_history, y_history, z_history, 'gray')

fig, ax = plt.subplots(3,1)
ax[0].plot(time_history, np.rad2deg(phi_history))
ax[1].plot(time_history, np.rad2deg(theta_history))
ax[2].plot(time_history, np.rad2deg(psi_history)) 

ax[0].set_ylabel('Roll [deg]')
ax[1].set_ylabel('Pitch [deg]')
ax[2].set_ylabel('Yaw [deg]')

plt.show()