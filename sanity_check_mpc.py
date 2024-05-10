"""
Sanity check to see if:
- MPC works with the JSBSim model environment
- The SITL data matches the JSBSim model

Send a desired location to the plane and see if it can reach it 
using the MPC controller

"""
import casadi as ca
import numpy as np
import pandas as pd

from src.models.Plane import Plane
from opt_control.PlaneOptControl import PlaneOptControl
from jsbsim_backend.aircraft import Aircraft, x8

import gymnasium as gym
import math 
import matplotlib.pyplot as plt
from src.sim_interface import OpenGymInterface
from src.conversions import meters_to_feet, mps_to_ktas
from src.conversions import local_to_global_position

def init_mpc_controller(mpc_control_constraints:dict,
                        state_constraints:dict,
                        mpc_params:dict, 
                        plane_model:dict) -> PlaneOptControl:

    plane_mpc = PlaneOptControl(
        control_constraints=mpc_control_constraints,
        state_constraints=state_constraints,
        mpc_params=mpc_params,
        casadi_model=plane_model)
    plane_mpc.init_optimization_problem()
    return plane_mpc

def enu_heading_to_ned_rads( 
        cartesian_angle_radians:float) -> float:
    """
    Converts a Cartesian angle in radians to a navigation 
    system angle in radians.
    
    Parameters:
    - cartesian_angle_radians: Angle in radians in the Cartesian coordinate system.
        
    East = 0 radians
    North = π/2 radians
    West = π radians
    South = 3π/2 radians
    
    Returns:
    - A navigation system angle in radians.
    """
    new_yaw = math.pi/2 - cartesian_angle_radians
    return new_yaw

sitl_flight_data = pd.read_csv('sitl_flight_data.csv')
x_pos = sitl_flight_data['x_pos'][0]
y_pos = sitl_flight_data['y_pos'][0]
z_pos = sitl_flight_data['z_pos'][0]

roll_cmd = sitl_flight_data['roll_cmd']
pitch_cmd = sitl_flight_data['pitch_cmd']
heading_cmd = sitl_flight_data['yaw_cmd']
airspeed_cmd = sitl_flight_data['vel_cmd']

x_sitl = sitl_flight_data['x_pos']
y_sitl = sitl_flight_data['y_pos']
z_sitl = sitl_flight_data['z_pos']

roll_sitl = sitl_flight_data['roll']
pitch_sitl = sitl_flight_data['pitch']
heading_sitl = sitl_flight_data['yaw']
airspeed_sitl = sitl_flight_data['vel']

global_pos = local_to_global_position([x_pos, y_pos, z_pos])

yaw_ned = enu_heading_to_ned_rads(heading_sitl[0])

init_state_dict = {
    "ic/u-fps": meters_to_feet(airspeed_sitl[0]),
    "ic/v-fps": 0.0,
    "ic/w-fps": 0.0,
    "ic/p-rad_sec": 0.0,
    "ic/q-rad_sec": 0.0,
    "ic/r-rad_sec": 0.0,
    "ic/h-sl-ft": meters_to_feet(global_pos[2]),
    "ic/long-gc-deg": global_pos[0],
    "ic/lat-gc-deg": global_pos[1],
    "ic/psi-true-deg": np.rad2deg(yaw_ned),
    "ic/theta-deg": np.rad2deg(pitch_sitl[0]),
    "ic/phi-deg": np.rad2deg(roll_sitl[0]),
    "ic/alpha-deg": 0.0,
    "ic/beta-deg": 0.0,
    "ic/num_engines": 1,
}

mpc_params = {
    'N': 15,
    'Q': ca.diag([1.0, 1.0, 1.0, 0, 0, 0.0, 0.0]),
    'R': ca.diag([0.0, 0.0, 0.0, 0.1]),
    'dt': 0.1
}

control_constraints = {
    'u_phi_min':  -np.deg2rad(45),
    'u_phi_max':   np.deg2rad(45),
    'u_theta_min':-np.deg2rad(10),
    'u_theta_max': np.deg2rad(10),
    'u_psi_min':  -np.deg2rad(45),
    'u_psi_max':   np.deg2rad(45),
    'v_cmd_min':   15,
    'v_cmd_max':   30
}

state_constraints = {
    'x_min': -np.inf,
    'x_max': np.inf,
    'y_min': -np.inf,
    'y_max': np.inf,
    'z_min': 30,
    'z_max': 100,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(15),
    'theta_max': np.deg2rad(15),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': 15,
    'airspeed_max': 30
}

plane = Plane()
plane.set_state_space()
mpc_control = init_mpc_controller(
    mpc_control_constraints=control_constraints,
    state_constraints=state_constraints,
    mpc_params=mpc_params,
    plane_model=plane)


jsb_freq = 200
gym_adapter = OpenGymInterface(init_conditions=init_state_dict,
                                 aircraft=x8,
                                 flight_dynamics_sim_hz=jsb_freq,
                                 use_mpc=True,
                                 mpc_controller=mpc_control)
# Set up the environment
# Set an action space mapped to a location 
# Then see if the plane can reach that location
init_states = gym_adapter.get_observation()
init_control =[
    init_states[3],
    init_states[4],
    init_states[5],
    init_states[6]
]

final_states = [
    50,
    50,
    30,
    50,
    0,
    0,
    25
]

N = len(x_sitl)

x_traj = []
y_traj = []
z_traj = []
roll_traj = []
pitch_traj = []
heading_traj = []
airspeed_traj = []
time_traj = []
h_cmd_traj = []
r_cmd_traj = []
x_ctrl = []
y_ctrl = []

dt = 0.03
#this is how long the control loop will run 
#round control freq to the nearest integer
control_freq = int(1/dt)
sampling_ratio = int(jsb_freq/control_freq)

control_counter = 0
for i in range(N):
#for i in range(len(x_sitl)):
    # solution_results, end_time = mpc_control.get_solution(
    #     init_states, 
    #     final_states,
    #     init_control)

    # idx_step = 1
    autopilot = gym_adapter.autopilot
    # x_cmd = solution_results['x'][idx_step]
    # y_cmd = solution_results['y'][idx_step]
    # z_cmd = solution_results['z'][idx_step]
    # roll_cmd = solution_results['phi'][idx_step]
    # pitch_cmd = solution_results['theta'][idx_step]
    # heading_cmd = solution_results['psi'][idx_step]
    # airspeed_cmd = solution_results['v_cmd'][idx_step]
    # print("heading cmd deg: ", np.rad2deg(heading_cmd))
    # print("roll cmd deg: ", np.rad2deg(roll_cmd))

    #use the sitl data as control commands
    x_cmd = x_sitl[i]
    y_cmd = y_sitl[i]
    z_cmd = z_sitl[i]
    r_cmd = roll_cmd[i]
    p_cmd = pitch_cmd[i]
    h_cmd = heading_cmd[i]
    a_cmd = airspeed_cmd[i]
    #autopilot.roll_hold(r_cmd)
    #autopilot.pitch_hold(p_cmd)
    # print("heading cmd deg: ", heading_cmd[i])
    # autopilot.heading_hold(np.rad2deg(-heading_cmd[i]))
    autopilot.airspeed_hold_w_throttle(mps_to_ktas(a_cmd))

    for j in range(sampling_ratio):
        #check if 
        # ## if you want to control the heading based on line of sight
        # dy = final_states[1] - init_states[1]
        # dx = final_states[0] - init_states[0]

        # # print("x and y cmd: ", x_cmd, y_cmd)
        # dy = y_cmd - init_states[1]
        # dx = x_cmd - init_states[0]
        # arctan_cmd_enu = np.arctan2(dy, dx)
        h_cmd_ned = enu_heading_to_ned_rads(h_cmd)        
        autopilot.heading_hold(np.rad2deg(h_cmd_ned)) 
        autopilot.altitude_hold(meters_to_feet(z_cmd))

        gym_adapter.run_backend()
        init_states = gym_adapter.get_observation()
        init_control =[
            init_states[3],
            init_states[4],
            init_states[5],
            init_states[6]
        ]
                
        x_traj.append(init_states[0])
        y_traj.append(init_states[1])
        z_traj.append(init_states[2])
        roll_traj.append(init_states[3])    
        pitch_traj.append(init_states[4])
        heading_traj.append(init_states[5])
        airspeed_traj.append(init_states[6])
        h_cmd_traj.append(h_cmd)
        time_traj.append(gym_adapter.sim.get_time())
        r_cmd_traj.append(r_cmd)

sitl_time = sitl_flight_data['current_time'][N-1] - sitl_flight_data['current_time'][0]
# sitl_time = np.linspace(0, sitl_time, len(x_traj))
print("sitl time: ", sitl_time)
sitl_time = sitl_flight_data['current_time'] - sitl_flight_data['current_time'][0]

fig, ax = plt.subplots()
ax.plot(x_traj, y_traj, label='JSBSIM Trajectory')
ax.scatter(x_sitl[0], y_sitl[0], label='JSBSIM Start')
ax.plot(x_pos, y_pos, color='green', marker='o', label='SITL Start')
ax.plot(x_sitl, y_sitl, color='red', label='SITL Trajectory')
# ax.scatter(final_states[0], final_states[1], color='red')
ax.legend()

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_traj, y_traj, z_traj, label='JSBSIM Trajectory')
ax.scatter(x_sitl[0], y_sitl[0], z_sitl[0], label='JSBSIM Start')
ax.plot(x_sitl, y_sitl, z_sitl, color='red', label='SITL Trajectory')
# ax.scatter(x_sitl[-1], y_sitl[-1], z_sitl[-1], color='green', label='SITL End')
ax.legend()

fig, ax = plt.subplots(4, 1)
ax[0].plot(time_traj,np.rad2deg(roll_traj), label='Roll JSBSIM')
ax[0].plot(sitl_time, np.rad2deg(roll_sitl), label='Roll SITL')
ax[0].plot(time_traj, np.rad2deg(r_cmd_traj), label='Roll Cmd', linestyle='dashed')

ax[1].plot(time_traj,np.rad2deg(pitch_traj), label='Pitch')
ax[1].plot(sitl_time, np.rad2deg(pitch_sitl), label='Pitch SITL') 


heading_sitl_ned = [enu_heading_to_ned_rads(h) for h in heading_sitl]
ax[2].plot(time_traj,np.rad2deg(heading_traj), label='Heading')
ax[2].plot(sitl_time, np.rad2deg(heading_sitl_ned), label='Heading SITL')
ax[2].plot(time_traj, np.rad2deg(h_cmd_traj), label='Heading Cmd')

ax[3].plot(time_traj,airspeed_traj, label='Airspeed')
ax[3].plot(sitl_time, airspeed_sitl, label='Airspeed SITL')

for a in ax:
    a.legend()

plt.show()
