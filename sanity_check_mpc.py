"""
Sanity check to see if MPC works with the 
JSBSim model environment

Send a desired location to the plane and see if it can reach it 
using the MPC controller

"""
import casadi as ca
import numpy as np
from src.models.Plane import Plane
from opt_control.PlaneOptControl import PlaneOptControl
from jsbsim_backend.aircraft import Aircraft, x8

import gymnasium as gym
import math 
from src.sim_interface import OpenGymInterface
from src.conversions import meters_to_feet, mps_to_ktas

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

def cartesian_to_navigation_radians( 
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

init_state_dict = {
    "ic/u-fps": mps_to_ktas(25),
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


gym_adapter = OpenGymInterface(init_conditions=init_state_dict,
                                 aircraft=x8,
                                 flight_dynamics_sim_hz=200,
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
    -150,
    -150,
    50,
    0,
    0,
    0,
    25
]

N = 500
x_traj = []
y_traj = []
roll_traj = []
pitch_traj = []
heading_traj = []

for i in range(N):
    solution_results, end_time = mpc_control.get_solution(
        init_states, 
        final_states,
        init_control)

    idx_step = 1
    autopilot = gym_adapter.autopilot
    x_cmd = solution_results['x'][idx_step]
    y_cmd = solution_results['y'][idx_step]
    z_cmd = solution_results['z'][idx_step]
    roll_cmd = solution_results['phi'][idx_step]
    pitch_cmd = solution_results['theta'][idx_step]
    heading_cmd = solution_results['psi'][idx_step]
    airspeed_cmd = solution_results['v_cmd'][idx_step]
    print("heading cmd deg: ", np.rad2deg(heading_cmd))
    print("roll cmd deg: ", np.rad2deg(roll_cmd))

    for j in range(10):
        ## if you want to control the heading based on line of sight
        dy = final_states[1] - init_states[1]
        dx = final_states[0] - init_states[0]
        arctan_cmd = np.arctan2(dy, dx)
        print("x and y cmd: ", x_cmd, y_cmd)
        print("heading cmd deg: ", np.rad2deg(arctan_cmd))
        heading_cmd = cartesian_to_navigation_radians(arctan_cmd)
        autopilot.heading_hold(np.rad2deg(heading_cmd)) 
        
        ## if you want to control the heading based on the MPC solution
        # dy = y_cmd - init_states[1]
        # dx = x_cmd - init_states[0]
        # arctan_cmd = np.arctan2(dy, dx)
        # heading_cmd = cartesian_to_navigation_radians(heading_cmd)
        # autopilot.heading_hold(np.rad2deg(heading_cmd))
        # autopilot.roll_hold(roll_cmd)
        
        autopilot.altitude_hold(meters_to_feet(50))
        autopilot.airspeed_hold_w_throttle(mps_to_ktas(15))
        gym_adapter.run_backend()
        init_states = gym_adapter.get_observation()
        init_control =[
            init_states[3],
            init_states[4],
            init_states[5],
            init_states[6]
        ]
        # print("sim time: ", gym_adapter.sim.get_time())
        print("states: ", init_states)
        
        x_traj.append(init_states[0])
        y_traj.append(init_states[1])
        roll_traj.append(init_states[3])    
        pitch_traj.append(init_states[4])
        heading_traj.append(init_states[5])
        
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x_traj, y_traj)
ax.scatter(final_states[0], final_states[1], color='red')
# ax.set_xlim([-20, 20])

fig, ax = plt.subplots(3, 1)
ax[0].plot(np.rad2deg(roll_traj), label='Roll')
ax[1].plot(np.rad2deg(pitch_traj), label='Pitch')
ax[2].plot(np.rad2deg(heading_traj), label='Heading')

plt.show()