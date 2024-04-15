"""
Simulate pursuer and evader drones
"""
import casadi as ca
import numpy as np
import math 
from src.models.Plane import Plane
from jsbsim_backend.simulator import FlightDynamics
from jsbsim_backend.aircraft import Aircraft, x8
from guidance_control.autopilot import X8Autopilot
from opt_control.PlaneOptControl import PlaneOptControl
from sim_interface import CLSimInterface
from conversions import feet_to_meters, meters_to_feet, ktas_to_mps, mps_to_ktas
from conversions import local_to_global_position

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
        
        return pro_nav, los_rate

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
        pursuer_vel_mag = current_states[-1]
        
        dx = target_states[0] - current_states[0]
        dy = target_states[1] - current_states[1]
        los = np.arctan2(dy, dx)        
        target_vel = target_states[-1]
        target_vx = target_vel * np.cos(target_states[5])
        target_vy = target_vel * np.sin(target_states[5])
        
        dt = 1/self.flight_dynamics_sim_hz
        # target_next_x = target_states[0] + (target_vx*dt)
        # target_next_y = target_states[1] + (target_vy*dt)
                
        # pursuer_vx = pursuer_vel_mag * np.cos(current_states[5])
        # pursuer_vy = pursuer_vel_mag * np.sin(current_states[5])
                        
        los = self.cartesian_to_navigation_radians(los)
        yaw = current_states[5]
        error_los = abs(los - yaw)
        print("error_los: ", error_los)

        if error_los > np.deg2rad(20):
            print("sending 15")
            vel_cmd = 15
        else:
            vel_cmd = 40
            
        vel_cmd = np.clip(vel_cmd, 15, 40)
        
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
        
        #limit the velocity command
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
            self.sim = FlightDynamics(aircraft=self.aircraft, 
                                      init_conditions=init_conditions)
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


evader_position = [0, 0, 50]
evader_geo_position = local_to_global_position(evader_position)
evader_state_dict = {
    "ic/u-fps": mps_to_ktas(25),
    "ic/v-fps": 0.0,
    "ic/w-fps": 0.0,
    "ic/p-rad_sec": 0.0,
    "ic/q-rad_sec": 0.0,
    "ic/r-rad_sec": 0.0,
    "ic/h-sl-ft": meters_to_feet(evader_position[2]),
    "ic/long-gc-deg": evader_geo_position[0],
    "ic/lat-gc-deg": evader_geo_position[1],
    "ic/psi-true-deg": 45,
    "ic/theta-deg": 0.0,
    "ic/phi-deg": 0.0,
    "ic/alpha-deg": 0.0,
    "ic/beta-deg": 0.0,
    "ic/num_engines": 1,
}

mpc_params = {
    'N': 10,
    'Q': ca.diag([1.0, 1.0, 1.0, 0, 0, 0.0, 0.0]),
    'R': ca.diag([0.1, 0.1, 0.1, 0.1]),
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

evader_model = Plane()
evader_model.set_state_space()
mpc_control = init_mpc_controller(
    mpc_control_constraints=control_constraints,
    state_constraints=state_constraints,
    mpc_params=mpc_params,
    plane_model=evader_model)

evader = PursuerInterface(
    init_conditions=evader_state_dict,
    evader_position=evader_position,
    control_constraints=control_constraints,
)

# generate a list of flight dynamic simulators for the pursuers
num_pursuers = 1
pursuer_list = []
min_distance = 50
for i in range(num_pursuers):
    rand_x = np.random.uniform(-100, 100)
    rand_y = np.random.uniform(-100, 100)
    rand_z = np.random.uniform(30, 100)
    rand_vel = np.random.uniform(15, 30)
    rand_x = 0
    rand_y = -50
    rand_z = 50 
    
    #make sure not too close to evader
    dist = np.sqrt((rand_x - evader_position[0])**2 +  \
        (rand_y - evader_position[1])**2)
    pursuer_position = [rand_x, rand_y, rand_z]
    pursuer_geo_position = local_to_global_position(pursuer_position)
    print(pursuer_geo_position)
    pursuer_init_conditions ={
        "ic/u-fps": mps_to_ktas(rand_vel),
        "ic/v-fps": 0.0,
        "ic/w-fps": 0.0,
        "ic/p-rad_sec": 0.0,
        "ic/q-rad_sec": 0.0,
        "ic/r-rad_sec": 0.0,
        "ic/h-sl-ft": meters_to_feet(pursuer_geo_position[2]),
        "ic/long-gc-deg": pursuer_geo_position[0],
        "ic/lat-gc-deg": pursuer_geo_position[1],
        "ic/psi-true-deg": 90,
        "ic/theta-deg": 0.0,
        "ic/phi-deg": 0.0,
        "ic/alpha-deg": 0.0,
        "ic/beta-deg": 0.0,
        "ic/num_engines": 1,
    }

    pursuer = PursuerInterface(
        init_conditions=pursuer_init_conditions,
        evader_position=evader_position,
        control_constraints=control_constraints,
    )
    
    pursuer_list.append(pursuer)
    
#test the pursuer interface
N_steps = 1000
pursuer_x_history = []
pursuer_y_history = []
pursuer_z_history = []

evader_x_history = []
evader_y_history = []
evader_z_history = []

dt = 1/pursuer.flight_dynamics_sim_hz
turn_cmd_history = []
for i in range(N_steps):
    #evader_position #= evader.get_observation()[0:3]

    evader_states = evader.get_observation()
    evader_x = evader_states[0]
    evader_y = evader_states[1]
    evader_z = evader_states[2]
    
    evader_states = evader.get_observation()
    for p in pursuer_list:

        turn_cmd, v_cmd = p.pursuit_nav(evader_states)
        
        pursuer_position = p.get_observation()[0:3]
        pursuer_vel = p.get_observation()[-1] + v_cmd
        print("vcmd: ", v_cmd)
        p.set_command(turn_cmd, v_cmd, 50)
        turn_cmd_history.append(turn_cmd)
        
        pursuer_x_history.append(pursuer_position[0])
        pursuer_y_history.append(pursuer_position[1])
        pursuer_z_history.append(pursuer_position[2])
        
        evader.set_command(np.deg2rad(10), 20, 50) 

        evader_x_history.append(evader_x)
        evader_y_history.append(evader_y)
        evader_z_history.append(evader_z)
        evader_position = [evader_x, evader_y, evader_z]
        
        distance = np.sqrt((pursuer_position[0] - evader_position[0])**2 + \
            (pursuer_position[1] - evader_position[1])**2)
        print("distance: ", distance)
        
    if distance < 5:
        print("evader caught")
        break
            
#plot position and heading of pursuers and evader
import matplotlib.pyplot as plt
#turn off latex rendering
plt.rcParams.update(plt.rcParamsDefault)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print("sim time (s): ", p.sim.get_time())

ax.scatter(evader_x_history[0], evader_y_history[1], 
           evader_z_history[2], color='b')
ax.scatter(pursuer_x_history[0], pursuer_y_history[0], 
           pursuer_z_history[0], color='r')
ax.plot(pursuer_x_history, pursuer_y_history, 
        pursuer_z_history, color='r', alpha=0.5, label='pursuer')
ax.plot(evader_x_history, evader_y_history, 
        evader_z_history, color='b', alpha=0.5, label='evader')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

fig, ax = plt.subplots()
ax.plot(np.rad2deg(turn_cmd_history))

plt.show()