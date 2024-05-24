import numpy as np

import casadi as ca

from matplotlib import pyplot as plt


class Plane():
    def __init__(self, 
                 include_time:bool=False,
                 dt_val:float=0.05,
                 max_roll_dg:float=45,
                 max_pitch_dg:float=25) -> None:
        self.include_time = include_time
        self.dt_val = dt_val
        self.define_states()
        self.define_controls() 
        
        self.max_roll_rad = np.deg2rad(max_roll_dg)
        self.max_pitch_rad = np.deg2rad(max_pitch_dg)
        
    def define_states(self):
        """define the states of your system"""
        #positions off the world in NED Frame
        self.x_f = ca.SX.sym('x_f')
        self.y_f = ca.SX.sym('y_f')
        self.z_f = ca.SX.sym('z_f')

        #attitude
        self.phi_f = ca.SX.sym('phi_f')
        self.theta_f = ca.SX.sym('theta_f')
        self.psi_f = ca.SX.sym('psi_f')
        self.v = ca.SX.sym('v')

        if self.include_time:
            self.states = ca.vertcat(
                self.x_f,
                self.y_f,
                self.z_f,
                self.phi_f,
                self.theta_f,
                self.psi_f, 
                self.v)
        else:
            self.states = ca.vertcat(
                self.x_f,
                self.y_f,
                self.z_f,
                self.phi_f,
                self.theta_f,
                self.psi_f,
                self.v 
            )

        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """
        controls for your system
        The controls are the roll, pitch, yaw, and airspeed
        Pitch is a little weird, if you send positive 
        """
        self.u_phi = ca.SX.sym('u_phi')
        self.u_theta = ca.SX.sym('u_theta')
        self.u_psi = ca.SX.sym('u_psi')
        self.v_cmd = ca.SX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0] 

    def set_state_space(self):
        """
        define the state space of your system
        NED Frame
        """
        self.g = 9.81 #m/s^2
        #body to inertia frame 
        self.x_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.cos(self.psi_f) 
        self.y_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)
        
        self.phi_fdot   = self.u_phi 
        self.theta_fdot = self.u_theta
        
        #check if the denominator is zero
        # self.v_cmd = ca.if_else(self.v_cmd == 0, 1e-6, self.v_cmd)
        self.v_dot = ca.sqrt(self.x_fdot**2 + self.y_fdot**2 + self.z_fdot**2)
        self.psi_fdot   = self.u_psi + (self.g * (ca.tan(self.phi_f) / self.v_cmd))

        # self.t_dot = self.t 
        
        if self.include_time:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.z_fdot,
                self.phi_fdot,
                self.theta_fdot,
                self.psi_fdot,
                self.v_dot
            )
        else:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.z_fdot,
                self.phi_fdot,
                self.theta_fdot,
                self.psi_fdot,
                self.v_dot
            )

        #ODE function
        self.function = ca.Function('f', 
            [self.states, self.controls], 
            [self.z_dot])
        
        
    def rk45(self, x, u, dt, use_numeric:bool=True):
        """
        Runge-Kutta 4th order integration
        x is the current state
        u is the current control input
        dt is the time step
        use_numeric is a boolean to return the result as a numpy array
        """
        k1 = self.function(x, u)
        k2 = self.function(x + dt/2 * k1, u)
        k3 = self.function(x + dt/2 * k2, u)
        k4 = self.function(x + dt * k3, u)
        
        next_step = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        #clip the values of the angles
        next_step[3] = np.clip(next_step[3], 
                               -self.max_roll_rad, 
                               self.max_roll_rad)
        next_step[4] = np.clip(next_step[4], 
                               -self.max_pitch_rad, 
                               self.max_pitch_rad)
                       
        #return as numpy row vector
        if use_numeric:
            next_step = np.array(next_step).flatten()
            return next_step
        else:
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
plane_example = Plane()
function = plane_example.set_state_space()
start_states = [
    0, 0, 0, #position 
    0, 0, 0, 0 #attitude and velocity
]
u = [np.deg2rad(0), #roll control 
     np.deg2rad(5), #pitch control
     0, #yaw control
     25] #airspeed

N = 50
dt = 0.01

T_end = 25
N = int(T_end/dt)

next_step = start_states

history = {
    'x':[],
    'y':[],
    'z':[],
    'phi':[],
    'theta':[],
    'psi':[],
}

for i in range(N):
    next_step = plane_example.rk45(next_step, u, dt)
    print(next_step)
    history['x'].append(next_step[0])
    history['y'].append(next_step[1])
    history['z'].append(-next_step[2]) #flip this to make z go up 
    history['phi'].append(np.rad2deg(next_step[3]))
    history['theta'].append(np.rad2deg(next_step[4]))
    history['psi'].append(np.rad2deg(next_step[5]))
    

### These are plots for sanity checks is all    
# import matplotlib.pyplot as plt
# # plot in 3D
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
# ax.plot(history['x'], history['y'], history['z'])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# fig, ax = plt.subplots(3, 1, figsize=(10, 10))
# ax[0].plot(history['phi'], label='phi')
# ax[1].plot(history['theta'], label='theta')
# ax[2].plot(history['psi'], label='psi')

# for a in ax:
#     a.legend()
#     a.grid()

# plt.show()
