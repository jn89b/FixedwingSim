import numpy as np
import casadi as ca
import pyDOE

class ToyCar():
    """
    Toy Car Example 
    
    3 States: 
    [x, y, psi]
     
     2 Inputs:
     [v, psi_rate]
    
    """
    def __init__(self,
                 dt_val:float=0.05,
                 psi_tau:float=0.05) -> None:
        self.define_states()
        self.define_controls()
        self.dt_val = dt_val
        self.psi_tau = psi_tau
        
    def define_states(self) -> None:
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.psi = ca.SX.sym('psi')
        
        self.states = ca.vertcat(
            self.x,
            self.y,
            self.psi
        )
        #column vector of 3 x 1
        self.n_states = self.states.size()[0] #is a column vector 
        
    def define_controls(self) -> None:
        self.v_cmd = ca.SX.sym('v_cmd')
        self.psi_cmd = ca.SX.sym('psi_cmd')
        
        self.controls = ca.vertcat(
            self.v_cmd,
            self.psi_cmd
        )
        #column vector of 2 x 1
        self.n_controls = self.controls.size()[0] 
        
    def set_state_space(self) -> None:
        #this is where I do the dynamics for state space
        self.x_dot = self.v_cmd * ca.cos(self.psi)
        self.y_dot = self.v_cmd * ca.sin(self.psi)
        #self.psi_dot = self.psi_cmd 
        
        self.psi_dot = (self.psi_cmd - self.psi) * \
            self.dt_val/self.psi_tau
        
        self.z_dot = ca.vertcat(
            self.x_dot, self.y_dot, self.psi_dot    
        )
        
        #ODE right hand side function
        self.function = ca.Function('dynamics', 
                        [self.states, self.controls],
                        [self.z_dot]
                    ) 
        
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
        
        if next_step[2] > np.pi:
            next_step[2] -= 2*np.pi
        elif next_step[2] < -np.pi:
            next_step[2] += 2*np.pi
                    
        #return as numpy row vector
        if use_numeric:
            next_step = np.array(next_step).flatten()
            return next_step
        else:
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        # 
    def generate_training_data(self, N:int, dt:float, constraint_params:dict, 
                               noise:bool=True, noise_val:float=1e-2, noise_deg:float=0.5):
        """ Generate training data using latin hypercube design
        # Arguments:
            N:   Number of data points to be generated
            uub: Upper input range (Nu,1)
            ulb: Lower input range (Nu,1)
            xub: Upper state range (Ny,1)
            xlb: Lower state range (Ny,1)

        # Returns:
            Z: Matrix (N, Nx + Nu) with state x and inputs u at each row
            Y: Matrix (N, Nx) where each row is the state x at time t+dt,
                with the input from the same row in Z at time t.
        """
        # Make sure boundary vectors are numpy arrays
        N = int(N)
        
        R = np.eye(self.n_states) * noise_val
        
        #check if the constraint parameters are a numpy array
        for key, value in constraint_params.items():
            if not isinstance(value, np.ndarray):
                constraint_params[key] = np.array(value)
        
        xlb = constraint_params["lower_state_bounds"]
        xub = constraint_params["upper_state_bounds"]
        ulb = constraint_params["lower_input_bounds"]
        uub = constraint_params["upper_input_bounds"]
        
        #state outputs with noise
        Y = np.zeros((N, self.n_states))
        
        # create control input design using a latin hypercube
        # latin hypercube dsign for unit cube [0,1]
        if self.n_controls > 0:
            U = pyDOE.lhs(self.n_controls, samples=N, criterion='maximin')
            for k in range(N):
                U[k] = U[k]*(uub - ulb) + ulb
        else:
            U = np.zeros((N, self.n_controls))
            
        #create state input design using latin hypercube doing the same thing
        X = pyDOE.lhs(self.n_states, samples=N, criterion='maximin')
        
        #scale the state inputs to the actual state space
        for k in range(N):
            X[k] = X[k]*(xub - xlb) + xlb
            
        num_parameters = 0
        parameters = pyDOE.lhs(num_parameters, samples=N)
        
        Y_no_noise = np.zeros((N, self.n_states))
        
        unexplained_x = 0.0 #* np.random.uniform(-1, 1, (N, self.n_states))
        unexplained_y = 0.0 #* np.random.uniform(-1, 1, (N, self.n_states))
        

        for i in range(N):
            unexplained_psi = np.random.uniform(np.deg2rad(5), np.deg2rad(10))
            if self.n_controls > 0:
                Y_no_noise[i,:] = self.rk45(X[i], U[i], dt)
                Y[i,:] = self.rk45(X[i], U[i], dt)
            else:
                Y_no_noise[i,:] = self.rk45(X[i], np.array([]), dt)
                Y[i,:] = self.rk45(X[i], np.array([]), dt)
            
            if noise:
                #add white noise to the state outputs
                Y[i,0] += np.random.uniform(-noise_val, noise_val) + unexplained_x
                Y[i,1] += np.random.uniform(-noise_val, noise_val) + unexplained_y
                noise_rad = np.random.uniform(-np.deg2rad(noise_deg), np.deg2rad(noise_deg)) + unexplained_psi
                Y[i,2] += noise_rad
                #Y[i,:] += np.random.multivariate_uniform(np.zeros(self.n_states), R)
            
            
        #concatenate the state and control inputs
        if self.n_controls > 0:
            Z = np.hstack((X, U))
        else:
            Z = X
        return Z, Y, Y_no_noise            
