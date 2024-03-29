import numpy as np

class PID:

    """
    A Proportional-Integral-Derivative (PID) controller class for controlling system responses.
    
    Attributes:
        pid_params (dict): Parameters for PID controller including 'kp' (proportional gain),
                           'ki' (integral gain), 'kd' (derivative gain), and 'dt' (time step).
        old_error (float): Previous error value to compute derivative and integral of error.
        use_control_constraints (bool, optional): Flag to use control constraints. Defaults to False.
        control_constraints (dict, optional): Constraints for the control output including 'max' and 'min' values.

    Methods:
        __init__(self, pid_params: dict, init_error: float, use_control_constraints: bool = False,
                 control_constraints: dict = None): Initializes the PID controller with parameters,
                 initial error, and optionally control constraints.
        init_pid_gains(self, pid_params: dict): Initializes PID gains from the provided parameters.
        compute_derivative_error(self, error: float): Computes the derivative of the error.
        compute_integral_error(self, error: float): Computes the integral of the error using the trapezoidal rule.
        compute_output(self, error: float): Computes the PID controller output based on current error,
                 applies control constraints if enabled, and updates the old error.
    
    The PID controller is used to calculate an output value that is proportional to the current error value.
    It integrates the past errors to eliminate the steady-state error and differentiates the current error
    to predict future errors, thereby improving the control system's stability and response time.
    """
    
    def __init__(self, pid_params:dict, init_error:float, 
                 use_control_constraints:bool=False,
                 control_constraints:dict=None) -> None:
        self.pid_params = pid_params
        
        self.init_pid_gains()        
        self.old_error = init_error
        self.use_control_constraints = use_control_constraints
        self.control_constraints = control_constraints
                
                
    def init_pid_gains(self) -> None:
        if self.pid_params['kp'] is not None:
            self.kp = self.pid_params['kp']
        else:
            print("kp is not specified, setting to 1")
            self.kp = 1.0
            
        if self.pid_params['ki'] is not None:
            self.ki = self.pid_params['ki']
        else:
            #note to user: if ki is not specified, it is set to 0
            print("ki is not specified, setting to 0")
            self.ki = 0.0
            
        if self.pid_params['kd'] is not None:
            self.kd = self.pid_params['kd']
        else:
            print("kd is not specified, setting to 0")
            self.kd = 0.0
            
        if self.pid_params['dt'] is not None:
            self.dt = self.pid_params['dt']
        else:
            print("dt is not specified, setting to 0.01")
            self.dt = 0.01
            
    def compute_derivative_error(self, error:float) -> float:
        #compute the derivative of the error
        derivative_error = (error - self.old_error)/self.dt
        return derivative_error
    
    def compute_integral_error(self, error:float) -> float:
        #compute the integral of the error using the trapezoidal rule
        total_error = error + self.old_error
        integral_error = (total_error * self.dt)/2
        return integral_error

    def compute_output(self, error:float) -> float:
        #compute the output of the PID controller
        p = self.kp * error
        i = self.ki * self.compute_integral_error(error)
        d = self.kd * self.compute_derivative_error(error)
        
                
        output = p + i + d
        
        if self.use_control_constraints:
            if output > self.control_constraints['max']:
                output = self.control_constraints['max']
            elif output < self.control_constraints['min']:
                output = self.control_constraints['min']
                
        self.old_error = error
        
        return output
            
