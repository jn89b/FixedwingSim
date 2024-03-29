import casadi as ca
import numpy as np
#from OptiMPC import OptiCasadi

from controls.OptiMPC import OptiCasadi

class PlaneMPC(OptiCasadi):
    """
    """
    def __init__(self, mpc_params: dict, 
                 control_constraints:dict,
                 state_constraints:dict,
                 control_indices:dict,
                 state_indices:dict,
                 casadi_model,
                 use_minimize_time:bool=False,
                 use_obs_avoidance:bool=False,
                 obs_avoid_params:dict=None,
                 use_dynamic_threats:bool=False,
                 dynamic_threat_params:dict=None) -> None:
        super().__init__(mpc_params, casadi_model)
        
        self.x0 = self.opti.parameter(self.casadi_model.n_states)
        self.u0 = self.opti.parameter(self.casadi_model.n_controls)
        self.xF = self.opti.parameter(self.casadi_model.n_states)
        
        self.control_constraints = control_constraints
        self.state_constraints = state_constraints
        
        self.control_indices = control_indices
        self.state_indices = state_indices
        
        self.init_mpc_params()
        self.init_decision_variables()
        self.set_init_constraints()
        
        #other parameters 
        self.use_minimize_time = use_minimize_time
        if use_minimize_time:
            self.T = self.opti.variable()
            # self.dt = self.T / self.N
        
        self.use_obs_avoidance = use_obs_avoidance
        self.obs_avoid_params = obs_avoid_params
        
        self.use_dynamic_threats = use_dynamic_threats
        self.dynamic_threat_params = dynamic_threat_params
        
    def set_init_constraints(self) -> None:
        pass
        #self.opti.subject_to(self.X[:,0] == self.x0)
    
    # def set_init_final_states(self, x0:np.array, xf:np.array) -> None:
    def set_init_final_states(self, init_state:np.ndarray, 
                              final_state:np.ndarray) -> None:
        init_state = init_state.tolist()
        final_state = final_state.tolist()
        self.opti.set_value(self.x0, init_state)
        self.opti.set_value(self.xF, final_state)
    
    def set_init_controls(self, u0:np.array) -> None:
        u0 = u0.tolist()
        self.opti.set_value(self.u0, u0)
    
    def set_state_constraints(self) -> None:
        state_constraints = self.state_constraints
        state_indices = self.state_indices
        
        print('State indices', state_indices)        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['x_dot_min'],
            self.X[state_indices['x_dot'],:],
            state_constraints['x_dot_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['y_dot_min'],
            self.X[state_indices['y_dot'],:],
            state_constraints['y_dot_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['z_dot_min'],
            self.X[state_indices['z_dot'],:],
            state_constraints['z_dot_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['phi_min'],
            self.X[state_indices['phi_dot'],:],
            state_constraints['phi_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['theta_min'],
            self.X[state_indices['theta_dot'],:],
            state_constraints['theta_max']))
        
        self.opti.subject_to(self.opti.bounded(
            state_constraints['psi_min'],
            self.X[state_indices['psi_dot'],:],
            state_constraints['psi_max']))
        
    def set_control_constraints(self) -> None:
        """
        """
        ctrl_constraints = self.control_constraints
        ctrl_indices = self.control_indices
        print('Control indices', ctrl_indices)

        u_phi = self.U[ctrl_indices['u_phi'],:]
        u_theta = self.U[ctrl_indices['u_theta'],:]
        u_psi = self.U[ctrl_indices['u_psi'],:]
        v_cmd = self.U[ctrl_indices['v_cmd'],:]
        
        self.opti.subject_to(self.opti.bounded(
            ctrl_constraints['u_phi_min'],
            u_phi,
            ctrl_constraints['u_phi_max']))
        
        self.opti.subject_to(self.opti.bounded(
            ctrl_constraints['u_theta_min'],
            u_theta,
            ctrl_constraints['u_theta_max']))
        
        self.opti.subject_to(self.opti.bounded(
            ctrl_constraints['u_psi_min'],
            u_psi,
            ctrl_constraints['u_psi_max']))
        
        self.opti.subject_to(self.opti.bounded(
            ctrl_constraints['v_cmd_min'],
            v_cmd,
            ctrl_constraints['v_cmd_max']))
    
    def set_dynamic_constraint(self) -> None:
        self.opti.subject_to(self.X[:,0] == self.x0)
        
        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            state_next = self.X[:, k+1]
            # time_state =     
            ##Runge Kutta
            k1 = self.casadi_model.function(states, controls)
            k2 = self.casadi_model.function(states + self.dt/2*k1, controls)
            k3 = self.casadi_model.function(states + self.dt/2*k2, controls)
            k4 = self.casadi_model.function(states + self.dt * k3, controls)
            state_next_RK4 = states + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            if self.use_minimize_time:
                self.X[-1, k+1] = self.X[-1, k] + self.dt  
            self.opti.subject_to(state_next == state_next_RK4)

    def compute_obstacle_cost(self, cost, x_position, y_position) -> None:
        obs_avoidance_weight = self.obs_avoid_params['weight']
        obs_x_vector = self.obs_avoid_params['x']
        obs_y_vector = self.obs_avoid_params['y']
        obs_radii_vector = self.obs_avoid_params['radii']
        safe_distance = self.obs_avoid_params['safe_distance']
                
        for i,x in enumerate(obs_x_vector):
            obs_x = obs_x_vector[i]
            obs_y = obs_y_vector[i]
            obs_radii = obs_radii_vector[i]

            obstacle_constraints = self.opti.variable(self.N+1)
            obstacle_distance = -ca.sqrt((obs_x - x_position)**2 + \
                (obs_y - y_position)**2)
            obstacle_distance = 1/obstacle_distance
            self.opti.subject_to(obstacle_distance.T >= \
                obstacle_constraints + obs_radii + safe_distance)

            avoidance_cost = obs_avoidance_weight * ca.sumsqr(obstacle_constraints)
        
        return avoidance_cost

    def compute_dynamic_threats_cost(self, cost:float) -> None:
        #ego vehicle
        print('Computing dynamic threats cost')
        x_pos    = self.X[0, :]
        y_pos    = self.X[1, :]
        z_pos    = self.X[2, :]
        psi      = self.X[5, :]
        ego_time = self.X[6, :]        
        #this is a list of dynamic threats refer to the Threat class
        threat_list = self.dynamic_threat_params['threats'] 
        num_threats = len(threat_list)
        
        for threat in threat_list:
            threat_x_traj = threat.x_traj
            threat_y_traj = threat.y_traj
            # threat_z_traj = threat.z_traj
            threat_psi_traj = threat.psi_traj
            threat_time_traj = threat.time_traj
            #make sure that the threat trajectory is the same length as the ego vehicle
            if len(threat_x_traj) != self.N+1:
                raise ValueError('Threat trajectory must be the same length as the ego vehicle', 
                                 len(threat_x_traj), self.N+1)
                
                
            threat_parameter = self.opti.parameter(self.N+1)
            threat_constraints = self.opti.variable(self.N+1)

            # distance = ca.sqrt((x_pos - threat_x_traj)**2 + \
            #     (y_pos - threat_y_traj)**2 + (z_pos - threat_z_traj)**2)

            distance = ca.sqrt((x_pos.T - threat_x_traj)**2 + \
                (y_pos.T - threat_y_traj)**2)

            #avoid division by zero
            distance = ca.if_else(distance < 1E-3, 1E-3, distance)
            distance_cost = 1/(distance)
   
            threat_radii = 1.0 
            safe_distance = 1
            
            
            #add the cost to the total cost
            threat_cost = self.dynamic_threat_params['weight'] * ca.sumsqr(distance_cost)
            cost += threat_cost
            
        return cost
            
    def set_cost_function(self, x_final:np.ndarray) -> None:
        Q = self.mpc_params['Q']
        R = self.mpc_params['R']
        cost = 0
            
        x_position = self.X[0, :]
        y_position = self.X[1, :]
        z_position = self.X[2, :]
        phi = self.X[3, :]
        theta = self.X[4, :]
        psi = self.X[5, :]
        
        #unit vector in the direction of the plane
        unit_vector_ego = ca.horzcat(ca.cos(psi), ca.sin(psi))
    
        #use obstacle avoidance if set
        if self.use_obs_avoidance and self.obs_avoid_params is not None:
            cost += self.compute_obstacle_cost(cost, x_position, y_position)
        
        if self.use_dynamic_threats and self.dynamic_threat_params is not None:
            cost += self.compute_dynamic_threats_cost(cost)
        
        #check if x_final is a numpy array
        if isinstance(x_final, np.ndarray):
           #turn into list
              x_final = x_final.tolist()
               
        error_x = x_final[0] - x_position
        error_y = x_final[1] - y_position
        
        sum_ex = ca.sumsqr(error_x) * 0.1
        sum_ey = ca.sumsqr(error_y) * 0.1
        cost += sum_ex + sum_ey
        print('Cost function set', cost)
        if self.use_minimize_time:
            #add constraint that time is minimized
            #self.opti.subject_to(self.X[6, 0] >= 0)
            self.opti.subject_to(self.T >= 0)
            self.opti.set_initial(self.T, 3.0)
            # self.opti.minimize(self.T)
            weight_time = 1E10
            self.opti.subject_to(self.X[6, 0] == 0)
            self.opti.subject_to(self.X[6, -1] == self.T)
            self.opti.minimize(weight_time * self.T)
        else:
            self.opti.minimize(cost)            

    def init_optimal_control_prob(self, start_states:np.array,
                                end_states:np.array) -> None:
        self.set_init_final_states(start_states, end_states)
        self.set_control_constraints()
        self.set_state_constraints()
        self.set_dynamic_constraint()
        self.set_cost_function(end_states)
        self.set_solution_options()
        print('Optimal control problem initialized')

    def solve(self) -> ca.OptiSol:
        sol = self.opti.solve()
        return sol
