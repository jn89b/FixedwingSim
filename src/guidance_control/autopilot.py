import numpy as np
import jsbsim_backend.properties as prp
import math
import control

from simple_pid import PID
from jsbsim_backend.simulator import FlightDynamics
from scipy import interpolate
from guidance_control.navigation import LocalNavigation
from conversions import mps_to_ktas, meters_to_feet, feet_to_meters

# Should this be derived from simulation ?
# def __init__(self):
#     super().__init__()

def interpolate(x:float, x_min:float, x_max:float, 
                y_min:float, y_max:float) -> float:
    """
    Linear interpolation function
    """
    return y_min + (y_max - y_min) * ((x - x_min) / (x_max - x_min))


class TECSParameters():
    def __init__(self) -> None:
       # Vehicle specific params
        self.max_sink_rate = 0.0          # Maximum sink rate [m/s]
        self.min_sink_rate = 0.0          # Minimum sink rate [m/s]
        self.max_climb_rate = 0.0         # Climb rate at max throttle [m/s]
        self.vert_accel_limit = 0.0       # Max vertical acceleration [m/s²]
        self.equivalent_airspeed_trim = 0.0  # Cruise airspeed [m/s]
        self.tas_min = 0.0                # Lower limit of true airspeed demand [m/s]
        self.pitch_max = 0.0              # Max pitch angle [rad]
        self.pitch_min = 0.0              # Min pitch angle [rad]
        self.throttle_trim = 0.0          # Normalized throttle at level flight [0,1]
        self.throttle_max = 0.0           # Upper throttle limit [0,1]
        self.throttle_min = 0.0           # Lower throttle limit [0,1]

        # Altitude control params
        self.altitude_error_gain = 0.0    # Altitude error inverse time constant [1/s]
        self.altitude_setpoint_gain_ff = 0.0  # Gain from altitude demand derivative to climb rate

        # Airspeed control params
        self.tas_error_percentage = 0.0   # Percentage of airspeed tracking errors [0,1]
        self.airspeed_error_gain = 0.0    # Airspeed error inverse time constant [1/s]

        # Energy control params
        self.ste_rate_time_const = 0.0    # Time constant for energy rate [s]
        self.seb_rate_ff = 0.0            # Energy balance rate feedforward gain

        # Pitch control params
        self.pitch_speed_weight = 0.0     # Speed control weighting for pitch calculation
        self.integrator_gain_pitch = 0.0  # Integrator gain for pitch demand
        self.pitch_damping_gain = 0.0     # Damping gain for pitch demand [s]

        # Throttle control params
        self.integrator_gain_throttle = 0.0  # Integrator gain for throttle demand
        self.throttle_damping_gain = 0.0  # Damping gain for throttle demand [s]
        self.throttle_slewrate = 0.0      # Throttle demand slew rate [1/s]

        # Load factor parameters
        self.load_factor_correction = 0.0    # Gain from load factor to energy rate [m²/s³]
        self.load_factor = 0.0               # Additional normal load factor

    def display_params(self):
        """A sample method to display parameters for debugging."""
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")


class TECSController():
    def __init__(self, 
                 TECSParams:TECSParameters,
                 dt:float) -> None:
        
        self.TECSParams = TECSParams
        self.dt = dt

    # def update(self, setpoint:Set)

class RateController():
    def __init__(self, 
                 P:float,
                 I:float,
                 D:float,
                 dt:float,
                 FFgain:float,
                 max_I:float,
                 min_I:float) -> None:
        self.P = P
        self.I = I
        self.D = D
        self.dt = dt
        self.FFgain = FFgain
        self.max_I = max_I
        self.min_I = min_I
        self.rate_i = 0.0
        self.is_saturated = False
    
    def update_integral(self, rate_error:float) -> None:
        """
        
        """
        #prevent further positive control saturation
        if rate_error > 0:
            #set minimum rate error to zero
            rate_error = min(rate_error, 0)
        
        #prevent further negative control saturation
        if rate_error < 0:
            #set maximum rate error to zero
            rate_error = max(rate_error, 0)
            
		# // I term factor: reduce the I gain with increasing rate error.
		# // This counteracts a non-linear effect where the integral builds up quickly upon a large setpoint
		# // change (noticeable in a bounce-back effect after a flip).
		# // The formula leads to a gradual decrease w/o steps, while only affecting the cases where it should:
		# // with the parameter set to 400 degrees, up to 100 deg rate error, i_factor is almost 1 (having no effect),
		# // and up to 200 deg error leads to <25% reduction of I.
        i_factor = rate_error / math.radians(400)
        i_factor = max(0.0, 1.0 - i_factor * i_factor)
        
        # perform integral update using first order approximation
        rate_i = self.rate_i + i_factor * self.I * rate_error * self.dt
        
        # do not allow the integrator to accumulate more than the maximum output
        if np.isfinite(rate_i):
            self.rate_i = np.clip(rate_i, self.min_I, self.max_I)
        
    def update(self, rate:float, rate_sp:float, 
               angular_accel:float, dt:float) -> float:
        """
        
        """
        rate_error = rate_sp - rate
        
        torque = (self.P*rate_error) + self.rate_i - (angular_accel*self.D) + \
            (self.FFgain*rate_sp)
        
        self.update_integral(rate_error)
            
        return torque 

class C172Autopilot:
    def __init__(self, sim):
        self.sim = sim
 
    def wing_leveler(self):
        error = self.sim[prp.roll_rad]
        kp = 50.0
        ki = 5.0
        kd = 17.0
        pid = PID(kp, ki, kd)
        output = pid(error)
        self.sim[prp.aileron_cmd] = output

    def hdg_hold(self, hdg):
        error = hdg - self.sim[prp.heading_deg]
        # Limit error to within 180 degrees (left or right)
        if error > 180:
            error = error - 180
        if error < 180:
            error = error + 180
        # Saturate error signal gain to be a maximum of 30 degrees
        if error < -30:
            error = -30
        if error > 30:
            error = 30
        # Convert error signal from degrees to radians
        error = error * (math.pi / 180)
        # Implementing a lag compensator as a single integrator (don't know how to do lag)
        c = 0.5
        hdg_lag = PID(0, c, 0)
        roll_error = hdg_lag(error) - self.sim[prp.roll_rad]
        kp = 6.0
        ki = 0.13
        kd = 6.0
        roll_pid = PID(kp, ki, kd)
        output = roll_pid(roll_error)
        self.sim[prp.aileron_cmd] = output

    def level_hold(self, level_ft:float):
        
        error = level_ft - self.sim[prp.altitude_sl_ft]
        # Limit climb error to a maximum of 100'
        if error > 100:
            error = 100
        if error < -100:
            error = -100
        # Convert error to percentage of maximum
        error = error/100
        # Lag desired climb rate (for stability) as a single integrator
        # c = 1.0
        # vs_lag = PID(0, c, 0)
        # error = vs_lag(error)
        # Gain scheduled climb rate
        ref_alt = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
                   7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0]
        vs_gain = [0.12, 0.11, 0.10, 0.096, 0.093, 0.086, 0.078,
                   0.069, 0.061, 0.053, 0.045, 0.037, 0.028]
        climb_gain_scheduling = interpolate.interp1d(ref_alt, vs_gain)
        vs_dem = error * climb_gain_scheduling(self.sim[prp.altitude_sl_ft])
        vs_error = vs_dem - self.sim[prp.altitude_rate_fps]
        # print('vertical speed error: ', vs_error)
        # Rate PID controller
        kp = 0.01
        ki = 0.00015
        kd = 0.0003
        vs_pid = PID(kp, ki, kd)
        output = vs_pid(vs_error)
        self.sim[prp.elevator_cmd] = output
        # print('elevator command: ', output)


class X8Autopilot:
    """
    The low-level autopilot control for the X8 fixed wing UAV aircraft

     ...

    Attributes:
    -----------
    sim : Simulation object
        an instance of the flight simulation flight dynamic model, used to interface with JSBSim
    nav : LocalNavigation object
        the core position and tracking methods used in the path planning methods
    track_bearing : float
        the bearing from a starting point to a target point [radians]
    track_bearing_in : float
        the bearing from a starting point 'a' to a target point 'b' [radians]
    track_bearing_out : float
        the bearing from a target point to 'b' to the following point 'c' [radians]
    track_distance : float
        the distance from a starting point to a target point [m]
    flag : bool
        a variable returned by a method to indicate a significant change in simulation state or termination condition
    track_id : int
        a counter for the points in a profile
    state: int
        the state or mode an autopilot is currently engaged in

    Methods:
    -------
    test_controls(elevator, aileron, tla)
        allows for manual input of the aircraft's controls
    pitch_hold(pitch_comm)
        maintains a commanded pitch attitude [radians] using a PI controller
    roll_hold(roll_comm)
        maintains a commanded roll attitude [radians] using a PID controller
    heading_hold(heading_comm)
        maintains a commanded heading [degrees] using a PD controller
    airspeed_hold_w_throttle(airspeed_comm)
        maintains a commanded airspeed [KTAS] using throttle_cmd
    altitude_hold(altitude_comm)
        maintains a demanded altitude [feet] using pitch attitude
    home_to_target(target_northing, target_easting, target_alt)
        homes towards a 2D (lat, long) point in space and uses altitude_hold to maintain an altitude
    track_to_target(target_northing, target_easting, target_alt)
        maintains a track from the point the simulation started at to the target point
    track_to_profile(profile)
        maintains a track along a series of points in the simulation and the defined altitude along each path segment



    """
    def __init__(self, sim):
        self.sim = sim
        self.nav = None
        # self.orbit_nav = None
        self.track_bearing = 0
        self.track_bearing_in = 0
        self.track_bearing_out = 0
        self.track_distance = 0
        self.flag = False
        self.track_id = -1
        self.state = 0
        
        ## Parameters for the controller from PX4 controller    
        self.time_const_pitch = 0.4 # seconds
        self.time_const_roll = 0.4 # seconds
        
        self.max_pitch_rate = np.deg2rad(60) # rad/s
        self.max_roll_rate = np.deg2rad(70) # rad/s
        self.max_yaw_rate = np.deg2rad(50) # rad/s

        #trim parameters
        self.trim_roll_rad = 0.0
        self.trim_pitch_rad = 0.0
        self.trim_yaw_rad = 0.0
        

        #these values are set from default PX4 controller
        self.roll_rate_control = RateController(
            P=0.05,
            I=0.1,
            D=0.0,
            dt=self.sim.sim_dt,
            FFgain=0.5,
            max_I=0.2,
            min_I=-0.2
        )
        
        self.pitch_rate_control = RateController(
            P=0.08,
            I=0.1,
            D=0.0,
            dt=self.sim.sim_dt,
            FFgain=0.5,
            max_I=0.1,
            min_I=-0.1
        )
        
        self.yaw_rate_control = RateController(
            P=0.05,
            I=0.1,
            D=0.0,
            dt=self.sim.sim_dt,
            FFgain=0.3,
            max_I=0.1,
            min_I=-0.1
        )

        self.controller_list = [self.roll_rate_control, 
                                self.pitch_rate_control, 
                                self.yaw_rate_control]
        
        self.min_airspeed = 15
        self.trim_airspeed = 25
        self.max_airspeed = 35

    def test_controls(self, elevator=0, aileron=0, tla=0) -> None:
        """
        Directly control the aircraft using control surfaces for the purpose of testing the model

        :param elevator: elevator angle [-30 to +30]
        :param aileron: aileron angle [-30 to +30]
        :param tla: Thrust Lever Angle [0 to 1]
        :return: None
        """
        self.sim[prp.elevator_cmd] = elevator
        self.sim[prp.aileron_cmd] = aileron
        self.sim[prp.throttle_cmd] = tla

    def control_pitch(self, pitch_sp_rad:float, 
                      euler_yaw_rate_sp_rad:float) -> float:
        """
        Based on PX4 controller for pitch attitude
        
        returns pitch_body_rate_sp_rad
        """
        
        pitch_rad = self.sim[prp.pitch_rad]
        roll_rad = self.sim[prp.roll_rad]
        
        pitch_error_rad = pitch_sp_rad - pitch_rad
        euler_rate_setpoint = pitch_error_rad / self.time_const_pitch
        
        #transform setpoint to body angular rates
        pitch_body_rate_sp_raw = np.cos(roll_rad) * euler_rate_setpoint \
            + np.cos(pitch_rad) * np.sin(roll_rad) * euler_yaw_rate_sp_rad

        #limit the rate
        pitch_body_rate_sp_rad = np.clip(pitch_body_rate_sp_raw,
                                            -self.max_pitch_rate,
                                            self.max_pitch_rate)
        
        return pitch_body_rate_sp_rad

    def pitch_controller(self, pitch_comm: float) -> None:
        """
        Based on PX4 controller for pitch attitude
        """
        
        pass

    def pitch_hold(self, pitch_comm: float) -> None:
        """
        Maintains a commanded pitch attitude [radians] using a PI controller with a rate component

        :param pitch_comm: commanded pitch attitude [radians]
        :return: None
        """
        error = pitch_comm - self.sim[prp.pitch_rad]
        #serror = pitch_comm - self.sim.get_property_value('attitude/pitch-rad') #self.sim[prp.pitch_rad]
        kp = 0.6
        ki = 0.1
        kd = 0.05
        controller = PID(kp, ki, kd)
        output = controller(error)
        # self.sim[prp.elevator_cmd] = output
        rate = self.sim[prp.q_radps]
        #rate = self.sim.get_property_value('velocities/q-rad_sec')
        rate_controller = PID(kd, 0.0, 0.0)
        rate_output = rate_controller(rate)
        output = output+rate_output
        #self.sim[prp.elevator_cmd] = output
        #self.sim['fcs/elevator-cmd-norm'] = output
        self.sim[prp.elevator_cmd] = output

    def control_roll(self, roll_sp_rad:float, 
                     euler_yaw_rate_sp_rad:float) -> None:
        """
        Based on PX4 controller for roll attitude
        
        returns roll_body_rate_sp_rad
        """
        roll_rad = self.sim[prp.roll_rad]
        pitch_rad = self.sim[prp.pitch_rad]
        roll_error = roll_sp_rad - roll_rad
        euler_rate_setpoint = roll_error / self.time_const_roll
        
        #transform setpoint to body angular rates
        roll_body_rate_sp_raw = euler_rate_setpoint - np.sin(pitch_rad) \
            * euler_yaw_rate_sp_rad
            
        #limit the rate
        roll_body_rate_sp_rad = np.clip(roll_body_rate_sp_raw, 
                                    -self.max_roll_rate, 
                                    self.max_roll_rate)
        
        return roll_body_rate_sp_rad

    def roll_controller(self, roll_comm: float) -> None:
        """
        Based on PX4 controller for roll attitude
        """
        pass

    
    def roll_hold(self, roll_commd_rad: float) -> None:
        """
        Maintains a commanded roll attitude [radians] using a PID controller

        :param roll_comm: commanded roll attitude [radians]
        :return: None
        """
        # Nichols Ziegler tuning Pcr = 0.29, Kcr = 0.0380, PID chosen
        error = roll_commd_rad - self.sim[prp.roll_rad]
        kp = 0.8
        ki = kp*0.0  # tbd, should use rlocus plot and pole-placement
        kd = 0.089
        controller = PID(kp, ki, 0.0)
        output = controller(error)
        rate = self.sim[prp.p_radps]
        #rate = self.sim.get_property_value('velocities/p-rad_sec')
        rate_controller = PID(kd, 0.0, 0.0)
        rate_output = rate_controller(rate)
        output = -output+rate_output
        self.sim[prp.aileron_cmd] = output

    def control_yaw(self, roll_sp_rad:float, 
                    euler_pitch_rate_sp_rad:float) -> None:
        """
        Based on PX4 controller for yaw attitude
        
        Returns body_rate_sp_rad
        """
        roll = self.sim[prp.roll_rad]
        pitch = self.sim[prp.pitch_rad]
        airspeed = feet_to_meters(self.sim[prp.airspeed])
        
        constrained_roll = None
        inverted_roll = False
        
        #roll is used as a feedforward term and inverted flight needs 
        # to be handled separately
        if np.abs(roll) < np.deg2rad(90):
            constrained_roll = np.clip(roll, -np.deg2rad(80), np.deg2rad(80))
        else:
            inverted_roll = True
            
            #inverted flight, constrain two extremes of -pi to pi to avoid 
            # infinite yaw rate 
            if roll > 0:
                constrained_roll = np.clip(roll, 
                                           np.deg2rad(100), 
                                           np.deg2rad(180))
            else:
                #left hemisphere
                constrained_roll = np.clip(roll,
                                             -np.deg2rad(180),
                                             -np.deg2rad(100))
                
        constrained_roll = np.clip(constrained_roll,
                                   -np.abs(roll_sp_rad),
                                      np.abs(roll_sp_rad))

        body_rate_sp_rad = None
        if not inverted_roll:
            # calculate desired yaw rate from coordinated turn constraint
            # no side slip
            euler_rate_sp_rad = np.tan(constrained_roll) * np.cos(pitch) \
                * 9.81 / airspeed
                
            #transform setpoint to body angular rates (jacobian)
            yaw_body_rate_sp_rad = -np.sin(roll) * euler_pitch_rate_sp_rad \
               + np.cos(roll) * np.sin(pitch) * euler_rate_sp_rad
            
            body_rate_sp_rad = np.clip(yaw_body_rate_sp_rad,
                                        -self.max_yaw_rate,
                                        self.max_yaw_rate)
            
        if not body_rate_sp_rad:
            body_rate_sp_rad = 0.0
            
        return body_rate_sp_rad
    
    
    def px4_position_controller(self) -> None:
        """
        https://docs.px4.io/main/en/flight_stack/controller_diagrams.html
        """
        
    def px4_attitude_controller(self, roll_sp_rad:float,
                                pitch_sp_rad:float,
                                yaw_sp_rad:float,
                                thrust_sp:float) -> np.ndarray:
        """
        Returns the torque commands for the aircraft based on 
        the PX4 controller the cascaded controller

        """
        
        # this is the the end of the first part of the cascade controller
        roll_body_rate_sp_rad = self.control_roll(roll_sp_rad, yaw_sp_rad)
        pitch_body_rate_sp_rad = self.control_pitch(pitch_sp_rad, yaw_sp_rad)
        yaw_body_rate_sp_rad = self.control_yaw(roll_sp_rad, pitch_sp_rad)
        print("roll body rate: ", np.rad2deg(roll_body_rate_sp_rad))
        print("pitch body rate: ", np.rad2deg(pitch_body_rate_sp_rad))
        print("yaw body rate: ", np.rad2deg(yaw_body_rate_sp_rad))
        
        #get the current angular rates of the controller
        current_roll_rate = self.sim[prp.p_radps]
        current_pitch_rate = self.sim[prp.q_radps]
        current_yaw_rate = self.sim[prp.r_radps]
        print("current roll rate: ", np.rad2deg(current_roll_rate))
        print("current pitch rate: ", np.rad2deg(current_pitch_rate))
        print("current yaw rate: ", np.rad2deg(current_yaw_rate))
        
        #get the angular acceleration of the controller
        current_roll_accel = self.sim[prp.pdot_radps2]
        current_pitch_accel = self.sim[prp.qdot_radps2]
        current_yaw_accel = self.sim[prp.rdot_radps2]
        
        #get trim conditions 
        trim_r = self.trim_roll_rad
        trim_p = self.trim_pitch_rad
        trim_y = self.trim_yaw_rad
        
        airspeed = feet_to_meters(self.sim[prp.airspeed])
        
        #check if control rates are saturated
        #bi-linear interpolation for airspeed for actuator saturation
        if airspeed < self.trim_airspeed:
            trim_r += interpolate(airspeed, self.min_airspeed, 
                                self.trim_airspeed, 0.0, 0.0)
            trim_p += interpolate(airspeed, self.min_airspeed,
                                    self.trim_airspeed, 0.0, 0.0)
            trim_y += interpolate(airspeed, self.min_airspeed,
                                    self.trim_airspeed, 0.0, 0.0)
        else:
            trim_r += interpolate(airspeed, self.trim_airspeed,
                                    self.max_airspeed, 0.0, 0.0)
            trim_p += interpolate(airspeed, self.trim_airspeed,
                                    self.max_airspeed, 0.0, 0.0)
            trim_y += interpolate(airspeed, self.trim_airspeed,
                                    self.max_airspeed, 0.0, 0.0)
            
        # run controller now 
		# Run attitude RATE controllers which need the desired attitudes 
        # from above, add trim.
        angular_acc_roll_sp = self.roll_rate_control.update(
            current_roll_rate, roll_body_rate_sp_rad, 
            current_roll_accel, self.sim.sim_dt)
        
        angular_acc_pitch_sp = self.pitch_rate_control.update(
            current_pitch_rate, pitch_body_rate_sp_rad, 
            current_pitch_accel, self.sim.sim_dt)
        
        angular_acc_yaw_sp = self.yaw_rate_control.update(
            current_yaw_rate, yaw_body_rate_sp_rad, 
            current_yaw_accel, self.sim.sim_dt)
        
        # get feedforward terms
        ffg_rollrate = self.roll_rate_control.FFgain 
        ffg_pitchrate = self.pitch_rate_control.FFgain
        ffg_yawrate = self.yaw_rate_control.FFgain
        
        feedforward_rollrate = ffg_rollrate * roll_body_rate_sp_rad
        feedforward_pitchrate = ffg_pitchrate * pitch_body_rate_sp_rad
        feedforward_yawrate = ffg_yawrate * yaw_body_rate_sp_rad

        #airspeed scaling is 1*1 so not included
        u_roll_acc = angular_acc_roll_sp + feedforward_rollrate
        u_pitch_acc = angular_acc_pitch_sp + feedforward_pitchrate
        u_yaw_acc = angular_acc_yaw_sp + feedforward_yawrate
        
        #check if 
        torque_sp = np.array([u_roll_acc, u_pitch_acc, u_yaw_acc])
        if np.isfinite(u_roll_acc):
            u_roll_acc = np.clip(u_roll_acc+trim_r, 
                                 -1.0, 
                                 1.0)
            torque_sp[0] = u_roll_acc
        else:
            self.roll_rate_control.rate_i = 0.0
            torque_sp[0] = trim_r
            
        if np.isfinite(u_pitch_acc):
            u_pitch_acc = np.clip(u_pitch_acc+trim_p, 
                                  -1.0, 
                                  1.0)
            torque_sp[1] = u_pitch_acc
        else:
            self.pitch_rate_control.rate_i = 0.0
            torque_sp[1] = trim_p
            
        if np.isfinite(u_yaw_acc):
            u_yaw_acc = np.clip(u_yaw_acc+trim_y, 
                                -1.0, 
                                1.0)
            torque_sp[2] = u_yaw_acc
        else:
            self.yaw_rate_control.rate_i = 0.0
            torque_sp[2] = trim_y
        
        # add feed-forward from roll control output to yaw control
        # output this can be used to counteract the adverse yaw effect
        # when rolling the aircraft
        torque_sp[2] = np.clip(torque_sp[2] + (0.0 * torque_sp[0]), 
                               -1.0, 1.0)   
        
        self.sim[prp.aileron_cmd] = torque_sp[0]
        self.sim[prp.elevator_cmd] = 0.0
        # self.sim[prp.rudder_cmd] = torque_sp[2]
        
        
    def heading_hold(self, heading_comm:float) -> None:
        """
        Maintains a commanded heading [degrees] using a PI controller

        Command reference is the desired heading in degrees
        Where 0 degrees is North, 90 degrees is East, 180 degrees is South, 270 degrees is West

        :param heading_comm: commanded heading [degrees]
        :return: None
        """
        # Attempted Nichols-Ziegler with Pcr = 0.048, Kcr=1.74, lead to a lot of overshoot
        # error = heading_comm_dg - self.sim[prp.heading_deg]
        # print("heading command: ", heading_comm)
        error = heading_comm - np.rad2deg(self.sim[prp.heading_rad])
        # if abs(error) > 2:
        #     self.roll_hold(np.deg2rad(0))
        #     return        
        # error = heading_comm - self.sim[prp.heading_rad]
        #print("attitude/psi-true-deg", self.sim.get_property_value('attitude/heading-true-deg'))
        #error = heading_comm_dg - self.sim.get_property_value('attitude/psi-true-deg')
        # Ensure the aircraft always tu rns the shortest way round
        
        #wrap error between 
        
        if error < -180:
            error = error + 360
        if error > 180:
            error = error - 360
        # if error < -math.pi:
        #     error = error + 2*math.pi
        # if error > math.pi:
        #     error = error - 2*math.pi
        # print(error)
        # kp = -2.0023 * 0.005
        # ki = -0.6382 * 0.005
        kp = -0.6
        ki = -0.1
        heading_controller = PID(kp, ki, 0.0)
        output = heading_controller(error)
        # Prevent over-bank +/- 30 radians
        if output < - 30 * (math.pi / 180):
            output = - 30 * (math.pi / 180)
        if output > 30 * (math.pi / 180):
            output = 30 * (math.pi / 180)

        # if output < -np.deg2rad(30):
        #     output = -np.deg2rad(30)
        # if output > np.deg2rad(30):
        #     output = np.deg2rad(30)

        self.roll_hold(output)

    def airspeed_hold_w_throttle(self, airspeed_comm: float) -> None:
        """
        Maintains a commanded airspeed [KTAS] using throttle_cmd and a PI controller

        :param airspeed_comm: commanded airspeed [KTAS]
        :return: None
        """
        # Appears fine with simple proportional controller, light airspeed instability at high speed (100kts)
        error = airspeed_comm - (self.sim[prp.airspeed] * 0.5925)  # set airspeed in KTAS'
        kp = 1.0
        ki = 0.035
        airspeed_controller = PID(kp, ki, 0.0)
        output = airspeed_controller(-error)
        # Clip throttle command from 0 to +1 can't be allowed to exceed this!
        if output > 1:
            output = 1
        if output < 0:
            output = 0
        self.sim[prp.throttle_cmd] = output

    def altitude_hold(self, altitude_comm) -> None:
        """
        Maintains a demanded altitude [feet] using pitch attitude

        :param altitude_comm: demanded altitude [feet]
        :return: None
        """
        #print('altitude command: ', altitude_comm)
        error = altitude_comm - self.sim[prp.altitude_sl_ft]
        #error = altitude_comm - self.sim.get_property_value("position/h-sl-ft")
        # print('error: ', error)
        kp = 0.3
        kd = 0.1
        # kp = 0.3
        ki = 0.1
        altitude_controller = PID(kp, ki, kd)
        output = altitude_controller(-error)
        # prevent excessive pitch +/- 15 degrees
        if output < - 15 * (math.pi / 180):
            output = - 15 * (math.pi / 180)
        if output > 15 * (math.pi / 180):
            output = 15 * (math.pi / 180)
        self.pitch_hold(output)

    def home_to_target(self, target_northing: float, target_easting: float, target_alt: float) -> bool:
        """
        Homes towards a 2D (lat, long) point in space and uses altitude_hold to maintain an altitude

        :param target_northing: latitude of target relative to current position [m]
        :param target_easting: longitude of target relative to current position [m]
        :param target_alt: demanded altitude for this path segment [feet]
        :return: flag==True if the simulation has reached a target in space
        """
        if self.nav is None:
            # initialize targeting/navigation object
            self.nav = LocalNavigation(self.sim)
            self.nav.set_local_target(target_northing, target_easting)
            self.flag = False
        if self.nav is not None:
            if not self.flag:
                # fly to target using bearing calculated from current position
                bearing = self.nav.bearing() * 180.0 / math.pi
                if bearing < 0:
                    bearing = bearing + 360
                distance = self.nav.distance()
                # when within 100m radius return flag and stop a/p functionality
                if distance < 100:
                    self.flag = True
                    self.nav = None
                    return self.flag
                self.heading_hold(bearing)
                self.altitude_hold(target_alt)

    def track_to_target(self, target_northing: float, target_easting: float, target_alt: float) -> bool:
        """
        Maintains a track from the point the simulation started at to the target point

        ...

        This ensures the aircraft does not fly a curved homing path if displaced from track but instead aims to
        re-establish the track to the pre-defined target point in space. The method terminates when the aircraft arrives
        at a point within 200m of the target point.
        :param target_northing: latitude of target relative to current position [m]
        :param target_easting: longitude of target relative to current position [m]
        :param target_alt: demanded altitude for this path segment [feet]
        :return: flag==True if the simulation has reached the target
        """
        if self.nav is None:
            # initialize target and track
            self.nav = LocalNavigation(self.sim)
            self.nav.set_local_target(target_northing, target_easting)
            self.track_bearing = self.nav.bearing() * 180.0 / math.pi
            if self.track_bearing < 0:
                self.track_bearing = self.track_bearing + 360.0
            self.track_distance = self.nav.distance()
            self.flag = False
        if self.nav is not None:
            # position relative to target
            bearing = self.nav.bearing() * 180.0 / math.pi
            if bearing < 0:
                bearing = bearing + 360
            distance = self.nav.distance()
            off_tk_angle = self.track_bearing - bearing
            distance_to_go = self.nav.distance_to_go(distance, off_tk_angle)
            # use a P controller to regulate the closure rate relative to the track
            error = off_tk_angle * distance_to_go
            kp = 0.01
            ki = 0.0
            kd = 0.0
            closure_controller = PID(kp, ki, kd)
            heading = closure_controller(-error) + bearing
            if distance < 200:
                self.flag = True
                self.nav = None
                return self.flag
            self.heading_hold(heading)
            self.altitude_hold(target_alt)

    def track_to_profile(self, profile: list) -> bool:
        """
        Maintains a track along a series of points in the simulation and the defined altitude along each path segment

        ...

        This ensures the aircraft does not fly a curved homing path if displaced from track but instead aims to
        re-establish the track to the pre-defined target point in space. The method switches to the next target point
        when the aircraft arrives at a point within 300m of the current target point. The method terminates when the
        final point(:tuple) in the profile(:list) is reached.
        :param profile: series of points used to define a path formatted with a tuple at each index of the list
            [latitude, longitude, altitude]
        :return: flag==True if the simulation has reached the final target
        """
        if self.nav is None:
            self.track_id = self.track_id + 1
            if self.track_id == len(profile) - 1:
                print('hit flag')
                self.flag = True
                return self.flag
            point_a = profile[self.track_id]
            point_b = profile[self.track_id + 1]
            # initialize target and track
            self.nav = LocalNavigation(self.sim)
            self.nav.set_local_target(point_b[0] - point_a[0], point_b[1] - point_a[1])
            self.track_bearing = self.nav.bearing() * 180.0 / math.pi
            if self.track_bearing < 0:
                self.track_bearing = self.track_bearing + 360.0
            self.track_distance = self.nav.distance()
            self.flag = False
        if self.nav is not None:
            bearing = self.nav.bearing() * 180.0 / math.pi
            if bearing < 0:
                bearing = bearing + 360
            distance = self.nav.distance()
            off_tk_angle = bearing - self.track_bearing
            if off_tk_angle > 180:
                off_tk_angle = off_tk_angle - 360.0
            # scale response with distance from target
            distance_to_go = self.nav.distance_to_go(distance, off_tk_angle)
            if distance_to_go > 3000:
                distance_to_go = 3000
            heading = (8 * 0.00033 * distance_to_go * off_tk_angle) + self.track_bearing

            # radius = (self.sim[prp.airspeed] * 0.5925 / (20.0 * math.pi)) * 1852  # rate 1 radius
            radius = 300
            self.heading_hold(heading)
            self.altitude_hold(self.track_id[2])
            if distance < radius:
                self.nav = None

    def arc_path(self, profile, radius) -> bool:
        """
        Maintains a track along a series of points in the simulation and the defined altitude along each path segment,
        ensures a smooth turn by flying an arc/filet with radius=radius at each point

        ...

        The aircraft maintains a track based on the bearing from the location the target was instantiated to the target.
        When within a radius or crossing a plane perpendicular to track the aircraft goes from straight track mode to
        flying a curved path of radius r around a point perpendicular to the point created by the confluence of the two
        tracks. The method switches to the next point when it reaches a point equidistant from the beginning of the arc.
        The method terminates when there is no further 'b' or 'c' points available i.e. 2 points before the final track.
        The method is based off the algorithm defined in Beard & McLain chapter 11, path planning:
        http://uavbook.byu.edu/doku.php
        :param profile: series of points used to define a path formatted with a tuple at each index of the list
            [latitude, longitude, altitude]
        :param radius: fillet radial distance [m], used to calculate z point
        :return: flag==True if the simulation has reached the termination condition
        """
        if self.nav is None:
            # print(self.state)
            print('Changing points !!!')
            self.track_id = self.track_id + 1
            print(self.track_id)
            if self.track_id == len(profile) - 2:
                print('hit flag')
                self.flag = True
                return self.flag
            point_a = profile[self.track_id]
            point_b = profile[self.track_id + 1]
            point_c = profile[self.track_id + 2]
            self.nav = LocalNavigation(self.sim)
            # Initialize track outbound from b
            self.nav.set_local_target(point_c[0] - point_b[0], point_c[1] - point_b[1])
            self.track_bearing_out = self.nav.bearing() * 180.0 / math.pi
            if self.track_bearing_out < 0:
                self.track_bearing_out = self.track_bearing_out + 360.0
            # Initialize track inbound to b
            self.nav.local_target_set = False
            self.nav.set_local_target(point_b[0] - point_a[0], point_b[1] - point_a[1])
            self.track_bearing_in = self.nav.bearing() * 180.0 / math.pi
            if self.track_bearing_in < 0:
                self.track_bearing_in = self.track_bearing_in + 360.0

            # # Track to an absolute point in space
            # self.nav.local_target_set = False
            # self.nav.set_local_target(point_b[0], point_b[1])
            # self.track_distance = self.nav.distance()
            self.flag = False
        if self.nav is not None:
            # Define angle of filet from out and in plane
            filet_angle = self.track_bearing_out - self.track_bearing_in
            if filet_angle < 0:
                filet_angle = filet_angle + 360
            if self.state == 0:
                # Calculate h_plane to transition from straight line state to curved filet state
                q = self.nav.unit_dir_vector(profile[self.track_id], profile[self.track_id + 1])
                w = profile[self.track_id + 1]
                try:
                    z_point = (w[0] - ((radius / math.tan(filet_angle / 2 * (math.pi / 180.0))) * q[0]),
                               w[1] - ((radius / math.tan(filet_angle / 2 * (math.pi / 180.0))) * q[1]))
                    cur = self.nav.get_local_pos()
                    h_point = (cur[0] - z_point[0], cur[1] - z_point[1])
                    h_val = (h_point[0] * q[0]) + (h_point[1] * q[1])
                    if h_val > 0:
                        # Entered h plane transition to curved segment
                        self.state = 1

                    # Track straight line segment
                    bearing = self.nav.bearing() * 180.0 / math.pi
                    if bearing < 0:
                        bearing = bearing + 360
                    distance = self.nav.distance()
                    off_tk_angle = bearing - self.track_bearing_in
                    if off_tk_angle > 180:
                        off_tk_angle = off_tk_angle - 360.0
                    # scale response with distance from target
                    distance_to_go = self.nav.distance_to_go(distance, off_tk_angle)
                    if distance_to_go > 3000:
                        distance_to_go = 3000
                    heading = (8 * 0.00033 * distance_to_go * off_tk_angle) + self.track_bearing_in
                except ZeroDivisionError:
                    heading = self.track_bearing_in  # TODO: find a way to deal with straight lines
                    print("You have straight lines don't do this!")
                self.heading_hold(heading)
                self.altitude_hold(altitude_comm=w[2])

            if self.state == 1:
                # filet location
                q0 = self.nav.unit_dir_vector(profile[self.track_id], profile[self.track_id + 1])
                q1 = self.nav.unit_dir_vector(profile[self.track_id + 1], profile[self.track_id + 2])
                q_grad = self.nav.unit_dir_vector(q1, q0)
                w = profile[self.track_id + 1]
                # center point of arc (mirrored from radius and apex of turn)
                center_point = (w[0] - ((radius / math.sin(filet_angle / 2 * (math.pi / 180.0))) * q_grad[0]),
                                w[1] - ((radius / math.sin(filet_angle / 2 * (math.pi / 180.0))) * q_grad[1]))
                z_point = (w[0] + ((radius / math.tan(filet_angle / 2 * (math.pi / 180.0))) * q1[0]),
                           w[1] + ((radius / math.tan(filet_angle / 2 * (math.pi / 180.0))) * q1[1]))
                turning_direction = math.copysign(1, (q0[0] * q1[1]) - (q0[1] * q1[0]))
                cur = self.nav.get_local_pos()
                h_point = (cur[0] - z_point[0], cur[1] - z_point[1])
                h_val = (h_point[0] * q1[0]) + (h_point[1] * q1[1])
                if h_val > 0:
                    self.nav = None
                    self.state = 0
                    return

                # Control circular (orbit) motion
                distance_from_center = math.sqrt(math.pow(cur[0] - center_point[0], 2) +
                                                 math.pow(cur[1] - center_point[1], 2))
                circ_x = cur[1] - center_point[1]
                circ_y = cur[0] - center_point[0]
                circle_angle = math.atan2(circ_x, circ_y)
                if circle_angle < 0:
                    circle_angle = circle_angle + (2 * math.pi)
                tangent_track = circle_angle + (turning_direction * (math.pi / 2))
                if tangent_track < 0:
                    tangent_track = tangent_track + (2 * math.pi)
                if tangent_track > 2 * math.pi:
                    tangent_track = tangent_track - (2 * math.pi)
                tangent_track = tangent_track * (180.0 / math.pi)
                error = (distance_from_center - radius) / radius
                k_orbit = 4.0
                heading = tangent_track + (math.atan(k_orbit * error) * (180.0 / math.pi))
                self.heading_hold(heading)
                self.altitude_hold(altitude_comm=w[2])


# class X8StateSpace:
#     """
#     State-Space based control system for the X8 UAV
#
#     ...
#
#     Attributes:
#     -----------
#     sim : Simulation object
#         an instance of the flight simulation flight dynamic model, used to interface with JSBSim
#
#     Methods:
#     -------
#     system_id(self)
#         identifies the FDM's non-linear state-space model
#     """
#
#     def __init__(self, sim, dt):
#         self.sim = sim
#         self.dt = dt
#
#     def system_id(self) -> control.iosys.InputOutputSystem:
#         """
#         Forms a state-space representation of the aircraft for use with controllers and analyzing
#         the models response.
#
#         :return state: the aircraft state-space class
#         """
#
#         # inputs to state-space system
#         control_in[0] = self.sim[prp.aileron_combined_rad]  # ail
#         control_in[1] = self.sim[prp.elevator]  # elev
#         control_in[2] = self.sim[prp.throttle]  # rudd
#
#         # states of state-space system
#         state[0] = self.sim[prp.lat_travel_m]  # x
#         state[1] = self.sim[prp.lng_travel_m]  # y
#         state[2] = self.sim[prp.altitude_sl_ft] / 3.28  # z
#
#         state[3] = self.sim[prp.roll_rad]
#         state[4] = self.sim[prp.pitch_rad]
#         state[5] = self.sim[prp.heading_rad]
#
#         state[6] = self.sim[prp.u_fps] / 3.28  # u
#         state[7] = self.sim[prp.v_fps] / 3.28  # v
#         state[8] = self.sim[prp.w_fps] / 3.28  # w
#
#         state[9] = self.sim[prp.p_radps]  # p
#         state[10] = self.sim[prp.q_radps]  # q
#         state[11] = self.sim[prp.r_radps]  # r
#
#
#         return control.iosys.InputOutputSystem(inputs=control_in,
#                                                outputs=None,
#                                                states=state,
#                                                dt=self.dt,
#                                                name='x8StateSpace')



















