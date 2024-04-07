import jsbsim
#import airsim
import os
import time
from typing import Dict, Union
import jsbsim_backend.properties as prp
from jsbsim_backend.aircraft import Aircraft, cessna172P, x8
#from jsbsim_backend.aircraft import Aircraft, cessna172P, x8
from conversions import feet_to_meters, meters_to_feet, knots_to_mps, mps_to_knots
#from src.jsbsim_aircraft import Aircraft, cessna172P, x8
#from src.conversions import feet_to_meters, meters_to_feet, knots_to_mps, mps_to_knots
import math

"""Initially based upon https://github.com/Gor-Ren/gym-jsbsim/blob/master/gym_jsbsim/simulation.py by Gordon Rennie"""


class FlightDynamics:
    """
    The core JSBSim simulation class

    ...

    Attributes:
    ----------
    fdm : object
        an object that is an instance of the JSBSim's flight dynamic model
    sim_dt : var
        the simulation update rate
    aircraft : Aircraft
        the aircraft type used, cessna172P by default
    init_conditions : Dict[prp.Property, float] = None
        the simulations initial conditions None by default as in basic_ic.xml
    debug_level : int
        the level of debugging sent to the terminal by jsbsim
        - 0 is limited
        - 1 is core values
        - 2 gives all calls within the C++ source code
    wall_clock_dt : bool
        activates a switch to speed up or slow down the simulation
    client : object - REMOVED FROM FORK
        connection to airsim for visualization

    Methods:
    ------
    load_model(model_name)
        Ensure the JSBSim flight dynamic model is found and loaded in correctly
    get_aircraft()
        returns the aircraft the simulator was initialized with
    get_loaded_model_name()
        returns the name of the fdm model used
    initialise(dt: float, model_name: str, init_conditions: Dict['prp.Property', float] = None)
        initializes an instance of JSBSim
    set_custom_initial_conditions(init_conditions: Dict['prp.Property', float] = None)
        allows for initial conditions different to basic_ic.xml to be used
    reinitialise(self, init_conditions: Dict['prp.Property', float] = None)
        restart the simulation with default initial conditions
    run()
        run JSBSim at the sim_dt rate
    get_time()
        returns the current JSBSim time
    get_local_position()
        returns the lat, long and altitude of JSBSim
    get_local_orientation()
        returns the euler angle orientation (roll, pitch, yaw) of JSBSim
    airsim_connect() - REMOVED FROM FORK
        connect to a running instance of airsim
    update_airsim() - REMOVED FROM FORK
        updates the airsim client with the JSBSim calculated pose information
    close()
        closes the JSBSim fdm instance
    start_engines()
        starts all available aircraft engines
    set_throttle_mixture_controls()
        sets aircraft mixture and throttle controls
    raise_landing_gear()
        raises the aircraft's landing gear
    """

    encoding = 'utf-8'
    #ROOT_DIR = os.path.abspath(r"c:\Users\quessy\Dev\jsbsim")
    
    def __init__(self,
                 sim_frequency_hz: float = 60.0,
                 aircraft: Aircraft = x8,
                 init_conditions:Dict = None,
                 return_metric_units: bool = True,
                 debug_level: int = 0):
        #self.fdm = jsbsim.FGFDMExec(root_dir=self.ROOT_DIR)
        self.fdm = jsbsim.FGFDMExec(None) # will need to map this to root 
        self.fdm.set_debug_level(debug_level)
        self.sim_frequency_hz = sim_frequency_hz
        self.sim_dt = 1.0 / sim_frequency_hz
        self.aircraft = aircraft
        self.init_conditions = init_conditions
        self.return_metric_units = return_metric_units
        self.initialise(self.sim_dt, self.aircraft.jsbsim_id, self.init_conditions)
        self.fdm.disable_output()
        self.wall_clock_dt = None
        # self.client = self.airsim_connect()

    def __getitem__(self, prop: Union[prp.BoundedProperty, prp.Property]) -> float:
        return self.fdm[prop.name]

    def __setitem__(self, prop: Union[prp.BoundedProperty, prp.Property], value) -> None:
        self.fdm[prop.name] = value

    def load_model(self, model_name: str) -> None:
        """
        Load a JSBSim xml formatted aircraft model into the JSBSim flight dynamic model

        :param model_name: name of aircraft model loaded into JSBSim
        :return: None
        """
        load_success = self.fdm.load_model(model_name)

        if not load_success:
            raise RuntimeError('JSBSim could not find specified model name: ' + model_name)

    def get_aircraft(self) -> Aircraft:
        """
        Get the Aircraft the JSBSim was initialised with

        :return: aircraft used in the simulator
        """
        return self.aircraft

    def get_loaded_model_name(self) -> str:
        """
        Get the name of the loaded aircraft model from the current JSBSim FDM instance

        :return: JSBSim model name
        """
        name: str = self.fdm.get_model_name().decode(self.encoding)
        if name:
            return name
        else:
            return None

    def initialise(self, dt: float, model_name: str, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Start JSBSim with custom initial conditions

        :param dt: simulation rate [s]
        :param model_name: the aircraft model used
        :param init_conditions: initial simulation conditions
        :return: None
        """
        # if init_conditions is not None:
        #     ic_file = 'minimal_ic.xml'
        # else:
        #     ic_file = 'basic_ic.xml'

        # ic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ic_file)
        # self.fdm.load_ic(ic_path, useStoredPath=False)
        for k, v in init_conditions.items():
            self.fdm[k] = v
            
        self.load_model(model_name)
        self.fdm.set_dt(dt)
        # self.set_custom_initial_conditions(init_conditions)

        success = self.fdm.run_ic()
        if not success:
            raise RuntimeError('JSBSim failed to initialise simulation conditions.')
        # else:
        #     print('JSBSim successfully initialised')

    def set_custom_initial_conditions(self, init_conditions: dict = None) -> None:
        """
        Set initial conditions different to what is found in the <name-ic.xml> file used

        :param init_conditions: the initial simulation conditions, defined based on prp JSBSim properties
        :return: None
        """
        # if init_conditions is not None:
        #     for prop, value in init_conditions.items():
        #         self[prop] = value
        if init_conditions is None:
            for k,v in self.init_conditions.items():
                self.fdm[k] = v

    def reinitialise(self, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Restart the simulator with initial conditions

        :param init_conditions: the initial simulation conditions, defined based on prp JSBSim properties,
        by default this is the original initialization file
        :return: None
        """
        self.set_custom_initial_conditions(init_conditions=init_conditions)
        no_output_reset_mode = 1
        self.fdm.reset_to_initial_conditions(no_output_reset_mode)
        # self.update_airsim()

    def run(self) -> bool:
        """
        Check if the FDM has terminated and if not advances one time step, slows by wall_clock_dt

        :return: True if FDM can advance
        """
        result = self.fdm.run()
        if self.wall_clock_dt is not None:
            time.sleep(self.wall_clock_dt)
        return result

    def get_time(self) -> float:
        """
        Get the current simulation time

        :return: the simulation time
        """
        sim_time = self[prp.sim_time_s]
        return sim_time

    def get_local_position(self) -> list:
        """
        Get the local absolute position from the simulation start point
        :return: position [lat, long, alt]
        """
        # lat = self[prp.lat_travel_m]
        # long = self[prp.lng_travel_m]
        lat = 111320 * self[prp.lat_geod_deg]
        lon = 40075000 * self[prp.lng_geoc_deg] * math.cos(self[prp.lat_geod_deg] * (math.pi / 180.0)) / 360
        alt = self[prp.altitude_sl_ft]
        
        if self.return_metric_units:
            lat = feet_to_meters(lat)
            lon = feet_to_meters(lon)
            alt = feet_to_meters(alt)
            
        position = [lon, lat, alt]
        return position

    def get_local_orientation(self) -> list:
        """
        Get the orientation of the vehicle

        :return: orientation [pitch, roll, yaw]
        # """
        roll = self[prp.roll_rad]
        pitch = self[prp.pitch_rad]
        # yaw = self[prp.heading_deg] * (math.pi / 180)
        yaw = self[prp.heading_rad]
        #wrap yaw to -pi to pi
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        # yaw = self.fdm.get_property_value("attitude/heading-true-rad")
        #self[prp.heading_rad]
        orientation = [roll, pitch, yaw]
        return orientation
    
    def get_states(self) -> dict:
        """
        Gets the current state of the aircraft
        """
        position = self.get_local_position()
        orientation = self.get_local_orientation()
        u_ms = feet_to_meters(self[prp.u_fps])
        v_ms = feet_to_meters(self[prp.v_fps])
        w_ms = feet_to_meters(self[prp.w_fps])
        mag_airspeed = math.sqrt(u_ms**2 + v_ms**2 + w_ms**2)
        
        states = {
            'x': position[0],
            'y': position[1],
            'z': position[2],
            'phi': orientation[0],
            'theta': orientation[1],
            'psi': orientation[2],
            'airspeed': mag_airspeed,
        }
        
        return states

    # @staticmethod
    # def airsim_connect() -> airsim.VehicleClient:
    #     """
    #     Connect to airsim client, exposing the CV mode UE4 graphic environment.

    #     :return: the airsim client object
    #     """
    #     client = airsim.VehicleClient()
    #     client.confirmConnection()
    #     return client

    # def update_airsim(self) -> None:
    #     """
    #     Update airsim with vehicle pose calculated by JSBSim

    #     :return: None
    #     """
    #     pose = self.client.simGetVehiclePose()
    #     position = self.get_local_position()
    #     pose.position.x_val = position[0]
    #     pose.position.y_val = position[1]
    #     pose.position.z_val = - position[2]
    #     euler_angles = self.get_local_orientation()
    #     pose.orientation = airsim.to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
    #     self.client.simSetVehiclePose(pose, False)  # boolean is whether to ignore collisions

    # def get_collision_info(self) -> airsim.VehicleClient.simGetCollisionInfo:
    #     """
    #     Gets collision info created by Airsim via the unreal engine

    #     Get the sim collision info object from Airsim with the following properties:
    #         - impact_point, where the aircraft collides with terrain
    #         - normal, the vector perpendicular to the point where the vehicle collided with terrain
    #         - position, the x, y, z position where the vehicle collided with the terrain
    #         - penetration_depth, how far through the terrain the collision has propagated
    #     The method: has_collided can also be called on the collision_info object to see whether or not a collision has
    #     occurred, returns true if it has penetrated the terrain
    #     :return: collision_info
    #     """
    #     collision_info = self.client.simGetCollisionInfo()
    #     return collision_info

    def close(self) -> None:
        """
        Close the JSBSim Flight Dynamic Model (FDM) currently running

        :return: None
        """
        if self.fdm:
            self.fdm = None

    def start_engines(self) -> None:
        """
        Start all available aircraft propulsion units

        :return: None
        """
        self[prp.all_engine_running] = -1

    def set_throttle_mixture_controls(self, throttle_cmd: float, mixture_cmd: float) -> None:
        """
        Set the throttle and mixture propulsion commands on an ICE powerplant, allows for a 2 engine aircraft too

        :param throttle_cmd: controls the throttle deflection (0 <-> 1)
        :param mixture_cmd: controls the mixture deflection (0 <-> 1)
        :return:
        """
        self[prp.throttle_cmd] = throttle_cmd
        # self[prp.mixture_cmd] = mixture_cmd

        try:
            self[prp.throttle_1_cmd] = throttle_cmd
            #self[prp.mixture_1_cmd] = mixture_cmd
        except KeyError:
            pass  # must be single-control aircraft

    def raise_landing_gear(self) -> None:
        """
        Raise the aircraft's landing gear

        :return: None
        """
        self[prp.gear] = 0.0
        self[prp.gear_all_cmd] = 0.0
