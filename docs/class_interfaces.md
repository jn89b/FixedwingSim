# Abstract Simulation Class
- Requires:
    - init_conditions:dict
    - aircraft_parameters:xml (for now set to default skywalker)
    - low_level_flight_controller_params: dict for kp,ki,kd
    - utilize_mpc:bool defaults to False 
    - mpc_parameters:dict parameters to feed for MPC


# Classes that will inherit
## OpenGym Environment
## ROS 2
## Unity/Unreal (Hold this off)