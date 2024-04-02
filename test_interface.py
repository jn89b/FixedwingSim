import fixedwing_sim
import gymnasium as gym

from jsbim_backend.aircraft import Aircraft, x8
from src.sim_interface import OpenGymInterface
from src.conversions import meters_to_feet, mps_to_knots
 



## Need to define these parameters first before 
# running the test
init_state_dict = {
    "ic/u-fps": mps_to_knots(25),
    "ic/v-fps": 0.0,
    "ic/w-fps": 0.0,
    "ic/p-rad_sec": 0.0,
    "ic/q-rad_sec": 0.0,
    "ic/r-rad_sec": 0.0,
    "ic/h-sl-ft": meters_to_feet(50),
    "ic/long-gc-deg": 0.0,
    "ic/lat-gc-deg": 0.0,
    "ic/psi-true-deg": 15,
    "ic/theta-deg": 0.0,
    "ic/phi-deg": 0.0,
    "ic/alpha-deg": 0.0,
    "ic/beta-deg": 0.0,
    "ic/num_engines": 1,
}

aircraft = x8
gym_adapter = OpenGymInterface(init_conditions=init_state_dict,
                                 aircraft=aircraft,)



#print entry point
print(gym.envs.registry.keys())
env = gym.make('UAMEnv', backend_interface=gym_adapter)

