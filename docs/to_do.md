# To Do
- Have an abstract class that does the simulation 


## Goals
- Build two environments:
    - DRL with high level attitude commands 
    - DRL with position commands fed to MPC  

- Train agent to go to target location with given payload:
    - DRL high level attitude commands
    - DRL with position commands fed to MPC
    - DRL with possible trajectories given to MPC  

- From there inherit the environments and add the following:
    - Static obstacles and dynamic threat configurations 

## How to do it
- Build the Open Gym Environment
    -[x] Test if the environment works
    -[] Once working train agent to do high level attitude commands to get a goal location 
    -[] Once that works have it parameterized where you can send MPC trajectory commands
- Parallelize training:
    https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb#scrollTo=GlcJPYN-6ebp
    