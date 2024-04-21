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
    -[x] Once that works have it parameterized where you can send MPC trajectory commands
- Parallelize training:
    https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb#scrollTo=GlcJPYN-6ebp
- Verify simulation accuracy:
    -[] Simulate JSB X8 skywalker and send MPC commands to a goal location 
        - Log the data of the trajectories 
    -[] Simulate Ardupilot X8 skywalker and 
    

# Notes for refactorign
- need to reinitialize the sim for everything f