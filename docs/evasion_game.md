# Evasion Process
- Action space: x,y,z commands
- Observation space: distance relative to threat(s) as well as its position
- Reward function:
    - Goal is to avoid for N seconds or if pursuer crashes 
    - Don't be within range of pursuer for now (keep this simple)
        - +1 if outside range of pursuer
        - -1 if inside range of  pursuer
- Environment:
    - Simulate threat(s) with same capabilities of aircraft running simple PN for guidance fed to simple PID controller for now

- I want to compare this trained model in comparison to my objective function
- If this is better use this as part of hiearchial framework for decision making


# Proportional Navigation
- If im making huge turns based on v^2/r then lower the velocity or curvature
