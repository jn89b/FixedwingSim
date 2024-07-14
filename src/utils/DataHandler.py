import numpy as np


class DataHandler():
    def __init__(self) -> None:
        self.x = []
        self.y = []
        self.z = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.u = []
        self.time_history = []
        
    def update_data(self,info_array:np.ndarray):
        self.x.append(info_array[0])
        self.y.append(info_array[1])
        self.z.append(info_array[2])
        self.roll.append(info_array[3])
        self.pitch.append(info_array[4])
        self.yaw.append(info_array[5])
        self.u.append(info_array[6])

    def update_time(self, time:float) -> None:
        self.time_history.append(time)