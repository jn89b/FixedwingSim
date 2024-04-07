"""
https://python.plainenglish.io/reference-frame-transformations-in-python-with-numpy-and-matplotlib-6adeb901e0b0

roll,pitch,yaw rotation is inertial frame to body frame

body frame to inertial frame is yaw,pitch,roll rotation

"""

import numpy as np
import math as m
import casadi as ca
import opt_control.rotation_utils as rot
#import rotation_utils as rot
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Effector():
    """
    Currently 2D only 
    """
    def __init__(self, effector_config:dict, use_casadi=False):
        
        self.effector_config = effector_config
        self.effector_range = effector_config['effector_range'] #meters
        self.effector_power = effector_config['effector_power'] #watts
    
        if effector_config['effector_type'] == 'directional':
            self.effector_type = 'directional'
            self.effector_angle = effector_config['effector_angle']
            self.effector_profile = create_triangle(self.effector_angle, self.effector_range)

        elif effector_config['effector_type'] == 'omnidirectional':
            self.effector_type = 'omnidirectional'
            self.effector_angle = 2*np.pi

        elif effector_config['effector_type'] == 'directional_3d':
            self.effector_type = 'directional_3d'
            self.effector_angle = effector_config['effector_angle']
            self.effector_profile = create_pyramid(self.effector_angle, self.effector_range)    
        
        else:
            raise Exception("Effector type not recognized")


        if use_casadi == True:
            # self.effector_profile = ca.DM(self.effector_profile)
            self.effector_power = ca.DM(self.effector_power)
            self.effector_range = ca.DM(self.effector_range)
            self.effector_angle = ca.DM(self.effector_angle)
            self.use_casadi = True
        else:
            self.use_casadi = False
        
        self.effector_location = None


    def define_effector_states(self):
        """
        Define the states of the effector
        """
    

    def set_effector_location(self, ref_point, ref_angle):
        """
        Set the location of the effector relative to a reference point
        and a reference angle
        """
        if self.use_casadi == True:
            self.effector_location = (rot.rot2d_casadi(ref_angle) @ self.effector_profile) + ref_point
            self.effector_location = ca.horzcat(self.effector_location, ref_point)
        else:
            self.effector_location = (rot.rot2d(ref_angle) @ self.effector_profile.T).T + ref_point
            #add the reference point to the effector location
            self.effector_location = np.vstack((self.effector_location, ref_point))
            
    def set_effector_location3d(self, ref_point, ref_roll, ref_pitch, ref_yaw):
        """
        Set the location of the effector relative to a reference point
        based on reference roll, pitch, and yaw
        """
        if self.use_casadi == True:
            self.effector_location = (rot.rot3d_casadi(ref_roll,ref_pitch, ref_yaw) @ \
                self.effector_profile.T).T #+ ref_point.T
            self.effector_location[0,:] + ref_point
            self.effector_location[1,:] + ref_point
            
            self.effector_location = ca.vertcat(self.effector_location, ref_point)
            # print("effector final shape is", self.effector_location.shape)
        else:
            self.effector_location = (rot.rot3d(ref_roll,ref_pitch, ref_yaw) @ \
                np.transpose(self.effector_profile)).T + ref_point
            #add the reference point to the effector location
            self.effector_location = np.vstack((self.effector_location, ref_point))

    def compute_power_density(self, target_distance, effector_factor, use_casadi=False):
        """
        Compute the power density at a given distance from the effector
        """
        
        #check if using casadi 
        if self.use_casadi == True:
            if self.effector_type == 'directional' or self.effector_type == 'directional_3d':
                return effector_factor * self.effector_power / (target_distance**2 * 4*ca.pi)
            
            elif self.effector_type == 'omnidirectional':
                return self.effector_power / (target_distance**2 * 4*ca.pi)
            
            else:
                raise Exception("Effector type not recognized")

        #if not using casadi
        else:
            if self.effector_type == 'directional' or self.effector_type == 'directional_3d':
                return self.effector_power / (target_distance**2 * 4*np.pi)

            elif self.effector_type == 'omnidirectional':
                return self.effector_power / (target_distance**2 * 4*np.pi)

            else:
                raise Exception("Effector type not recognized")
    

    def is_inside_effector(self, target:np.array):
        """check if target is inside the effector"""
        #catch if effector location has not been set
        if self.effector_location is None:
            raise Exception("Effector location has not been set")
            
        point = Point(target)
        polygon = Polygon(self.effector_location)

        if polygon.contains(point):
            return True
        else:
            return False


def create_triangle_casadi(angle, distance):
    #this is double the angle 
    angle = angle/2
    p_x = distance * ca.cos(angle)
    p_y = distance * ca.sin(angle)
    p_1 = ca.vertcat(p_x, p_y)

    p_y = -distance * ca.sin(angle)
    p_2 = ca.vertcat(p_x, p_y) #the other point on the the triangle

    #compute the midpoint between p_1 and p_2
    # p_3 = (p_1 + p_2) / 2

    #return as an array of points
    return ca.vertcat(p_1, p_2)


#create 2d triangle with a given angle and distance
def create_triangle(angle, distance):
    angle = angle/2
    p_x = distance * np.cos(angle)
    p_y = distance * np.sin(angle)
    p_1 = np.array([p_x, p_y])

    p_y = -distance * np.sin(angle)
    p_2 = np.array([p_x, p_y]) #the other point on the the triangle

    #compute the midpoint between p_1 and p_2
    # p_3 = (p_1 + p_2) / 2

    #return as an array of points
    return np.array([p_1, p_2])

def create_pyramid(angle, distance):
    angle = angle/2
    p_x = distance * np.cos(angle)
    p_y = distance * np.sin(angle)
    p_1 = np.array([p_x, p_y, 0])

    p_y = -distance * np.sin(angle)
    p_2 = np.array([p_x, p_y, 0]) #the other point on the the triangle

    #return as an array of points
    return np.array([p_1, p_2])

def create_semicircle(mid_point, current_psi, diameter, num_points=100):
    r = diameter / 2
    print("mid point: ", mid_point)
    #create a half circle
    t = np.linspace(current_psi - np.pi/2, current_psi + np.pi/2, num_points)
    x = mid_point[0] + r * np.cos(t)
    y = mid_point[1] + r * np.sin(t)
    #return as a list of points
    return np.array([x, y])

# current_psi = np.deg2rad(15)
# effector_angle = np.deg2rad(30)
# effector_range = 1
# ref_point = np.array([4.30877077348727, 0.5046069115552776])

# effector_config = {
#         'effector_range': 1, 
#         'effector_power': 1, 
#         'effector_type': 'directional', 
#         'effector_angle': effector_angle
#         }

# effector = Effector(effector_config)
# effector.set_effector_location(ref_point, current_psi)

# #show effector points
# print("effector points: ", effector.effector_location)

# #check inside polygon 
# target = np.array([0.5, 0.5])
# print("is inside effector: ", effector.is_inside_effector(target))

# #plot the effector points
# import matplotlib.pyplot as plt
# #set equal aspect ratio
# plt.gca().set_aspect('equal', adjustable='box')
# plt.plot(effector.effector_location[:,0], effector.effector_location[:,1], 'o')
# plt.plot(ref_point[0], ref_point[1], 'o', color='red')
# plt.plot(target[0], target[1], 'o')
# plt.show()


