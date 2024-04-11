"""
This file is used to simulate multiple drones in the same environment.
One will be the pursuer and the other will be the evader map with the
same capabilities as the pursuer
"""

from conversions import meters_to_feet, feet_to_meters
import math 

def local_to_global_position(local_position: list) -> list:
    """Converts local position (in meters) to 
    global position (degrees and meters for altitude).
    """
    
    # Conversion factors
    meters_per_degree_latitude = 111320
    equatorial_circumference_meters = 40075000
    
    x_meters, y_meters, z_meters = local_position
    lat_degrees = y_meters / meters_per_degree_latitude
    lon_degrees = (x_meters / (equatorial_circumference_meters * math.cos(math.radians(lat_degrees)) / 360))
    
    return [lon_degrees, lat_degrees, z_meters]

def get_local_position(geo_coords: list, return_metric: bool = True) -> list:
    """Converts global position (degrees and meters for altitude) 
    to local position (in meters)."""
    
    lon_degrees, lat_degrees, alt_meters = geo_coords
    lat_meters = lat_degrees * 111320
    lon_meters = (lon_degrees * (40075000 * math.cos(math.radians(lat_degrees)) / 360))
    
    # Directly return in meters to avoid unnecessary conversion if return_metric is True
    return [lon_meters, lat_meters, alt_meters]

if __name__ == "__main__":
    
    test_position = [35, 4, 10]
    test_geo = local_to_global_position(test_position)
    print("geo: ", test_geo)
    
    test_position_2 = [25.3, 5, 50]
    test_geo_2 = local_to_global_position(test_position_2)
    print("geo: ", test_geo_2)
    
    converted_position = get_local_position(test_geo)
    converted_position_2 = get_local_position(test_geo_2)
    
    print("converted position: ", converted_position)
    print("converted position 2: ", converted_position_2)
    
    