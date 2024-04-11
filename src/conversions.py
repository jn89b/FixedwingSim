import math 

def feet_to_meters(feet:float) -> float:
    return feet * 0.3048

def meters_to_feet(meters:float) -> float:
    return meters / 0.3048

def knots_to_mps(knots:float) -> float:
    return knots * 0.514444

def mps_to_knots(mps:float) -> float:
    return mps / 0.514444

def mps_to_ktas(mps:float) -> float:
    return mps * 1.94384

def ktas_to_mps(ktas:float) -> float:
    return ktas / 1.94384

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


