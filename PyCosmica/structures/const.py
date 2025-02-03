import numpy as np

# Constants
PI = 3.141592653589793  # Pi
HALF_PI = PI / 2
AU_KM = 149597870.691  # 1 AU in km
AU_M = 149597870691.0  # 1 AU in m
AU_CM = 14959787069100.0  # 1 AU in cm
MeV = 1e6  # MeV -> eV
GeV = 1e9  # GeV -> eV
C = 3e8  # Light Velocity in m/s
theta_north_limit = 0.000010  # Max latitude at north pole
theta_south_limit = PI - theta_north_limit  # Max latitude at south pole
R_mirror = 0.3  # [AU] Internal heliosphere bounduary - mirror radius

# Verbose Levels
VERBOSE_LOW = 1
VERBOSE_MED = 2
VERBOSE_HIGH = 3

# Magnetic field disturbance and threshold
polar_zone = 30
cos_polar_zone = np.cos(polar_zone * PI / 180.0)
delta_m = 2.000000e-05
tilt_L_max_activity_threshold = 50

# Solar Parameters
omega = 3.03008e-6  # Solar angular velocity
R_helio = 0.004633333  # Solar radius in AU

# Maximum number of simulation regions
n_max_regions = 335

del np
