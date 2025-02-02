from typing import NamedTuple
import jax.numpy as jnp

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
cos_polar_zone = jnp.cos(polar_zone * PI / 180.0)
delta_m = 2.000000e-05
tilt_L_max_activity_threshold = 50

# Solar Parameters
omega = 3.03008e-6  # Solar angular velocity
R_helio = 0.004633333  # Solar radius in AU

# Maximum number of simulation regions
n_max_regions = 335


# Struct Equivalents Using NamedTuple
class Options(NamedTuple):
    verbose: int
    input_file: str  # Placeholder for FILE*, as JAX does not support file objects


class ParticleDescription(NamedTuple):
    T0: float  # Rest mass in GeV/n
    Z: float  # Atomic number
    A: float  # Mass number


class HeliosphereBoundRadius(NamedTuple):
    R_ts_nose: float  # Termination shock position
    R_ts_tail: float  # Termination shock position (tail)
    R_hp_nose: float  # Heliopause position
    R_hp_tail: float  # Heliopause position (tail)


class HeliosphereProperties(NamedTuple):
    V0: float  # Radial solar wind speed [AU/s]
    K0_paral: tuple[float, float]  # Parallel diffusion parameter [high, low activity]
    K0_perp: tuple[float, float]  # Perpendicular diffusion parameter [high, low activity]
    gauss_var: tuple[float, float]  # Gaussian variation for diffusion parameter
    g_low: float  # Parameter for evaluating Kpar
    r_const: float  # Another parameter for evaluating Kpar
    tilt_angle: float  # Tilt angle of neutral sheet
    A_sun: float  # Normalization constant of HMF
    P0_d: float  # Drift suppression rigidity
    P0_dNS: float  # NS drift suppression rigidity
    plateau: float  # Time-dependent plateau for high-rigidity suppression
    polarity: int  # HMF polarity


class HeliosheatProperties(NamedTuple):
    V0: float  # Radial solar wind speed [AU/s]
    K0: float  # Parallel diffusion parameter


class SimulatedHeliosphere(NamedTuple):
    N_regions: int  # Number of inner heliosphere regions
    R_boundary_effe: list[HeliosphereBoundRadius]  # Boundaries in effective heliosphere
    R_boundary_real: list[HeliosphereBoundRadius]  # Real boundaries heliosphere
    is_high_activity_period: list[bool]  # Boolean array for high activity periods


class HeliosphericParameters(NamedTuple):
    K0: float
    ssn: float
    V0: float
    tilt_angle: float
    smooth_tilt: float
    B_earth: float
    polarity: int
    solar_phase: int
    NMCR: float
    heliosphere_bound_radius: HeliosphereBoundRadius

    @classmethod
    def from_list(cls, arr: list[float]):
        arr = arr[:6] + [int(arr[6]), int(arr[7]), arr[8], HeliosphereBoundRadius(*arr[9:])]
        return cls(*arr)


class HeliosheatParameters(NamedTuple):
    K0: float
    V0: float

    @classmethod
    def from_list(cls, arr: list[float]):
        return cls(*arr)


class Position3D(NamedTuple):
    r: float  # heliocentric radial component
    th: float  # heliocentric polar component
    phi: float  # heliocentric azimutal - longitudinal angle component

    @classmethod
    def from_list(cls, arr: list[float]):
        return cls(*arr)


class MonteCarloResult(NamedTuple):
    N_registered: int
    N_bins: int
    LogBin0_lowEdge: float  # lower boundary of first bin
    DeltaLogR: float  # Bin amplitude in log scale
    BoundaryDistribution: list[float]


class SimParameters(NamedTuple):
    output_file_name: str
    N_part: int  # Number of events to simulate
    N_T: int  # Number of energy bins
    N_initial_positions: int  # Number of initial positions (also Carrington rotations)
    T_centr: list[float]  # Energy array to be simulated
    initial_position: list[Position3D]  # Initial positions (assuming vect3D_t as array)
    ion_to_be_simulated: ParticleDescription  # Ion being simulated
    results: MonteCarloResult  # Placeholder for MonteCarloResult_t output
    relative_bin_amplitude: float  # Relative amplitude of energy bin
    heliosphere_to_be_simulated: SimulatedHeliosphere  # Heliosphere properties for the simulation
    prop_medium: list[HeliosphereProperties]  # Properties of the interplanetary medium
    prop_heliosheat: list[HeliosheatProperties]  # Properties of Heliosheat

    def to_jit(self):
        return SimParametersJit.from_sim(self)

class SimParametersJit(NamedTuple):
    N_part: int  # Number of events to simulate
    N_T: int  # Number of energy bins
    N_initial_positions: int  # Number of initial positions (also Carrington rotations)
    T_centr: list[float]  # Energy array to be simulated
    initial_position: list[Position3D]  # Initial positions (assuming vect3D_t as array)
    ion_to_be_simulated: ParticleDescription  # Ion being simulated
    results: MonteCarloResult  # Placeholder for MonteCarloResult_t output
    relative_bin_amplitude: float  # Relative amplitude of energy bin
    heliosphere_to_be_simulated: SimulatedHeliosphere  # Heliosphere properties for the simulation
    prop_medium: list[HeliosphereProperties]  # Properties of the interplanetary medium
    prop_heliosheat: list[HeliosheatProperties]  # Properties of Heliosheat

    @classmethod
    def from_sim(cls, sim: SimParameters):
        params = sim._asdict()
        params.pop('output_file_name')
        return cls(**params)

class QuasiParticle(NamedTuple):
    r: float
    th: float
    phi: float
    R: float
    t_fly: float