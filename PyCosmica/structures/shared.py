from typing import NamedTuple


class HeliosphereBoundRadius(NamedTuple):
    R_ts_nose: float  # Termination shock position
    R_ts_tail: float  # Termination shock position (tail)
    R_hp_nose: float  # Heliopause position
    R_hp_tail: float  # Heliopause position (tail)


class Position3D(NamedTuple):
    r: float  # heliocentric radial component
    th: float  # heliocentric polar component
    phi: float  # heliocentric azimutal - longitudinal angle component

    @classmethod
    def from_list(cls, arr: list[float]):
        return cls(*arr)


class ParticleDescription(NamedTuple):
    T0: float  # Rest mass in GeV/n
    Z: float  # Atomic number
    A: float  # Mass number


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
