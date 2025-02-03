from typing import NamedTuple

from PyCosmica.structures.shared import HeliosphereBoundRadius


class Options(NamedTuple):
    verbose: int
    input_file: str  # Placeholder for FILE*, as JAX does not support file objects


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
