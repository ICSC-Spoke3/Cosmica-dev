from typing import NamedTuple, Optional

import numpy as np

from . import func
from .isotopes import Isotope


class NDArrayBase(np.ndarray):
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __array_finalize__(self, obj):
        if obj is None: return


class EnergyVec(NDArrayBase):
    def to_rigidity(self, isotope: Isotope) -> 'RigidityVec':
        return RigidityVec(func.en_to_rig(self, isotope.Z, isotope.A))


class RigidityVec(NDArrayBase):
    def to_energy(self, isotope: Isotope) -> 'EnergyVec':
        return EnergyVec(func.rig_to_en(self, isotope.Z, isotope.A))


class FluxVec(NDArrayBase):
    pass

class ErrorVec(NDArrayBase):
    pass

class RigidityFlux(NamedTuple):
    rigidity: RigidityVec
    flux: FluxVec

    def to_energy(self, isotope: Isotope) -> 'EnergyFlux':
        return EnergyFlux(*func.rig_to_en_flux(self.rigidity, self.flux, isotope.Z, isotope.A))


class EnergyFlux(NamedTuple):
    energy: EnergyVec
    flux: FluxVec

    def to_rigidity(self, isotope: Isotope) -> 'RigidityFlux':
        return RigidityFlux(*func.en_to_rig_flux(self.energy, self.flux, isotope.Z, isotope.A))
