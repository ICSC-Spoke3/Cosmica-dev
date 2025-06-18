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


class RigidityFlux(NamedTuple):
    rigidity: RigidityVec
    flux: FluxVec
    isotope: Optional[Isotope] = None

    def to_energy(self, isotope: Optional[Isotope] = None) -> 'EnergyFlux':
        raise NotImplementedError


class EnergyFlux(NamedTuple):
    energy: EnergyVec
    flux: FluxVec
    isotope: Optional[Isotope] = None

    def to_rigidity(self, isotope: Optional[Isotope] = None) -> 'RigidityFlux':
        if isotope is None:
            if self.isotope is None:
                raise ValueError("Missing isotope")
            isotope = self.isotope
        return RigidityFlux(*func.en_to_rig_flux(self.energy, self.flux, isotope.A, isotope.Z))
