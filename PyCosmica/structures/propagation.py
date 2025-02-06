from typing import NamedTuple

from jax import tree_map, lax
from jax.typing import ArrayLike

from PyCosmica.structures.shared import Position3D, ParticleDescription, HeliosphereBoundRadius, HeliosphereProperties, \
    HeliosheatProperties


class MonteCarloResult(NamedTuple):
    N_registered: int
    N_bins: int
    LogBin0_lowEdge: float  # lower boundary of first bin
    DeltaLogR: float  # Bin amplitude in log scale
    BoundaryDistribution: list[float]


class SimulatedHeliosphere(NamedTuple):
    N_regions: int  # Number of inner heliosphere regions
    R_boundary_effe: list[HeliosphereBoundRadius]  # Boundaries in effective heliosphere
    R_boundary_real: list[HeliosphereBoundRadius]  # Real boundaries heliosphere
    is_high_activity_period: list[bool]  # Boolean array for high activity periods


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


class PropagationState(NamedTuple):
    r: ArrayLike
    th: ArrayLike
    phi: ArrayLike
    R: ArrayLike
    t_fly: ArrayLike
    rad_zone: ArrayLike
    init_zone: ArrayLike
    key: ArrayLike

    @property
    def _particle(self):
        return QuasiParticle(self.r, self.th, self.phi, self.R, self.t_fly)


class PropagationConstantsItem(NamedTuple):
    time_out: ArrayLike
    N_regions: ArrayLike  # Number of inner heliosphere regions
    max_dt: ArrayLike
    particle: ParticleDescription
    R_boundary_effe_init: HeliosphereBoundRadius  # Boundaries in effective heliosphere
    R_boundary_effe_rad: HeliosphereBoundRadius  # Boundaries in effective heliosphere
    R_boundary_real: HeliosphereBoundRadius  # Real boundaries heliosphere
    is_high_activity_period: ArrayLike
    LIM: HeliosphereProperties
    HS_init: HeliosheatProperties
    HS_rad: HeliosheatProperties


class PropagationConstants(NamedTuple):
    time_out: ArrayLike
    N_regions: ArrayLike  # Number of inner heliosphere regions
    max_dt: ArrayLike
    particle: ParticleDescription
    R_boundary_effe: HeliosphereBoundRadius  # Boundaries in effective heliosphere
    R_boundary_real: HeliosphereBoundRadius  # Real boundaries heliosphere
    is_high_activity_period: ArrayLike
    LIM: HeliosphereProperties
    HS: HeliosheatProperties

    def _at_index(self, init_zone, rad_zone) -> PropagationConstantsItem:
        def init_index(v):
            return lax.dynamic_index_in_dim(v, init_zone, -1, False)

        def rad_index(v):
            return lax.dynamic_index_in_dim(v, rad_zone, -1, False)

        def initrad_index(v):
            return lax.dynamic_index_in_dim(v, init_zone + rad_zone, -1, False)

        return PropagationConstantsItem(
            time_out=self.time_out,
            N_regions=self.N_regions,
            max_dt=self.max_dt,
            particle=self.particle,
            R_boundary_effe_init=tree_map(init_index, self.R_boundary_effe),
            R_boundary_effe_rad=tree_map(rad_index, self.R_boundary_effe),
            R_boundary_real=tree_map(init_index, self.R_boundary_real),
            is_high_activity_period=tree_map(init_index, self.is_high_activity_period),
            LIM=tree_map(initrad_index, self.LIM),
            HS_init=tree_map(init_index, self.HS),
            HS_rad=tree_map(rad_index, self.HS),
        )
