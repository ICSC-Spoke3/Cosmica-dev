from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.typing import ArrayLike
from math import floor
from tqdm import tqdm

from PyCosmica.structures import QuasiParticle, HeliosphereBoundRadius, SimParametersJit, HeliosphereProperties, \
    HeliosheatProperties
from PyCosmica.utils import pytrees_stack, pytrees_flatten
from PyCosmica.utils.heliosphere_model import radial_zone_scalar


class PropagationConstants(NamedTuple):
    time_out: ArrayLike
    N_regions: int  # Number of inner heliosphere regions
    R_boundary_effe: HeliosphereBoundRadius  # Boundaries in effective heliosphere
    R_boundary_real: HeliosphereBoundRadius  # Real boundaries heliosphere
    is_high_activity_period: ArrayLike
    LIM: HeliosphereProperties
    HS: HeliosheatProperties


class PropagationState(NamedTuple):
    r: ArrayLike
    th: ArrayLike
    phi: ArrayLike
    R: ArrayLike
    t_fly: ArrayLike
    rad_zone: ArrayLike
    init_zone: ArrayLike

    @property
    def _particle(self):
        return QuasiParticle(self.r, self.th, self.phi, self.R, self.t_fly)


def propagation_kernel(state: PropagationState, const: PropagationConstants) -> PropagationState:
    data = state._asdict()
    data['t_fly'] += 1
    data['r'] += const.LIM.V0.at[state.init_zone + state.rad_zone].get()
    # data['r'] += jnp.stack(const.LIM.V0)[state.init_zone + state.rad_zone]
    # jax.debug.print('{}', data['r'])
    return PropagationState(**data)


def propagation_condition(state: PropagationState, const: PropagationConstants) -> bool:
    return (state.rad_zone >= 0) & (const.time_out > state.t_fly)


def propagation_loop(init_state: PropagationState, const: PropagationConstants) -> QuasiParticle:
    def propagation_condition_const(state):
        return propagation_condition(state, const)

    def propagation_kernel_const(state):
        return propagation_kernel(state, const)

    final_state = lax.while_loop(
        propagation_condition_const,
        propagation_kernel_const,
        init_state
    )
    return final_state._particle


def propagation_source(state: PropagationState, const: PropagationConstants, rep: int):
    print('compile')
    return jax.vmap(propagation_loop, in_axes=None, axis_size=rep)(state, const)


def propagation_vector(sim: SimParametersJit):
    hs = sim.heliosphere_to_be_simulated

    for k, v in sim._asdict().items():
        if isinstance(v, list):
            print(f'{k}: {len(v)} (len)')
        else:
            print(f'{k}: {v}')
    print()

    for k, v in hs._asdict().items():
        if isinstance(v, list):
            print(f'{k}: {len(v)} (len)')
        else:
            print(f'{k}: {v}')
    print()

    const = PropagationConstants(
        time_out=100000,
        N_regions=hs.N_regions,
        R_boundary_effe=pytrees_stack(hs.R_boundary_effe),
        R_boundary_real=pytrees_stack(hs.R_boundary_real),
        is_high_activity_period=pytrees_stack(hs.is_high_activity_period),
        LIM=pytrees_stack(sim.prop_medium),
        HS=pytrees_stack(sim.prop_heliosheat),
    )

    part_per_pos = floor(sim.N_part / sim.N_initial_positions)

    base_states = pytrees_stack([
        PropagationState(
            R=0,
            t_fly=0,
            rad_zone=radial_zone_scalar(b, hs.N_regions, p),
            init_zone=i,
            **p._asdict(),
        ) for i, (p, b) in enumerate(zip(sim.initial_position, hs.R_boundary_effe))
    ])

    sources_map = jax.vmap(propagation_source, in_axes=(0, None, None))
    sources_map_jit = jax.jit(sources_map, static_argnames='rep')

    out = []
    for R in tqdm(sim.T_centr):
        states = base_states._replace(R=jnp.full_like(base_states.R, R))
        res = sources_map_jit(states, const, part_per_pos)
        out.append(pytrees_flatten(res))

    return out
