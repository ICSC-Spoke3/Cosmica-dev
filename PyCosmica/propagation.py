import jax
from jax import Array, lax, numpy as jnp
from math import floor
from tqdm import tqdm

from PyCosmica.sde import diffusion_tensor_symmetric
from PyCosmica.structures import QuasiParticle, SimParametersJit, PropagationState, PropagationConstants
from PyCosmica.utils import pytrees_stack, pytrees_flatten
from PyCosmica.utils.heliosphere_model import radial_zone_scalar


def propagation_kernel(state: PropagationState, const: PropagationConstants) -> PropagationState:
    const_item = const._at_index(state.init_zone, state.rad_zone)
    key, subkey = jax.random.split(state.key)
    x, y, z, w = jax.random.normal(subkey, (4,))

    tmp = diffusion_tensor_symmetric(state, const_item, w)
    # data = state._asdict()
    # data['t_fly'] += 1
    # data['r'] = jax.random.uniform(state.rand)
    # jax.debug.print('{}', const_items.LIM)
    # state = state._replace(r=state.r + const_item.LIM.K0_perp[0])
    # data['r'] += const.LIM.V0.at[state.init_zone + state.rad_zone].get()
    # data['r'] += jnp.stack(const.LIM.V0)[state.init_zone + state.rad_zone]
    # jax.debug.print('{}', data['r'])
    # data['rand'] = nxt
    state = state._replace(t_fly=state.t_fly + 1)
    state = state._replace(key=key)
    return state


def propagation_condition(state: PropagationState, const: PropagationConstants) -> bool:
    return (state.rad_zone >= 0) & (const.time_out > state.t_fly)


def propagation_loop(init_state: PropagationState, const: PropagationConstants, key: Array) -> QuasiParticle:
    init_state = init_state._replace(key=key)

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


def propagation_source(state: PropagationState, const: PropagationConstants, key: Array, rep: int):
    print('compile')
    keys = jax.random.split(key, rep)
    return jax.vmap(propagation_loop, in_axes=(None, None, 0))(state, const, keys)


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
        particle=sim.ion_to_be_simulated,
        N_regions=hs.N_regions,
        R_boundary_effe=pytrees_stack(hs.R_boundary_effe),
        R_boundary_real=pytrees_stack(hs.R_boundary_real),
        is_high_activity_period=pytrees_stack(hs.is_high_activity_period),
        LIM=pytrees_stack(sim.prop_medium),
        HS=pytrees_stack(sim.prop_heliosheat),
    )

    part_per_pos = floor(sim.N_part / sim.N_initial_positions)

    keys = jax.random.split(jax.random.key(42), (sim.N_T, sim.N_initial_positions))

    base_states = pytrees_stack([
        PropagationState(
            R=0,
            t_fly=0,
            rad_zone=radial_zone_scalar(b, hs.N_regions, p),
            init_zone=i,
            key=0,
            **p._asdict(),
        ) for i, (p, b) in enumerate(zip(sim.initial_position, hs.R_boundary_effe))
    ])

    sources_map = jax.vmap(propagation_source, in_axes=(0, None, 0, None))
    sources_map_jit = jax.jit(sources_map, static_argnames='rep')

    print(jax.make_jaxpr(sources_map, static_argnums=3)(base_states, const, keys[0], part_per_pos))
    # print(sources_map_jit.lower(base_states, const, keys[0], part_per_pos).as_text())

    out = []
    for R, k in tqdm(zip(sim.T_centr[:10], keys), total=10):
        states = base_states._replace(R=jnp.full_like(base_states.R, R))
        res = sources_map_jit(states, const, k, part_per_pos)
        out.append(pytrees_flatten(res))

    return out
