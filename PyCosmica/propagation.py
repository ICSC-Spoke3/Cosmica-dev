import jax
from jax import Array, lax, numpy as jnp
from math import floor
from tqdm import tqdm

from PyCosmica.sde import diffusion_tensor_symmetric
from PyCosmica.structures import *
from PyCosmica.utils import *


def propagation_kernel(state: PropagationState, const: PropagationConstants) -> PropagationState:
    const_item = const._at_index(state.init_zone, state.rad_zone)

    dt = const_item.max_dt

    key, subkey = jax.random.split(state.key)
    x, y, z, w = jax.random.normal(subkey, (4,))

    conv_diff = diffusion_tensor_symmetric(state, const_item, w)

    diff = square_root_diffusion_term(state, const_item, conv_diff)

    for k in diff:
        state = state._replace(R=lax.select(jnp.isnan(k) | jnp.isinf(k), -jnp.inf, state.R))

    adv_term = advective_term(state, const_item, conv_diff)
    en_loss = energy_loss(state, const_item)

    dt = adaptive_dt(const_item, diff, adv_term)

    new_r = state.r + adv_term.r * dt + x * diff.rr * jnp.sqrt(dt)

    def in_mirror():
        return state

    def out_mirror():
        return state._replace(
            r=new_r,
            th=state.th + adv_term.th * dt + (x * diff.tr + y * diff.tt) * jnp.sqrt(dt),
            phi=state.phi + adv_term.phi * dt + (x * diff.pr + y * diff.pt + z * diff.pp) * jnp.sqrt(dt),
            R=state.R + en_loss * dt,
            t_fly=state.t_fly + dt,
        )

    state = lax.cond(new_r < R_mirror, in_mirror, out_mirror)


    th = jnp.fabs(state.th)
    th = jnp.fabs(jnp.fmod(2. * PI + sign(PI - th) * th, PI))
    th = lax.select(th > theta_south_limit, 2. * theta_south_limit - th, th)
    th = lax.select(th < theta_north_limit, 2. * theta_north_limit - th, th)
    phi = jnp.fmod(state.phi, 2. * PI)
    phi = jnp.fmod(2. * PI + phi, 2. * PI)
    state = state._replace(th=th, phi=phi)

    return state._replace(
        rad_zone=radial_zone(const_item.R_boundary_effe_init, const_item.N_regions, state._position),
        key=key,
    )


def propagation_condition(state: PropagationState, const: PropagationConstants) -> bool:
    return (state.rad_zone >= 0) & (const.time_out > state.t_fly) & (state.R < -1)


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
        time_out=200,
        particle=sim.ion_to_be_simulated,
        min_dt=.01,
        max_dt=50.,
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
            R=0.,
            t_fly=0.,
            rad_zone=radial_zone_scalar(b, hs.N_regions, p),
            init_zone=i,
            key=0,
            **p._asdict(),
        ) for i, (p, b) in enumerate(zip(sim.initial_position, hs.R_boundary_effe))
    ])

    sources_map = jax.vmap(propagation_source, in_axes=(0, None, 0, None))
    sources_map_jit = jax.jit(sources_map, static_argnames='rep')

    print(jax.make_jaxpr(sources_map, static_argnums=3)(base_states, const, keys[0], part_per_pos))
    with open('tmp.stablehlo', 'w') as f:
        f.write(sources_map_jit.lower(base_states, const, keys[0], part_per_pos).as_text())

    out = []
    for R, k in tqdm(zip(sim.T_centr, keys), total=len(sim.T_centr)):
        states = base_states._replace(R=jnp.full_like(base_states.R, R))
        res = sources_map_jit(states, const, k, part_per_pos)
        out.append(pytrees_flatten(res))

    out: QuasiParticle = pytrees_flatten(pytrees_stack(out))
    print((out.R == -1).mean())
    print(out.t_fly.mean())
    return out
