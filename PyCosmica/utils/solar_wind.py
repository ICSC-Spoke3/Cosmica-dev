from jax import Array, lax, numpy as jnp

from PyCosmica.structures import PropagationState, PropagationConstantsItem, L_tl, s_tl, V_high
from PyCosmica.utils import boundary, smooth_transition


def solar_wind_speeed(state: PropagationState, const: PropagationConstantsItem) -> Array:
    V0 = lax.select(state.rad_zone < const.N_regions, const.LIM.V0, const.HS_init.V0)

    def inner_heliosphere():
        def high_activity():
            return V0

        def low_activity():
            V_ang = lax.select(V_high / V0 <= 2., V_high / V0 - 1., 1.)
            return lax.select(jnp.abs(jnp.cos(state.th)) > V_ang,
                              V_high,
                              V0 * (1 + jnp.abs(jnp.cos(state.th))))

        return lax.cond(const.is_high_activity_period, high_activity, low_activity)

    def near_heliosheat():
        R_ts_effe = boundary(state.th, state.phi, const.R_boundary_effe_init.R_ts_nose,
                             const.R_boundary_effe_init.R_ts_tail)

        def is_heliosheat():
            R_ts_real = boundary(state.th, state.phi, const.R_boundary_real.R_ts_nose,
                                 const.R_boundary_real.R_ts_tail)

            decrease_factor = smooth_transition(1, 1. / s_tl, R_ts_effe, L_tl, state.r)
            decrease_factor = lax.select(state.r > R_ts_effe,
                                         decrease_factor * (R_ts_real / (R_ts_real - R_ts_effe + state.r)) ** 2,
                                         decrease_factor)
            return V0 * decrease_factor

        return lax.cond(state.r >= R_ts_effe - L_tl, is_heliosheat, inner_heliosphere)

    return lax.cond(state.rad_zone >= const.N_regions - 1, near_heliosheat, inner_heliosphere)


def solar_wind_derivative(state: PropagationState, const: PropagationConstantsItem) -> Array:
    V0 = lax.select(state.rad_zone < const.N_regions, const.LIM.V0, const.HS_init.V0)

    def high_activity():
        return 0.

    def low_activity():
        V_ang = lax.select(V_high / V0 <= 2., V_high / V0 - 1., 1.)
        cos_th, sin_th = jnp.cos(state.th), jnp.sin(state.th)
        return lax.select(jnp.abs(cos_th) > V_ang, # if
                          0.,
                          lax.select(cos_th < 0, # elif
                                     V0 * sin_th,
                                     lax.select(cos_th == 0, # elif
                                                0.,
                                                -V0 * sin_th))) #else

    return lax.cond(const.is_high_activity_period, high_activity, low_activity)
