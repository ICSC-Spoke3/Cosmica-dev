from jax import Array, lax, numpy as jnp

from PyCosmica.structures import *
from PyCosmica.utils.solar_wind import solar_wind_speeed, solar_wind_derivative
from PyCosmica.utils.generic_math import smooth_transition, sign, beta_R


def eval_p0_drift_suppression_factor(regular: bool, solar_phase: int, tilt_angle_deg: float, ssn: float) -> float:
    if regular:  # Regular drift
        initial_val = 0.5
        final_val = 4.
        if solar_phase == 0:
            center_of_transition = 73.
            smoothness = 1.
        else:
            center_of_transition = 65.
            smoothness = 5.
    else:  # NS drift
        initial_val = 0.5
        final_val = ssn / 50.
        if solar_phase == 0:
            center_of_transition = 68.
            smoothness = 1.
        else:
            center_of_transition = 57.
            smoothness = 5.

    return smooth_transition(initial_val, final_val, center_of_transition, smoothness, tilt_angle_deg)


def eval_high_rigidity_drift_suppression_plateau(solar_phase: int, tilt_angle_deg: float) -> float:
    # Plateau time dependence
    if solar_phase == 0:
        center_of_transition = 35.
        smoothness = 5.
    else:
        center_of_transition = 40.
        smoothness = 5.

    return 1. - smooth_transition(1., 0., center_of_transition, smoothness, tilt_angle_deg)


# ----------------------------------------------------------------
#  Drift model functions
# ----------------------------------------------------------------
def eval_high_rigi_supp(state: PropagationState, const: PropagationConstantsItem) -> Array:
    return const.LIM.plateau + (1. - const.LIM.plateau) / (
            1. + jnp.exp(high_rigi_suppression_smoothness * (state.R - high_rigi_suppression_trans_point))
    )


def eval_E_drift_polar(state: PropagationState, V_sw: ArrayLike) -> Array:
    return ((delta_m * state.r * V_sw + omega * R_helio * (state.r - R_helio) * jnp.sin(state.th) ** 2) ** 2 +
            (R_helio * V_sw * jnp.sin(state.th)) ** 2)


def eval_E_drift_nonpolar(state: PropagationState, V_sw: ArrayLike) -> Array:
    return (omega * (state.r - R_helio) * jnp.sin(state.th)) ** 2 + V_sw ** 2


def eval_C_drift_reg_polar(state: PropagationState, const: PropagationConstantsItem, Ka: ArrayLike, fth: ArrayLike,
                           E: ArrayLike) -> Array:
    red_factor = state.R ** 2 / (state.R ** 2 + const.LIM.P0_d ** 2)
    return red_factor * jnp.sin(state.th) * R_helio * fth * Ka * state.r / (const.LIM.A_sun * E ** 2)


def eval_C_drift_reg_nonpolar(state: PropagationState, const: PropagationConstantsItem, Ka: ArrayLike, fth: ArrayLike,
                              E: ArrayLike) -> Array:
    red_factor = state.R ** 2 / (state.R ** 2 + const.LIM.P0_d ** 2)
    return red_factor * omega * fth * Ka * state.r / (const.LIM.A_sun * E ** 2)


def eval_C_drift_ns_polar(state: PropagationState, const: PropagationConstantsItem, Ka: ArrayLike, E: ArrayLike,
                          Dftheta_dtheta: ArrayLike, V_sw: ArrayLike) -> Array:
    red_factor = state.R ** 2 / (state.R ** 2 + const.LIM.P0_dNS ** 2)
    return red_factor * R_helio ** 2 * jnp.sin(state.th) ** 2 * V_sw * Dftheta_dtheta * Ka * state.r / (
            const.LIM.A_sun * E)


def eval_C_drift_ns_nonpolar(state: PropagationState, const: PropagationConstantsItem, Ka: ArrayLike, E: ArrayLike,
                             Dftheta_dtheta: ArrayLike, V_sw: ArrayLike) -> Array:
    red_factor = state.R ** 2 / (state.R ** 2 + const.LIM.P0_dNS ** 2)
    return red_factor * V_sw * Dftheta_dtheta * Ka * state.r / (const.LIM.A_sun * E)


def drift_pm89(state: PropagationState, const: PropagationConstantsItem) -> Position3D:
    Ka = eval_Ka(state, const)
    tilt_angle = const.LIM.tilt_angle
    tilt_pos_th = PI / 2. - tilt_angle
    state_tilted = state._replace(th=tilt_pos_th)

    V_sw = solar_wind_speeed(state, const)
    V_sw_pm89 = solar_wind_speeed(state_tilted, const)
    dV_dth = solar_wind_derivative(state, const)

    dth_ns = jnp.fabs(
        (GeV / (C * AU_M)) * (2. * state.r * state.R) / (const.LIM.A_sun * jnp.sqrt(
            1 + Gamma_Bfield(state_tilted, V_sw_pm89) ** 2 +
            state._is_polar_region * delta_Bfield(state_tilted) ** 2)))
    th_mez = PI / 2. - .5 * jnp.sin(jnp.minimum(PI / 2., tilt_angle + dth_ns))
    f_th, df_th_dth = eval_fth_dfth(state, th_mez)

    high_rigi_supp = eval_high_rigi_supp(state, const)

    def is_polar():
        E_drift = eval_E_drift_polar(state, V_sw)
        C_drift_reg = eval_C_drift_reg_polar(state, const, Ka, f_th, E_drift)

        v_r = - C_drift_reg * omega * R_helio * 2. * (state.r - R_helio) * jnp.sin(state.th) * (
                (2. * (delta_m * state.r) ** 2 + (R_helio * jnp.sin(state.th)) ** 2) * V_sw ** 3 * jnp.cos(state.th)
                - 0.5 * ((delta_m * state.r * V_sw) ** 2
                         - (omega * R_helio * (state.r - R_helio) * jnp.sin(state.th) ** 2) ** 2
                         + (R_helio * V_sw * jnp.sin(state.th)) ** 2)
                * jnp.sin(state.th) * dV_dth)
        v_th = - C_drift_reg * omega * R_helio * V_sw * jnp.sin(state.th) ** 2 * (
                2. * state.r * (state.r - R_helio) * (
                state.r * (delta_m * V_sw) ** 2 + (omega * R_helio) ** 2 * (state.r - R_helio) * jnp.sin(state.th) ** 4)
                - (4. * state.r - 3. * R_helio) * E_drift)
        v_phi = 2. * C_drift_reg * V_sw * (
                - (delta_m * state.r) ** 2 * (delta_m * state.r + R_helio * jnp.cos(state.th)) * V_sw ** 3
                + 2. * delta_m * state.r * E_drift * V_sw
                - (omega * R_helio) ** 2 * (state.r - R_helio) * jnp.sin(state.th) ** 4 * (
                        delta_m * state.r ** 2 * V_sw - R_helio * (state.r - R_helio) * V_sw * jnp.cos(state.th)
                        + R_helio * (state.r - R_helio) * jnp.sin(state.th) * dV_dth))

        C_drift_ns = eval_C_drift_ns_polar(state, const, Ka, E_drift, df_th_dth, V_sw)
        v_r += -C_drift_ns * omega * jnp.sin(state.th) * (state.r - R_helio)
        v_phi += -C_drift_ns * V_sw

        return Position3D(v_r * high_rigi_supp, v_th * high_rigi_supp, v_phi * high_rigi_supp)

    def is_not_polar():
        E_drift = eval_E_drift_nonpolar(state, V_sw)
        C_drift_reg = eval_C_drift_reg_nonpolar(state, const, Ka, f_th, E_drift)
        v_r = - 2. * C_drift_reg * (state.r - R_helio) * (
                0.5 * ((omega * (state.r - R_helio) * jnp.sin(state.th)) ** 2 - V_sw ** 2) * jnp.sin(state.th) * dV_dth
                + V_sw ** 3 * jnp.cos(state.th))
        v_th = - C_drift_reg * V_sw * jnp.sin(state.th) * (
                2. * omega ** 2 * state.r * (state.r - R_helio) ** 2 * jnp.sin(state.th) ** 2
                - (4. * state.r - 3. * R_helio) * E_drift)
        v_phi = 2. * C_drift_reg * V_sw * omega * (state.r - R_helio) ** 2 * jnp.sin(state.th) * (
                V_sw * jnp.cos(state.th) - jnp.sin(state.th) * dV_dth)

        C_drift_ns = eval_C_drift_ns_nonpolar(state, const, Ka, E_drift, df_th_dth, V_sw)
        v_r += - C_drift_ns * omega * (state.r - R_helio) * jnp.sin(state.th)
        v_phi += - C_drift_ns * V_sw
        return Position3D(v_r * high_rigi_supp, v_th * high_rigi_supp, v_phi * high_rigi_supp)

    return lax.cond(state._is_polar_region, is_polar, is_not_polar)


def Gamma_Bfield(state: PropagationState, V_sw: ArrayLike) -> Array:
    return (omega * (state.r - R_helio) * jnp.sin(state.th)) / V_sw


def delta_Bfield(state: PropagationState) -> Array:
    return state.r / R_helio * delta_m / jnp.sin(state.th)


def eval_Ka(state: PropagationState, const: PropagationConstantsItem) -> Array:
    return (sign(const.particle.Z) * GeV * beta_R(state, const)) * state.R / 3.


def eval_fth_dfth(state: PropagationState, th_mez: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    def neutral_sheet_not_flat():
        a_f = jnp.arccos(PI / (2. * th_mez) - 1.)
        b_f = 1. - 2. * state.th / PI
        f_th = 1. / a_f * jnp.arctan(b_f * jnp.tan(a_f))
        df_th = - 2. * jnp.tan(a_f) / (a_f * PI * (1. + b_f ** 2 * jnp.tan(a_f) ** 2))
        return f_th, df_th

    def neutral_sheet_flat():
        # if (th > Pi / 2) return 1.; if (th < Pi / 2) return -1.; if (th == Pi / 2) return 0.;
        return (state.th > PI / 2) * 1. + (state.th < PI / 2) * -1., 0.

    return lax.cond(th_mez < PI / 2, neutral_sheet_not_flat, neutral_sheet_flat)
