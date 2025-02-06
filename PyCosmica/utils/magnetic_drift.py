from jax import Array, lax, numpy as jnp
from jax.typing import ArrayLike

from PyCosmica.structures import *
from PyCosmica.utils import *


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


def eval_E_drift(state: PropagationState, is_polar_region: ArrayLike, V_sw: ArrayLike) -> Array:
    def is_polar():
        return ((delta_m * state.r * V_sw + omega * R_helio * (state.r - R_helio) * jnp.sin(state.th) ** 2) ** 2 +
                (R_helio * V_sw * jnp.sin(state.th)) ** 2)

    def is_not_polar():
        return (omega * (state.r - R_helio) * jnp.sin(state.th)) ** 2 + V_sw ** 2

    return lax.cond(is_polar_region, is_polar, is_not_polar)


def eval_C_drift_reg(state: PropagationState, const: PropagationConstantsItem,
                     is_polar_region: ArrayLike, Ka: ArrayLike, fth: ArrayLike, E: ArrayLike) -> Array:
    red_factor = state.R ** 2 / (state.R ** 2 + const.LIM.P0_d ** 2)
    return red_factor * lax.select(
        is_polar_region,
        jnp.sin(state.th) * R_helio,
        omega
    ) * fth * Ka * state.r / (const.LIM.A_sun * E ** 2)


def eval_C_drift_ns(state: PropagationState, const: PropagationConstantsItem,
                    is_polar_region: ArrayLike, Ka: ArrayLike,
                    E: ArrayLike, Dftheta_dtheta: ArrayLike, V_sw: ArrayLike) -> Array:
    red_factor = state.R ** 2 / (state.R ** 2 + const.LIM.P0_dNS ** 2)

    return red_factor * lax.select(
        is_polar_region,
        R_helio ** 2 * jnp.sin(state.th) ** 2,
        1.
    ) * V_sw * Dftheta_dtheta * Ka * state.r / (const.LIM.A_sun * E)


def drift_pm89(state: PropagationState, const: PropagationConstantsItem,
               is_polar_region: ArrayLike, Ka: ArrayLike, fth: ArrayLike, Dftheta_dtheta: ArrayLike,
               V_sw: ArrayLike, dV_dth: ArrayLike, high_rigi_supp: ArrayLike) -> Position3D:
    def apply_high_rigi_supp(r_, th_, phi_):
        return r_ * high_rigi_supp, th_ * high_rigi_supp, phi_ * high_rigi_supp

    def is_polar():
        # Polar region
        E = eval_E_drift(state, 1., V_sw)
        C = eval_C_drift_reg(state, const, 1., const.LIM.A_sun, Ka, fth, E)
        # Regular drift contribution
        v_r = - C * omega * R_helio * 2. * (state.r - R_helio) * jnp.sin(state.th) * (
                (2. * (delta_m * state.r) ** 2 + (R_helio * jnp.sin(state.th)) ** 2) * V_sw ** 3 * jnp.cos(state.th)
                - 0.5 * ((delta_m * state.r * V_sw) ** 2
                         - (omega * R_helio * (state.r - R_helio) * jnp.sin(state.th) ** 2) ** 2
                         + (R_helio * V_sw * jnp.sin(state.th)) ** 2)
                * jnp.sin(state.th) * dV_dth)
        v_th = - C * omega * R_helio * V_sw * jnp.sin(state.th) ** 2 * (
                2. * state.r * (state.r - R_helio) * (
                state.r * (delta_m * V_sw) ** 2 + (omega * R_helio) ** 2 * (state.r - R_helio) * jnp.sin(state.th) ** 4)
                - (4. * state.r - 3. * R_helio) * E)
        v_phi = 2. * C * V_sw * (
                - (delta_m * state.r) ** 2 * (delta_m * state.r + R_helio * jnp.cos(state.th)) * V_sw ** 3
                + 2. * delta_m * state.r * E * V_sw
                - (omega * R_helio) ** 2 * (state.r - R_helio) * jnp.sin(state.th) ** 4 * (
                        delta_m * state.r ** 2 * V_sw - R_helio * (state.r - R_helio) * V_sw * jnp.cos(state.th)
                        + R_helio * (state.r - R_helio) * jnp.sin(state.th) * dV_dth))

        # ns contribution
        C = eval_C_drift_ns(state, const, 1., Ka, E, Dftheta_dtheta, V_sw)
        v_r += - C * omega * jnp.sin(state.th) * (state.r - R_helio)
        v_phi += - C * V_sw

        return Position3D(*apply_high_rigi_supp(v_r, v_th, v_phi))

    def is_not_polar():
        # Not Polar region (assume B_th = 0)
        E = eval_E_drift(state, 0., V_sw)
        # Regular drift contribution
        C = eval_C_drift_reg(state, const, 1., const.LIM.A_sun, Ka, fth, E)
        v_r = - 2. * C * (state.r - R_helio) * (
                0.5 * (omega ** 2 * (state.r - R_helio) ** 2 * jnp.sin(state.th) ** 2 - V_sw ** 2) * jnp.sin(
            state.th) * dV_dth
                + V_sw ** 3 * jnp.cos(state.th))
        v_th = - C * V_sw * jnp.sin(state.th) * (
                2. * omega ** 2 * state.r * (state.r - R_helio) ** 2 * jnp.sin(state.th) ** 2
                - (4. * state.r - 3. * R_helio) * E)
        v_phi = 2. * C * V_sw * omega * (state.r - R_helio) ** 2 * jnp.sin(state.th) * (
                V_sw * jnp.cos(state.th) - jnp.sin(state.th) * dV_dth)

        # ns contribution
        C = eval_C_drift_ns(state, const, 1., Ka, E, Dftheta_dtheta, V_sw)
        v_r += - C * omega * (state.r - R_helio) * jnp.sin(state.th)
        v_phi += - C * V_sw

        return Position3D(*apply_high_rigi_supp(v_r, v_th, v_phi))

    return lax.cond(is_polar_region, is_polar, is_not_polar)


def Gamma_Bfield(state: PropagationState, V_sw: ArrayLike) -> Array:
    return (omega * (state.r - R_helio) * jnp.sin(state.th)) / V_sw


def delta_Bfield(state: PropagationState) -> Array:
    return state.r / R_helio * delta_m / jnp.sin(state.th)


def eval_Ka(state: PropagationState, const: PropagationConstantsItem) -> Array:
    return (sign(const.particle.Z) * GeV * beta_R(state, const)) * state.R / 3.


def eval_fth(state: PropagationState, theta_mez: ArrayLike) -> Array:
    def neutral_sheet_not_flat():
        return 1. / jnp.arccos(PI / (2. * theta_mez) - 1.) * jnp.arctan(
            (1. - (2. * state.th / PI)) * jnp.tan(jnp.arccos(PI / (2. * theta_mez) - 1.)))

    def neutral_sheet_flat():
        # if (th > Pi / 2) return 1.; if (th < Pi / 2) return -1.; if (th == Pi / 2) return 0.;
        return (state.th > PI / 2) * 1. + (state.th < PI / 2) * -1.

    return lax.cond(theta_mez < PI / 2, neutral_sheet_not_flat, neutral_sheet_flat)


def eval_Dftheta_dtheta(state: PropagationState, theta_mez: ArrayLike) -> Array:
    def neutral_sheet_not_flat():
        a_f = jnp.arccos(PI / (2. * theta_mez) - 1.)
        return - 2. * jnp.tan(a_f) / (a_f * PI * (
                1. + (1. - 2. * state.th / PI) * (1. - 2. * state.th / PI) * jnp.tan(a_f) ** 2))

    def neutral_sheet_flat():
        return 0.

    return lax.cond(theta_mez < PI / 2, neutral_sheet_not_flat, neutral_sheet_flat)
