from jax import Array, lax, numpy as jnp
from jax.typing import ArrayLike

from PyCosmica.structures import delta_m, rhelio, omega, omega, rhelio, \
    high_rigi_suppression_smoothness, high_rigi_suppression_trans_point, \
    PropagationState, PropagationConstantsItem, \
    ConvectionDiffusionTensor, DiffusionTensor, Position3D
from PyCosmica.utils import solar_wind_speeed


# ----------------------------------------------------------------
#  B-field functions
# ----------------------------------------------------------------
def eval_Bth(state: PropagationState, is_polar_region: ArrayLike) -> Array:
    return jnp.where(is_polar_region, state.r * delta_m / (rhelio * jnp.sin(state.th)), 0.0)


def eval_Bph(state: PropagationState, pol_sign: ArrayLike, V_SW: ArrayLike) -> Array:
    return - pol_sign * ((omega * (state.r - rhelio) * jnp.sin(state.th)) / V_SW)


def eval_HMF_Mag(Bth: ArrayLike, Bph: ArrayLike) -> Array:
    return jnp.sqrt(1.0 + Bth * Bth + Bph * Bph)


def eval_sqrtBR2BT2(pol_sign: ArrayLike, Bth: ArrayLike, Bph: ArrayLike) -> Array:
    return jnp.sqrt(pol_sign * pol_sign + Bth * Bth)


def eval_sinPsi(sign_A_sun: ArrayLike, Bph: ArrayLike, HMF_Mag: ArrayLike) -> Array:
    return sign_A_sun * (-Bph / HMF_Mag)


def eval_cosPsi(pol_sign: ArrayLike, Bth: ArrayLike, HMF_Mag: ArrayLike) -> Array:
    return jnp.sqrt(pol_sign * pol_sign + Bth * Bth) / HMF_Mag


def eval_sinZeta(pol_sign: ArrayLike, sign_A_sun: ArrayLike, Bth: ArrayLike) -> Array:
    return sign_A_sun * (Bth / jnp.sqrt(pol_sign * pol_sign + Bth * Bth))


def eval_cosZeta(pol_sign: ArrayLike, sign_A_sun: ArrayLike, Bth: ArrayLike) -> Array:
    return sign_A_sun * (pol_sign / jnp.sqrt(pol_sign * pol_sign + Bth * Bth))


# ----------------------------------------------------------------
#  Derivatives of B-field functions
# ----------------------------------------------------------------
def eval_dBth_dr(state: PropagationState, is_polar_region: ArrayLike) -> Array:
    return jnp.where(is_polar_region, -delta_m / (rhelio * jnp.sin(state.th)), 0.0)


def eval_dBph_dr(state: PropagationState, pol_sign: ArrayLike, V_SW: ArrayLike) -> Array:
    return pol_sign * ((state.r - 2.0 * rhelio) * omega * jnp.sin(state.th)) / (state.r * V_SW)


def eval_dBth_dth(state: PropagationState, is_polar_region: ArrayLike) -> Array:
    val = state.r * delta_m / (rhelio * jnp.sin(state.th) * jnp.sin(state.th)) * (-jnp.cos(state.th))
    return jnp.where(is_polar_region, val, 0.0)


def eval_dBph_dth(state: PropagationState, pol_sign: ArrayLike, V_SW: ArrayLike, dV_SWdth: ArrayLike,
                  DelDirac: ArrayLike) -> Array:
    num = -(state.r - rhelio) * omega * (
            -pol_sign * (jnp.cos(state.th) * V_SW - jnp.sin(state.th) * dV_SWdth)
            + 2.0 * jnp.sin(state.th) * V_SW * DelDirac
    )
    den = V_SW * V_SW
    return num / den


def eval_dBMag_dth(pol_sign: ArrayLike, DelDirac: ArrayLike,
                   Bth: ArrayLike, dBth_dth: ArrayLike,
                   Bph: ArrayLike, dBph_dth: ArrayLike,
                   HMF_Mag: ArrayLike) -> Array:
    num = (pol_sign * (-2.0 * DelDirac)
           + Bth * dBth_dth
           + Bph * dBph_dth)
    return num / HMF_Mag


def eval_dBMag_dr(state: PropagationState, pol_sign: ArrayLike, Bth: ArrayLike,
                  Bph: ArrayLike, dBth_dr: ArrayLike, dBph_dr: ArrayLike,
                  HMF_Mag: ArrayLike):
    num = (pol_sign * (-2.0 * pol_sign / state.r)
           + Bth * dBth_dr
           + Bph * dBph_dr)
    return num / HMF_Mag


def eval_DsinPsi_dr(sign_A_sun: ArrayLike, HMF_Mag: ArrayLike, Bph: ArrayLike,
                    dBph_dr: ArrayLike, dBMag_dr: ArrayLike) -> Array:
    return -sign_A_sun * (dBph_dr * HMF_Mag - Bph * dBMag_dr) / (HMF_Mag * HMF_Mag)


def eval_DsinPsi_dtheta(sign_A_sun: ArrayLike, HMF_Mag: ArrayLike, Bph: ArrayLike,
                        dBph_dth: ArrayLike, dBMag_dth: ArrayLike) -> Array:
    return -sign_A_sun * (
            dBph_dth * HMF_Mag - Bph * dBMag_dth
    ) / (HMF_Mag * HMF_Mag)


def eval_dsqrtBR2BT2_dr(state: PropagationState, pol_sign: ArrayLike, Bth: ArrayLike,
                        sqrtBR2BT2: ArrayLike, dBth_dr: ArrayLike) -> Array:
    num = (pol_sign * (-2.0 * pol_sign / state.r) + Bth * dBth_dr)
    return num / sqrtBR2BT2


def eval_dsqrtBR2BT2_dth(pol_sign: ArrayLike, Bth: ArrayLike, sqrtBR2BT2: ArrayLike,
                         dBth_dth: ArrayLike, DelDirac: ArrayLike) -> Array:
    num = (pol_sign * (-2.0 * DelDirac) + Bth * dBth_dth)
    return num / sqrtBR2BT2


def eval_DcosPsi_dr(state: PropagationState, pol_sign: ArrayLike, Bth: ArrayLike,
                    Bph: ArrayLike, sqrtBR2BT2: ArrayLike, dBth_dr: ArrayLike,
                    dBph_dr: ArrayLike, HMF_Mag: ArrayLike) -> Array:
    num = Bph * (
            -pol_sign * pol_sign * dBph_dr
            + Bph * pol_sign * (-2.0 * pol_sign / state.r)
            + Bth * (-Bth * dBph_dr + Bph * dBth_dr)
    )
    return num / (sqrtBR2BT2 * (HMF_Mag ** 2) * HMF_Mag)


def eval_DcosPsi_dtheta(pol_sign: ArrayLike, Bth: ArrayLike, Bph: ArrayLike,
                        sqrtBR2BT2: ArrayLike, dBph_dth: ArrayLike, dBth_dth: ArrayLike,
                        HMF_Mag: ArrayLike, DelDirac: ArrayLike) -> Array:
    num = Bph * (
            -pol_sign * pol_sign * dBph_dth
            + Bph * pol_sign * (-2.0 * DelDirac)
            + Bth * (-Bth * dBph_dth + Bph * dBth_dth)
    )
    return num / (sqrtBR2BT2 * (HMF_Mag ** 2) * HMF_Mag)


def eval_DsinZeta_dr(sign_A_sun: ArrayLike, Bth: ArrayLike,
                     dBth_dr: ArrayLike, sqrtBR2BT2: ArrayLike,
                     dsqrtBR2BT2_dr: ArrayLike) -> Array:
    num = dBth_dr * sqrtBR2BT2 - Bth * dsqrtBR2BT2_dr
    return sign_A_sun * (num / (sqrtBR2BT2 * sqrtBR2BT2))


def eval_DsinZeta_dtheta(sign_A_sun: ArrayLike, Bth: ArrayLike,
                         dBth_dth: ArrayLike, sqrtBR2BT2: ArrayLike,
                         dsqrtBR2BT2_dth: ArrayLike) -> Array:
    num = dBth_dth * sqrtBR2BT2 - Bth * dsqrtBR2BT2_dth
    return sign_A_sun * (num / (sqrtBR2BT2 * sqrtBR2BT2))


def eval_DcosZeta_dr(state: PropagationState, sign_A_sun: ArrayLike, pol_sign: ArrayLike,
                     sqrtBR2BT2: ArrayLike, dsqrtBR2BT2_dr: ArrayLike) -> Array:
    num = ((-2.0 * pol_sign / state.r) * sqrtBR2BT2
           - pol_sign * dsqrtBR2BT2_dr)
    return sign_A_sun * (num / (sqrtBR2BT2 * sqrtBR2BT2))


def eval_DcosZeta_dtheta(sign_A_sun: ArrayLike, pol_sign: ArrayLike,
                         sqrtBR2BT2: ArrayLike, dsqrtBR2BT2_dth: ArrayLike,
                         DelDirac: ArrayLike) -> Array:
    num = ((-2.0 * DelDirac) * sqrtBR2BT2
           - pol_sign * dsqrtBR2BT2_dth)
    return sign_A_sun * (num / (sqrtBR2BT2 * sqrtBR2BT2))


# ----------------------------------------------------------------
#  Diffusion tensor functions
# ----------------------------------------------------------------

def square_root_diffusion_term(state: PropagationState, const: PropagationConstantsItem,
                               conv_diff: ConvectionDiffusionTensor) -> Array:
    rr = jnp.sqrt(2. * conv_diff.rr)

    def in_heliosphere():
        sin_theta = jnp.sin(state.th)
        tr = 2. * conv_diff.tr / state.r
        tt = 2. * conv_diff.tt / (state.r * state.r)
        pr = 2. * conv_diff.pr / (state.r * sin_theta)
        pt = 2. * conv_diff.pt / (state.r * state.r * sin_theta)
        pp = 2. * conv_diff.pp / (state.r * state.r * sin_theta * sin_theta)

        tr = tr / rr
        pr = pr / rr
        tt = jnp.sqrt(tt - tr * tr)
        pt = 1. / tt * (pt - tr * pr)
        pp = jnp.sqrt(pp - pr * pr - pt * pt)

        return DiffusionTensor(rr, tr, tt, pr, pt, pp)

    def out_heliosphere():
        return DiffusionTensor(rr, 0., 0., 0., 0., 0.)

    return lax.cond(state.rad_zone < const.N_regions, in_heliosphere, out_heliosphere)


def advective_term_radius(state: PropagationState, const: PropagationConstantsItem,
                          conv_diff: ConvectionDiffusionTensor, v_drift_rad: ArrayLike) -> Array:
    V_sw = solar_wind_speeed(state, const)

    def in_heliosphere():
        return 2. * conv_diff.rr / state.r + conv_diff.DKrr_dr + conv_diff.tr / (
                state.r * jnp.tan(state.th)) + conv_diff.DKtr_dt / state.r - v_drift_rad - V_sw

    def out_heliosphere():
        return 2. * conv_diff.rr / state.r + conv_diff.DKrr_dr - V_sw

    return lax.cond(state.rad_zone < const.N_regions, in_heliosphere, out_heliosphere)


def advective_term_theta(state: PropagationState, const: PropagationConstantsItem,
                         conv_diff: ConvectionDiffusionTensor, v_drift_th: ArrayLike) -> Array:
    def in_heliosphere():
        r2 = state.r * state.r
        return conv_diff.tr / r2 + conv_diff.tt / (
                jnp.tan(state.th) * r2) + conv_diff.DKrt_dr / state.r + conv_diff.DKtt_dt / r2 - v_drift_th / state.r

    def out_heliosphere():
        return 0.

    return lax.cond(state.rad_zone < const.N_regions, in_heliosphere, out_heliosphere)


def advective_term_phi(state: PropagationState, const: PropagationConstantsItem,
                       conv_diff: ConvectionDiffusionTensor, v_drift_phi: ArrayLike) -> Array:
    def in_heliosphere():
        sin_theta = jnp.sin(state.th)
        r2 = state.r * state.r
        return conv_diff.pr / (r2 * sin_theta) + conv_diff.DKrp_dr / (state.r * sin_theta) + conv_diff.DKtp_dt / (
                r2 * sin_theta) + (-v_drift_phi / (state.r * sin_theta))

    def out_heliosphere():
        return 0.

    return lax.cond(state.rad_zone < const.N_regions, in_heliosphere, out_heliosphere)



