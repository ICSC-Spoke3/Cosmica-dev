from PyCosmica.structures import delta_m, R_helio, omega, PropagationState
from jax.typing import ArrayLike
from jax import Array

import jax
import jax.numpy as jnp

# ----------------------------------------------------------------
#  B-field functions
# ----------------------------------------------------------------
def eval_Bth(state: PropagationState, is_polar_region: ArrayLike) -> Array:
    return jnp.where(is_polar_region, state.r * delta_m / (R_helio * jnp.sin(state.th)), 0.0)

def eval_Bph(state: PropagationState, pol_sign: ArrayLike, V_SW: ArrayLike) -> Array:
    return - pol_sign * ((omega * (state.r - R_helio) * jnp.sin(state.th)) / V_SW)

def eval_HMF_Mag(Bth: ArrayLike, Bph: ArrayLike) -> Array:
    return jnp.sqrt(1.0 + Bth * Bth + Bph * Bph)

def eval_sqrtBR2BT2(pol_sign: ArrayLike, Bth: ArrayLike, Bph: ArrayLike) -> Array:
    return jnp.sqrt(pol_sign * pol_sign + Bth * Bth)

def eval_sinPsi(sign_asun: ArrayLike, Bph: ArrayLike, HMF_Mag: ArrayLike) -> Array:
    return sign_asun * (-Bph / HMF_Mag)

def eval_cosPsi(pol_sign: ArrayLike, Bth: ArrayLike, HMF_Mag: ArrayLike) -> Array:
    return jnp.sqrt(pol_sign * pol_sign + Bth * Bth) / HMF_Mag

def eval_sinZeta(pol_sign: ArrayLike, sign_asun: ArrayLike, Bth: ArrayLike) -> Array:
    return sign_asun * (Bth / jnp.sqrt(pol_sign * pol_sign + Bth * Bth))

def eval_cosZeta(pol_sign: ArrayLike, sign_asun: ArrayLike, Bth: ArrayLike) -> Array:
    return sign_asun * (pol_sign / jnp.sqrt(pol_sign * pol_sign + Bth * Bth))

# ----------------------------------------------------------------
#  Derivatives of B-field functions
# ----------------------------------------------------------------
def eval_dBth_dr(state: PropagationState, is_polar_region: ArrayLike) -> Array:
    return jnp.where(is_polar_region, -delta_m / (R_helio * jnp.sin(state.th)), 0.0)

def eval_dBph_dr(state: PropagationState, pol_sign: ArrayLike, V_SW: ArrayLike) -> Array:
    return pol_sign * ((state.r - 2.0 * R_helio) * omega * jnp.sin(state.th)) / (state.r * V_SW)

def eval_dBth_dth(state: PropagationState,is_polar_region: ArrayLike) -> Array:
    val = state.r * delta_m / (R_helio * jnp.sin(state.th) * jnp.sin(state.th)) * (-jnp.cos(state.th))
    return jnp.where(is_polar_region, val, 0.0)

def eval_dBph_dth(state: PropagationState, pol_sign: ArrayLike, V_SW: ArrayLike, dV_SWdth: ArrayLike, DelDirac: ArrayLike) -> Array:
    num = -(state.r - R_helio) * omega * (
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

def eval_dBMag_dr(state:PropagationState, pol_sign: ArrayLike, Bth: ArrayLike, 
                  Bph: ArrayLike, dBth_dr: ArrayLike, dBph_dr: ArrayLike,
                  HMF_Mag: ArrayLike):
    num = (pol_sign * (-2.0 * pol_sign / state.r)
                 + Bth * dBth_dr
                 + Bph * dBph_dr)
    return num / HMF_Mag

def eval_DsinPsi_dr(sign_asun: ArrayLike, HMF_Mag: ArrayLike, Bph: ArrayLike,
                    dBph_dr: ArrayLike, dBMag_dr: ArrayLike) -> Array:
    return -sign_asun * ( dBph_dr * HMF_Mag - Bph * dBMag_dr) / (HMF_Mag * HMF_Mag)

def eval_DsinPsi_dtheta(sign_asun: ArrayLike, HMF_Mag: ArrayLike, Bph: ArrayLike,
                        dBph_dth: ArrayLike, dBMag_dth: ArrayLike) -> Array:
    return -sign_asun * (
        dBph_dth * HMF_Mag - Bph * dBMag_dth
    ) / (HMF_Mag * HMF_Mag)

def eval_dsqrtBR2BT2_dr(state:PropagationState, pol_sign: ArrayLike, Bth: ArrayLike,
                        sqrtBR2BT2: ArrayLike, dBth_dr: ArrayLike) -> Array:
    num = (pol_sign * (-2.0 * pol_sign / state.r) + Bth * dBth_dr)
    return num / sqrtBR2BT2

def eval_dsqrtBR2BT2_dth(pol_sign: ArrayLike, Bth: ArrayLike, sqrtBR2BT2: ArrayLike,
                         dBth_dth: ArrayLike, DelDirac: ArrayLike) -> Array:
    num = (pol_sign * (-2.0 * DelDirac) + Bth * dBth_dth)
    return num / sqrtBR2BT2

def eval_DcosPsi_dr(state:PropagationState, pol_sign: ArrayLike, Bth: ArrayLike, 
                    Bph: ArrayLike, sqrtBR2BT2: ArrayLike, dBth_dr: ArrayLike, 
                    dBph_dr: ArrayLike, HMF_Mag: ArrayLike) -> Array:
    num = Bph * (
        -pol_sign * pol_sign * dBph_dr
        + Bph * pol_sign * (-2.0 * pol_sign / state.r)
        + Bth * (-Bth * dBph_dr + Bph * dBth_dr)
    )
    return num / (sqrtBR2BT2 * (HMF_Mag**2) * HMF_Mag)

def eval_DcosPsi_dtheta(pol_sign: ArrayLike, Bth: ArrayLike, Bph: ArrayLike,
                        sqrtBR2BT2: ArrayLike, dBph_dth: ArrayLike, dBth_dth: ArrayLike,
                        HMF_Mag: ArrayLike, DelDirac: ArrayLike) -> Array:
    num = Bph * (
        -pol_sign * pol_sign * dBph_dth
        + Bph * pol_sign * (-2.0 * DelDirac)
        + Bth * (-Bth * dBph_dth + Bph * dBth_dth)
    )
    return num / (sqrtBR2BT2 * (HMF_Mag**2) * HMF_Mag)

def eval_DsinZeta_dr(sign_asun: ArrayLike, Bth: ArrayLike,
                     dBth_dr: ArrayLike, sqrtBR2BT2: ArrayLike,
                     dsqrtBR2BT2_dr: ArrayLike) -> Array:
    num = dBth_dr * sqrtBR2BT2 - Bth * dsqrtBR2BT2_dr
    return sign_asun * (num / (sqrtBR2BT2 * sqrtBR2BT2))

def eval_DsinZeta_dtheta(sign_asun: ArrayLike, Bth: ArrayLike,
                         dBth_dth: ArrayLike, sqrtBR2BT2: ArrayLike,
                         dsqrtBR2BT2_dth: ArrayLike) -> Array:
    num = dBth_dth * sqrtBR2BT2 - Bth * dsqrtBR2BT2_dth
    return sign_asun * (num / (sqrtBR2BT2 * sqrtBR2BT2))

def eval_DcosZeta_dr(state:PropagationState, sign_asun: ArrayLike, pol_sign: ArrayLike,
                     sqrtBR2BT2: ArrayLike, dsqrtBR2BT2_dr: ArrayLike) -> Array:
    num = ((-2.0 * pol_sign / state.r) * sqrtBR2BT2
                 - pol_sign * dsqrtBR2BT2_dr)
    return sign_asun * (num / (sqrtBR2BT2 * sqrtBR2BT2))

def eval_DcosZeta_dtheta(sign_asun: ArrayLike, pol_sign: ArrayLike,
                         sqrtBR2BT2: ArrayLike, dsqrtBR2BT2_dth: ArrayLike,
                         DelDirac: ArrayLike) -> Array:
    num = ((-2.0 * DelDirac) * sqrtBR2BT2
                 - pol_sign * dsqrtBR2BT2_dth)
    return sign_asun * (num / (sqrtBR2BT2 * sqrtBR2BT2))
