from PyCosmica.structures import delta_m, R_helio, omega, PropagationState, PropagationConstantsItem
from PyCosmica.sde import Tensor3D, DiffusionTensor
from PyCosmica.utils import solar_wind_speeed
from jax.typing import ArrayLike
from jax import Array, lax, numpy as jnp


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

# ----------------------------------------------------------------
#  Diffusion tensor functions
# ----------------------------------------------------------------

def SquareRoot_DiffusionTerm(
    state: PropagationState, consts:PropagationConstantsItem,
    conv_diff: DiffusionTensor) -> Array:

    rr = jnp.sqrt(2.0 * conv_diff.rr)
    
    def in_heliosphere():
        sintheta = jnp.sin(state.th)
        KSym_tr = 2.0 * conv_diff.tr / state.r
        KSym_tt = 2.0 * conv_diff.tt / (state.r * state.r)
        KSym_pr = 2.0 * conv_diff.pr / (state.r * sintheta)
        KSym_pt = 2.0 * conv_diff.pt / (state.r * state.r * sintheta)
        KSym_pp = 2.0 * conv_diff.pp / (state.r * state.r * sintheta * sintheta)

        tr = KSym_tr / rr
        pr = KSym_pr / rr
        tt = jnp.sqrt(KSym_tt - tr * tr)
        pt = 1.0 / tt * (KSym_pt - tr * pr)
        pp = jnp.sqrt(KSym_pp - pr * pr - pt * pt)

        return Tensor3D(rr, tr, tt, pr, pt, pp)
        
    def out_heliosphere():
        return Tensor3D(rr, 0.0, 0.0, 0.0, 0.0, 0.0)
   
    return lax.cond(state.rad_zone < consts.N_regions, in_heliosphere, out_heliosphere)



def AdvectiveTerm_radius(v_drift_rad: ArrayLike,
                        conv_diff: DiffusionTensor,
                        state: PropagationState, consts: PropagationConstantsItem) -> Array:
    
    AdvTerm += -solar_wind_speeed(state.init_zone, state.rad_zone, state.r, state.th, state.phi)
    
    def in_heliosphere():
        tantheta = jnp.tan(state.th)
        AdvTerm += 2.0 * conv_diff.rr / state.r + conv_diff.DKrr_dr + conv_diff.tr / (state.r * tantheta) + DKtr_dt / state.r - v_drift_rad
        return AdvTerm
    
    def out_heliosphere():
        AdvTerm += 2.0 * conv_diff.rr / state.r + conv_diff.DKrr_dr
        return AdvTerm
    
    return lax.cond(state.rad_zone < consts.N_regions, in_heliosphere, out_heliosphere)



def AdvectiveTerm_theta (state: PropagationState, conv_diff: DiffusionTensor,
                         consts: PropagationConstantsItem, v_drift_th: ArrayLike) -> Array:
    def in_heliosphere():
        r2 = state.r * state.r
        tantheta = jnp.tan(state.th)
        AdvTerm = conv_diff.tr / r2 + conv_diff.tt / (tantheta * r2) + conv_diff.DKrt_dr / state.r + conv_diff.DKtt_dt / r2 - v_drift_th / state.r
        return AdvTerm
    
    def out_heliosphere():
        return 0.0
    
    return lax.cond(state.rad_zone < consts.N_regions, in_heliosphere, out_heliosphere)



def AdvectiveTerm_phi (state: PropagationState,  conv_diff: DiffusionTensor, 
                       consts: PropagationConstantsItem, v_drift_phi: ArrayLike) -> Array:
    def in_heliosphere():
        sintheta = jnp.sin(state.th)
        r2 = state.r * state.r
        AdvTerm = conv_diff.pr / (r2 * sintheta) + conv_diff.DKrp_dr / (state.r * sintheta) + conv_diff.DKtp_dt / (r2 * sintheta) + (-v_drift_phi / (state.r * sintheta))
        return AdvTerm
    
    def out_heliosphere():
        return 0.0
    
    return lax.cond(state.rad_zone < consts.N_regions, in_heliosphere, out_heliosphere)