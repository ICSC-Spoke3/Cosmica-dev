from jax import Array, lax, numpy as jnp

from PyCosmica.structures import *
from PyCosmica.utils import *


def diffusion_tensor_symmetric(state: PropagationState, const: PropagationConstantsItem,
                               w: Array) -> ConvectionDiffusionTensor:
    def in_heliosphere():
        Kpar, dKpar_dr, Kperp, dKperp_dr, Kperp2, dKperp2_dr = diffusion_tensor_hmf_frame(
            state, const, beta_R(state, const), w)
        is_polar_region = jnp.fabs(jnp.cos(state.th)) > cos_polar_zone

        pol_sign = lax.select((state.th - PI / 2.) > 0, -1, 1)
        sign_A_sun = lax.select(const.LIM.A_sun > 0, 1, -1)

        V_sw = solar_wind_speeed(state, const)

        B_th = eval_Bth(state, is_polar_region)
        B_ph = eval_Bph(state, pol_sign, V_sw)
        HMF_mag = eval_HMF_Mag(B_th, B_ph)

        sin_zeta = eval_sinZeta(pol_sign, sign_A_sun, B_th)
        cos_zeta = eval_cosZeta(pol_sign, sign_A_sun, B_th)
        sin_psi = eval_sinPsi(sign_A_sun, B_ph, HMF_mag)
        cos_psi = eval_cosPsi(pol_sign, B_th, HMF_mag)

        rr, tt, pp, tr, pr, pt = lax.select(
            is_polar_region,
            jnp.array([
                Kperp * sin_zeta ** 2 + cos_zeta ** 2 * (Kpar * cos_psi ** 2 + Kperp2 * sin_psi ** 2),
                Kperp * cos_zeta ** 2 + sin_zeta ** 2 * (Kpar * cos_psi ** 2 + Kperp2 * sin_psi ** 2),
                Kpar * sin_psi ** 2 + Kperp2 * cos_psi ** 2,
                sin_zeta * cos_zeta * (Kpar * cos_psi ** 2 + Kperp2 * sin_psi ** 2 - Kperp),
                -(Kpar - Kperp2) * sin_psi * cos_psi * cos_zeta,
                -(Kpar - Kperp2) * sin_psi * cos_psi * sin_zeta,
            ]),
            jnp.array([
                cos_zeta ** 2 * (Kpar * cos_psi ** 2 + Kperp2 * sin_psi ** 2),
                Kperp * cos_zeta ** 2,
                Kpar * sin_psi ** 2 + Kperp2 * cos_psi ** 2,
                0.,
                -(Kpar - Kperp2) * sin_psi * cos_psi * cos_zeta,
                0.,
            ]),
        )

        sqrt_BR2BT2 = eval_sqrtBR2BT2(pol_sign, B_th, B_ph)
        dB_th_dr = eval_dBth_dr(state, is_polar_region)
        dsqrt_BR2BT2_dr = eval_dsqrtBR2BT2_dr(state, pol_sign, B_th, sqrt_BR2BT2, dB_th_dr)
        del_dirac = jnp.astype(state.th != PI / 2., float)
        dB_th_dth = eval_dBth_dth(state, is_polar_region)

        dsqrt_BR2BT2_dth = eval_dsqrtBR2BT2_dth(pol_sign, B_th, sqrt_BR2BT2, dB_th_dth, del_dirac)
        dB_ph_dr = eval_dBph_dr(state, pol_sign, V_sw)
        dB_mag_dr = eval_dBMag_dr(state, pol_sign, B_th, B_ph, dB_th_dr, dB_ph_dr, HMF_mag)

        D_cos_zeta_dr = eval_DcosZeta_dr(state, sign_A_sun, pol_sign, sqrt_BR2BT2, dsqrt_BR2BT2_dr)
        D_cos_zeta_dth = eval_DcosZeta_dtheta(sign_A_sun, pol_sign, sqrt_BR2BT2, dsqrt_BR2BT2_dth, del_dirac)
        D_cos_psi_dr = eval_DcosPsi_dr(state, pol_sign, B_th, B_ph, sqrt_BR2BT2, dB_th_dr, dB_ph_dr, HMF_mag)
        D_sin_psi_dr = eval_DsinPsi_dr(sign_A_sun, HMF_mag, B_ph, dB_ph_dr, dB_mag_dr)

        def is_polar():
            dV_sw_dth = solar_wind_derivative(state, const)
            dB_ph_dth = eval_dBph_dth(state, pol_sign, V_sw, dV_sw_dth, del_dirac)
            dB_mag_dth = eval_dBMag_dth(pol_sign, del_dirac, B_th, dB_th_dth, B_ph, dB_ph_dth, HMF_mag)

            D_sin_zeta_dr = eval_DsinZeta_dr(sign_A_sun, B_th, dB_th_dr, sqrt_BR2BT2, dsqrt_BR2BT2_dr)
            D_sin_zeta_dth = eval_DsinZeta_dtheta(sign_A_sun, B_th, dB_th_dth, sqrt_BR2BT2, dsqrt_BR2BT2_dth)
            D_cos_psi_dth = eval_DcosPsi_dtheta(pol_sign, B_th, B_ph, sqrt_BR2BT2, dB_ph_dth, dB_th_dth, HMF_mag,
                                                del_dirac)
            D_sin_psi_dth = eval_DsinPsi_dtheta(sign_A_sun, HMF_mag, B_ph, dB_ph_dth, dB_mag_dth)

            return ConvectionDiffusionTensor(
                rr, tr, tt, pr, pt, pp,
                2. * cos_zeta * (
                        cos_psi ** 2 * Kpar + Kperp2 * sin_psi ** 2) * D_cos_zeta_dr + sin_zeta ** 2 * dKperp_dr + cos_zeta ** 2 * (
                        2. * cos_psi * Kpar * D_cos_psi_dr + cos_psi ** 2 * dKpar_dr + sin_psi * (
                        sin_psi * dKperp2_dr + 2. * Kperp2 * D_sin_psi_dr)) + 2. * Kperp * sin_zeta * D_sin_zeta_dr,
                2. * cos_zeta * Kperp * D_cos_zeta_dth + sin_zeta ** 2 * (
                        2. * cos_psi * Kpar * D_cos_psi_dth + 2. * sin_psi * Kperp2 * D_sin_psi_dth) + 2. * (
                        cos_psi ** 2 * Kpar + Kperp2 * sin_psi ** 2) * sin_zeta * D_sin_zeta_dth,
                (-Kperp + cos_psi ** 2 * Kpar + Kperp2 * sin_psi ** 2) * (
                        sin_zeta * D_cos_zeta_dr + cos_zeta * D_sin_zeta_dr) + cos_zeta * sin_zeta * (
                        2. * cos_psi * Kpar * D_cos_psi_dr + cos_psi ** 2 * dKpar_dr - dKperp_dr + sin_psi * (
                        sin_psi * dKperp2_dr + 2. * Kperp2 * D_sin_psi_dr)),
                (-Kperp + cos_psi ** 2 * Kpar + Kperp2 * sin_psi ** 2) * (
                        sin_zeta * D_cos_zeta_dth + cos_zeta * D_sin_zeta_dth) + cos_zeta * sin_zeta * (
                        2. * cos_psi * Kpar * D_cos_psi_dth + 2. * sin_psi * Kperp2 * D_sin_psi_dth),
                cos_zeta * (Kperp2 - Kpar) * sin_psi * D_cos_psi_dr + cos_psi * (
                        Kperp2 - Kpar) * sin_psi * D_cos_zeta_dr + cos_psi * cos_zeta * sin_psi * (
                        dKperp2_dr - dKpar_dr) + cos_psi * cos_zeta * (Kperp2 - Kpar) * D_sin_psi_dr,
                (Kperp2 - Kpar) * (sin_psi * sin_zeta * D_cos_psi_dth + cos_psi * (
                        sin_zeta * D_sin_psi_dth + sin_psi * D_sin_zeta_dth)),

            )

        def is_not_polar():
            return ConvectionDiffusionTensor(
                rr, tr, tt, pr, pt, pp,
                2. * cos_zeta * (cos_psi ** 2 * Kpar + Kperp2 * sin_psi ** 2) * D_cos_zeta_dr + cos_zeta ** 2 * (
                        2. * cos_psi * Kpar * D_cos_psi_dr + cos_psi ** 2 * dKpar_dr + sin_psi * (
                        sin_psi * dKperp2_dr + 2. * Kperp2 * D_sin_psi_dr)),
                2. * cos_zeta * Kperp * D_cos_zeta_dth,
                0.,
                0.,
                cos_zeta * (Kperp2 - Kpar) * sin_psi * D_cos_psi_dr + cos_psi * (
                        Kperp2 - Kpar) * sin_psi * D_cos_zeta_dr + cos_psi * cos_zeta * sin_psi * (
                        dKperp2_dr - dKpar_dr) + cos_psi * cos_zeta * (Kperp2 - Kpar) * D_sin_psi_dr,
                0.,
            )

        return lax.cond(is_polar_region, is_polar, is_not_polar)

    def in_heliosheat():
        rr, dKrr_dr = diffusion_coeff_heliosheat(state, const, beta_R(state, const))
        return ConvectionDiffusionTensor(rr, 0., 0., 0., 0., 0., dKrr_dr, 0., 0., 0., 0., 0.)

    return lax.cond(state.rad_zone < const.N_regions, in_heliosphere, in_heliosheat)
