from PyCosmica.structures import *
from PyCosmica.utils.heliosphere_model import boundary_scalar
from PyCosmica.utils.generic_math import smooth_transition


def rescale_to_effective_heliosphere(bound_real: HeliosphereBoundRadius,
                                     a_drift: Position3D) -> (HeliosphereBoundRadius, Position3D):
    R_ts_nose = 100.
    R_ts_tail = bound_real.R_ts_tail * R_ts_nose / bound_real.R_ts_nose
    R_hp_nose = R_ts_nose + bound_real.R_hp_nose - bound_real.R_ts_nose
    R_hp_tail = R_ts_tail + bound_real.R_hp_tail - bound_real.R_ts_tail

    HM_Rts_d = boundary_scalar(a_drift.th, a_drift.phi, R_ts_nose, R_ts_tail)
    RW_Rts_d = boundary_scalar(a_drift.th, a_drift.phi, bound_real.R_ts_nose, bound_real.R_ts_tail)

    part_r = (a_drift.r / RW_Rts_d * HM_Rts_d) if a_drift.r <= RW_Rts_d else HM_Rts_d + a_drift.r - RW_Rts_d

    return (
        HeliosphereBoundRadius(R_ts_nose, R_ts_tail, R_hp_nose, R_hp_tail),
        Position3D(part_r, a_drift.th, a_drift.phi)
    )


def k0_fit_ssn(p, solar_phase, ssn):
    """
    Fit k0 based on solar phase and sunspot number
    :param p: solar polarity of HMF
    :param solar_phase: 0=rising / 1=Declining phase of solar activity cycle
    :param ssn: sunspot number
    :return: k0, gauss_var
    """

    # If solar polarity is positive
    if p > 0.:
        # If solar phase is rising
        if solar_phase == 0:
            k0 = 0.0002743 - 2.11e-6 * ssn + 1.486e-8 * ssn * ssn - 3.863e-11 * ssn * ssn * ssn
            gauss_var = 0.1122
        # If solar phase is declining
        else:
            k0 = 0.0002787 - 1.66e-6 * ssn + 4.658e-9 * ssn * ssn - 6.673e-12 * ssn * ssn * ssn
            gauss_var = 0.1324
    # If solar polarity is negative
    else:
        # If solar phase is rising
        if solar_phase == 0:
            k0 = 0.0003059 - 2.51e-6 * ssn + 1.284e-8 * ssn * ssn - 2.838e-11 * ssn * ssn * ssn
            gauss_var = 0.1097
        # If solar phase is declining
        else:
            k0 = 0.0002876 - 3.715e-6 * ssn + 2.534e-8 * ssn * ssn - 5.689e-11 * ssn * ssn * ssn
            gauss_var = 0.14
    return k0, gauss_var


def k0_fit_nmc(nmc):
    """
    Fit k0 based on nmc (neutral sheet magnetic crossing)
    :param nmc: neutral sheet magnetic crossing
    :return: k0, gauss_var
    """

    return jnp.exp(-10.83 - 0.0041 * nmc + 4.52e-5 * nmc * nmc), 0.1045


def k0_corr_factor(p, q, solar_phase, tilt):
    """
    Correction factor to K0 for the Kparallel
    :param p: solar polarity of HMF
    :param q: signum of particle charge
    :param solar_phase: 0=rising, 1=Declining phase of solar activity cycle
    :param tilt: Tilt angle of neutral sheet (in degree)
    :return: the correction factor

    @authors: 2017 Stefano
    """

    k0_corr_maxv = 1.5  # Maximum value of the correction factor
    k0_corr_minv = 1.  # Minimum value of the correction factor
    k0_corr_p0_asc = 18.  # Tilt angle at which the correction factor is maximum during the ascending phase
    k0_corr_p1_asc = 40.  # Tilt angle at which the correction factor is minimum during the ascending phase
    k0_corr_p0_des = 5.  # Tilt angle at which the correction factor is maximum during the descending phase
    k0_corr_p1_des = 53.  # Tilt angle at which the correction factor is minimum during the descending phase
    k0_corr_maxv_neg = 0.7  # Maximum value of the correction factor for negative polarity
    k0_corr_p0_asc_neg = 5.8  # Tilt angle at which the correction factor is maximum during the ascending phase for negative polarity
    k0_corr_p1_asc_neg = 47.  # Tilt angle at which the correction factor is minimum during the ascending phase for negative polarity
    k0_corr_p0_des_neg = 5.8  # Tilt angle at which the correction factor is maximum during the descending phase for negative polarity
    k0_corr_p1_des_neg = 58.  # Tilt angle at which the correction factor is minimum during the descending phase for negative polarity

    # If q (signum of particle charge) is positive
    if q > 0:
        # If p (solar polarity of HMF) is positive
        if q * p > 0:
            # If solar phase is rising
            if solar_phase == 0:
                return smooth_transition(k0_corr_maxv, k0_corr_minv, k0_corr_p1_asc, k0_corr_p0_asc, tilt)
            # If solar phase is declining
            else:
                return smooth_transition(k0_corr_maxv, k0_corr_minv, k0_corr_p1_des, k0_corr_p0_des, tilt)
        # If p (solar polarity of HMF) is negative
        else:
            return 1
    # If q (signum of particle charge) is negative
    if q < 0:
        # If p (solar polarity of HMF) is positive
        if q * p > 0:
            # If solar phase is rising
            if solar_phase == 0:
                return smooth_transition(k0_corr_maxv, k0_corr_minv, k0_corr_p1_asc, k0_corr_p0_asc, tilt)
            # If solar phase is declining
            else:
                return smooth_transition(k0_corr_maxv, k0_corr_minv, k0_corr_p1_des, k0_corr_p0_des, tilt)
        # If p (solar polarity of HMF) is negative
        else:
            # If solar phase is rising
            if solar_phase == 0:
                return smooth_transition(k0_corr_maxv_neg, k0_corr_minv, k0_corr_p1_asc_neg, k0_corr_p0_asc_neg, tilt)
            # If solar phase is declining
            else:
                return smooth_transition(k0_corr_maxv_neg, k0_corr_minv, k0_corr_p1_des_neg, k0_corr_p0_des_neg, tilt)
    return 1


def eval_k0(is_high_activity_period, p, q, solar_phase, tilt, nmc, ssn):
    """
    Evaluate diffusion parameter from fitting procedures
    :param is_high_activity_period: if True, high activity period
    :param p: solar polarity of HMF
    :param q: signum of particle charge
    :param solar_phase: 0=rising, 1=Declining phase of solar activity cycle
    :param tilt: Tilt angle of neutral sheet (in degree)
    :param nmc: neutral sheet magnetic crossing
    :param ssn: sunspot number
    :return: k0, kerr

    @authors: 2022 Stefano
    """

    #   float3 output;
    # k0_paral is corrected by a correction factor
    k0cor = k0_corr_factor(p, q, solar_phase, tilt)
    # If high activity period and nmc (neutral sheet magnetic crossing) is greater than 0
    if is_high_activity_period and nmc > 0:
        k0, kerr = k0_fit_nmc(nmc)
    else:
        k0, kerr = k0_fit_ssn(p, solar_phase, ssn)
    return float(k0 * k0cor), float(k0), float(kerr)


def g_low_comp(solar_phase: int, polarity: int, tilt: float) -> float:
    """
    Evaluate g_low parameter (for Kparallel).

    :param solar_phase: 0 = rising, 1 = declining phase of solar activity cycle
    :param polarity: Polarity of the cycle
    :param tilt: Tilt angle of neutral sheet (in degrees)
    :return: g_low value
    """

    # Assign values based on Polarity
    if polarity > 0:
        max_value_of_g_low = 0.6
        cab_trans_point_des = 45
        cab_smoothness_des = 5.0
        cab_trans_point_asc = 60
        cab_smoothness_asc = 9.0
    else:
        max_value_of_g_low = 0.5
        cab_trans_point_des = 45
        cab_smoothness_des = 10.0
        cab_trans_point_asc = 60
        cab_smoothness_asc = 9.0

    # Compute g_low based on SolarPhase
    if solar_phase == 1:
        g_low = max_value_of_g_low * smooth_transition(1, 0, cab_trans_point_des, cab_smoothness_des, tilt)
    else:
        g_low = max_value_of_g_low * smooth_transition(1, 0, cab_trans_point_asc, cab_smoothness_asc, tilt)

    return g_low


def r_const_comp(solar_phase: int, polarity: int, tilt: float) -> float:
    """
    Evaluate rconst parameter (for Kparallel).

    :param solar_phase: 0 = rising, 1 = declining phase of solar activity cycle
    :param polarity: Magnetic polarity (+1 or -1)
    :param tilt: Tilt angle of neutral sheet (in degrees)
    :return: rconst value
    """

    # Assign transition points and smoothness based on polarity
    if polarity > 0:
        r_const_trans_point_des = 45
        r_const_smoothness_des = 5.0
        r_const_trans_point_asc = 60
        r_const_smoothness_asc = 9.0
    else:
        r_const_trans_point_des = 45
        r_const_smoothness_des = 10.0
        r_const_trans_point_asc = 60
        r_const_smoothness_asc = 9.0

    # Compute rconst using SmoothTransition function
    if solar_phase == 1:
        rconst_value = smooth_transition(4, 1, r_const_trans_point_des, r_const_smoothness_des, tilt)
    else:
        rconst_value = smooth_transition(4, 1, r_const_trans_point_asc, r_const_smoothness_asc, tilt)

    return rconst_value


def a_sum_comp(V0: float, B_earth: float, polarity: int, ):
    return float(float(polarity) * (AU_M ** 2) * B_earth * 1e-9 / jnp.sqrt(
        1. + ((omega * (1 - R_helio)) / (V0 / AU_KM)) * ((omega * (1 - R_helio)) / (V0 / AU_KM))))
