import numpy as np
from scipy.interpolate import interp1d


## Magic physics functions!

def en_to_rig(en: np.ndarray, Z: int, A: int) -> np.ndarray:
    """
    Convert energy to rigidity
    :param en: energy
    :param Z: atomic number
    :param A: atomic mass
    :return: rigidity
    """
    T0 = 0.931494061
    if np.fabs(Z) == 1:
        T0 = 0.938272046
    if A == 0:
        T0 = 5.11e-4
        A = 1
    return A / np.fabs(Z) * np.sqrt(en * (en + 2. * T0))


def rig_to_en(rig: np.ndarray, Z: int, A: int) -> np.ndarray:
    """
    Convert rigidity to energy
    :param rig: rigidity
    :param Z: atomic number
    :param A: atomic mass
    :return: energy
    """
    T0 = 0.931494061
    if np.fabs(Z) == 1:
        T0 = 0.938272046
    if A == 0:
        T0 = 5.11e-4
        A = 1
    return np.sqrt((Z * Z) / (A * A) * (rig * rig) + (T0 * T0)) - T0


def d_rig_to_en(en: np.ndarray, rig: np.ndarray, Z: int, A: int):
    """
    Derivative of convert rigidity to energy
    :param en: energy
    :param rig: rigidity
    :param Z: atomic number
    :param A: atomic mass
    :return: energy
    """

    T0 = 0.931494061
    if np.fabs(Z) == 1:
        T0 = 0.938272046
    if A == 0:
        T0 = 5.11e-4
        A = 1
    return (Z * Z) / (A * A) * rig / (en + T0)


def smooth_transition(initial_val, final_val, center_of_transition, smoothness, x):
    """
    Smooth transition between InitialVal to FinalVal centered at CenterOfTransition as function of x
    If smoothness == 0 use a sharp transition (tanh), otherwise use a smooth transition
    :param initial_val: initial value
    :param final_val: final value
    :param center_of_transition: center of transition
    :param smoothness: smoothness
    :param x: x value
    :return: the transition value
    """

    if smoothness == 0:
        return final_val if x >= center_of_transition else initial_val
    else:
        return (initial_val + final_val) / 2. - (initial_val - final_val) / 2. * np.tanh(
            (x - center_of_transition) / smoothness)


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

    return np.exp(-10.83 - 0.0041 * nmc + 4.52e-5 * nmc * nmc), 0.1045


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
    return k0 * k0cor, kerr


def lin_log_interpolation(vx, vy, vx_new):
    """
    Linear-log interpolation
    :param vx: x values
    :param vy: y values
    :param vx_new: new x values
    :return: interpolated y values
    """

    vx = np.asarray(vx)
    vy = np.asarray(vy)
    vx_new = np.asarray(vx_new)
    lvx, lvy = np.log10(vx), np.log10(vy)
    lvx_new = np.log10(vx_new)
    lvy_interp = interp1d(lvx, lvy, bounds_error=False, fill_value='extrapolate')(lvx_new)
    return 10 ** lvy_interp


def beta_eval(t, t0):
    """
    Evaluate beta factor
    :param t: energy
    :param t0: energy offset
    :return: beta factor
    """

    tt = t + t0
    t2 = tt + t0
    return np.sqrt(t * t2) / tt


def rig_to_en_flux_factor(en: np.ndarray, rig: np.ndarray, Z: int, A: int) -> np.ndarray:
    """
    Convert rigidity to energy flux factor
    :param en: energy
    :param rig: rigidity
    :param Z: atomic number
    :param A: atomic mass
    :return: energy flux factor
    """
    T0 = 0.931494061
    if np.fabs(Z) == 1.:
        T0 = 0.938272046
    if A == 0.:
        T0 = 5.11e-4
        A = 1.
    return Z * Z / (A * A) * rig / (en + T0)


def en_to_rig_flux(en: np.ndarray, flux: np.ndarray, Z: int, A: int) -> (np.ndarray, np.ndarray):
    """
    Convert energy to rigidity flux
    :param en: energy values
    :param flux: spectra values
    :param A: mass number
    :param Z: charge
    :return: rigidity, flux
    """
    rig = en_to_rig(en, Z, A)
    flux = np.array([flux * rig_to_en_flux_factor(t, r, Z, A) for t, r, flux in zip(en, rig, flux)])
    return rig, flux
