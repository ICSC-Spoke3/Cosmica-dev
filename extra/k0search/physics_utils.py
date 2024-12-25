from datetime import datetime

import numpy as np


## Magic physics functions!

def en_to_rig(t, mass_number=1., z=1.):
    """
    Convert energy to rigidity
    :param t: energy
    :param mass_number: mass number
    :param z: charge
    :return: rigidity
    """
    t0 = 0.931494061
    if np.fabs(z) == 1:
        t0 = 0.938272046
    if mass_number == 0:
        t0 = 5.11e-4
        mass_number = 1
    return mass_number / np.fabs(z) * np.sqrt(t * (t + 2. * t0))


def rig_to_en(r, mass_number=1., z=1.):
    """
    Convert rigidity to energy
    :param r: rigidity
    :param mass_number: mass number
    :param z: charge
    :return: energy
    """
    t0 = 0.931494061
    if np.fabs(z) == 1:
        t0 = 0.938272046
    if mass_number == 0:
        t0 = 5.11e-4
        mass_number = 1
    return np.sqrt((z * z) / (mass_number * mass_number) * (r * r) + (t0 * t0)) - t0


def smooth_transition(initial_val, final_val, center_of_transition, smoothness, x):
    """
    Smooth transition between InitialVal to FinalVal centered at CenterOfTransition as function of x
    If smoothness == 0 use a sharp transition (tanh), otherwise use a smooth transition
    :param initial_val: initial value
    :param final_val: final value
    :param center_of_transition: center of transition
    :param smoothness: smoothness
    :param x: x
    :return: the value
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
    k0_corr_maxv = 1.5              # Maximum value of the correction factor
    k0_corr_minv = 1.               # Minimum value of the correction factor
    k0_corr_p0_asc = 18.            # Tilt angle at which the correction factor is maximum during the ascending phase
    k0_corr_p1_asc = 40.            # Tilt angle at which the correction factor is minimum during the ascending phase
    k0_corr_p0_des = 5.             # Tilt angle at which the correction factor is maximum during the descending phase    
    k0_corr_p1_des = 53.            # Tilt angle at which the correction factor is minimum during the descending phase
    k0_corr_maxv_neg = 0.7          # Maximum value of the correction factor for negative polarity
    k0_corr_p0_asc_neg = 5.8        # Tilt angle at which the correction factor is maximum during the ascending phase for negative polarity
    k0_corr_p1_asc_neg = 47.        # Tilt angle at which the correction factor is minimum during the ascending phase for negative polarity
    k0_corr_p0_des_neg = 5.8        # Tilt angle at which the correction factor is maximum during the descending phase for negative polarity
    k0_corr_p1_des_neg = 58.        # Tilt angle at which the correction factor is minimum during the descending phase for negative polarity

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


def initialize_output_dict(sim_el: list, exp_data: np.ndarray, output_dict: dict, k0_ref: float, k0_ref_err: float,
                           debug=False):
    """
    Initialize the output dictionary with experimental data
    :param sim_el: the simulation element
    :param exp_data: the experimental data
    :param output_dict: the output dictionary
    :param k0_ref: the reference k0
    :param k0_ref_err: the reference k0 error
    :param debug: if True, more verbose output will be printed
    :return: None
    """

    ions = sim_el[1].strip()
    init_time = datetime.strptime(sim_el[3], '%Y%m%d').date()
    if debug:
        print(f"Processing simulation for ion: {ions}, date: {init_time}")

    for t, f, er_inf, er_sup in exp_data:
        if debug:
            print(f"Processing rigidity value: {t}")

        # Initialize the output dictionary
        if t not in output_dict[ions]:
            output_dict[ions][t] = {}
            if debug:
                print(f"Initialized OUTPUT_DICT for ion: {ions}, rigidity: {t}")

        # Initialize the output dictionary for the current ion and rigidity value
        output_dict[ions][t][init_time] = {
            "diffBest": 9e99,
            "K0best": 0,
            "K0Min": 9e99,
            "K0Max": 0,
            "Fluxbest": 0,
            "FluxMin": 9e99,
            "FluxMax": 0,
            "fExp": f,
            "ErExp_inf": er_inf,
            "ErExp_sup": er_sup,
            "K0ref": k0_ref,
            "K0Err_ref": k0_ref_err * k0_ref,
        }
        if debug:
            print(f"Initialized data for ion: {ions}, rigidity: {t}, date: {init_time}")

    if debug:
        print("Final OUTPUT_DICT:")
        for ion, data in output_dict.items():
            print(f"Ion: {ion}, Data: {data}")
