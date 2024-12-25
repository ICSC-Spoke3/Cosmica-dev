from datetime import datetime

import numpy as np


## Magic physics functions!

def en_to_rig(t, mass_number=1., z=1.):
    t0 = 0.931494061
    if np.fabs(z) == 1:
        t0 = 0.938272046
    if mass_number == 0:
        t0 = 5.11e-4
        mass_number = 1
    return mass_number / np.fabs(z) * np.sqrt(t * (t + 2. * t0))


def rig_to_en(r, mass_number=1., z=1.):
    t0 = 0.931494061
    if np.fabs(z) == 1:
        t0 = 0.938272046
    if mass_number == 0:
        t0 = 5.11e-4
        mass_number = 1
    return np.sqrt((z * z) / (mass_number * mass_number) * (r * r) + (t0 * t0)) - t0


def smooth_transition(initial_val, final_val, center_of_transition, smoothness, x):
    # smooth transition between  InitialVal to FinalVal centered at CenterOfTransition as function of x
    # if smoothness== 0 use a sharp transition
    if smoothness == 0:
        return final_val if x >= center_of_transition else initial_val
    else:
        return (initial_val + final_val) / 2. - (initial_val - final_val) / 2. * np.tanh(
            (x - center_of_transition) / smoothness)


def k0_fit_ssn(p, solar_phase, ssn):
    if p > 0.:
        if solar_phase == 0:  # Rising
            k0 = 0.0002743 - 2.11e-6 * ssn + 1.486e-8 * ssn * ssn - 3.863e-11 * ssn * ssn * ssn
            gauss_var = 0.1122
        else:  # Declining
            k0 = 0.0002787 - 1.66e-6 * ssn + 4.658e-9 * ssn * ssn - 6.673e-12 * ssn * ssn * ssn
            gauss_var = 0.1324
    else:
        if solar_phase == 0:  # Rising
            k0 = 0.0003059 - 2.51e-6 * ssn + 1.284e-8 * ssn * ssn - 2.838e-11 * ssn * ssn * ssn
            gauss_var = 0.1097
        else:  # Declining
            k0 = 0.0002876 - 3.715e-6 * ssn + 2.534e-8 * ssn * ssn - 5.689e-11 * ssn * ssn * ssn
            gauss_var = 0.14
    return k0, gauss_var


def k0_fit_nmc(nmc):
    return np.exp(-10.83 - 0.0041 * nmc + 4.52e-5 * nmc * nmc), 0.1045


def k0_corr_factor(p, q, solar_phase, tilt):
    #   /*Authors: 2017 Stefano */
    #   /* * description: Correction factor to K0 for the Kparallel. This correction is introduced
    #                     to account for the fact that K0 is evaluated with a model not including particle drift.
    #                     Thus, the value need a correction once to be used in present model
    #       \param p            solar polarity of HMF
    #       \param q            signum of particle charge
    #       \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
    #       \param tilt         Tilt angle of neutral sheet (in degree)
    #   */
    k0_corr_maxv = 1.5
    k0_corr_minv = 1.
    k0_corr_p0_asc = 18.
    k0_corr_p1_asc = 40.
    k0_corr_p0_des = 5.
    k0_corr_p1_des = 53.
    k0_corr_maxv_neg = 0.7
    k0_corr_p0_asc_neg = 5.8
    k0_corr_p1_asc_neg = 47.
    k0_corr_p0_des_neg = 5.8
    k0_corr_p1_des_neg = 58.

    if q > 0:
        if q * p > 0:
            if solar_phase == 0:  # ascending
                return smooth_transition(k0_corr_maxv, k0_corr_minv, k0_corr_p1_asc, k0_corr_p0_asc, tilt)
            else:  # descending
                return smooth_transition(k0_corr_maxv, k0_corr_minv, k0_corr_p1_des, k0_corr_p0_des, tilt)
        else:
            return 1
    if q < 0:
        if q * p > 0:
            if solar_phase == 0:  # ascending
                return smooth_transition(k0_corr_maxv, k0_corr_minv, k0_corr_p1_asc, k0_corr_p0_asc, tilt)
            else:  # descending
                return smooth_transition(k0_corr_maxv, k0_corr_minv, k0_corr_p1_des, k0_corr_p0_des, tilt)
        else:
            if solar_phase == 0:  # ascending
                return smooth_transition(k0_corr_maxv_neg, k0_corr_minv, k0_corr_p1_asc_neg, k0_corr_p0_asc_neg, tilt)
            else:  # descending
                return smooth_transition(k0_corr_maxv_neg, k0_corr_minv, k0_corr_p1_des_neg, k0_corr_p0_des_neg, tilt)
    return 1


def eval_k0(is_high_activity_period, p, q, solar_phase, tilt, nmc, ssn):
    #   /*Authors: 2022 Stefano */
    #   /* * description: Evaluate diffusion parameter from fitting procedures.
    #       \param p            solar polarity of HMF
    #       \param q            signum of particle charge
    #       \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
    #       \param tilt         Tilt angle of neutral sheet (in degree)
    #       \return x = k0_paral
    #               y = k0_perp
    #               z = GaussVar
    #   */
    #   float3 output;
    k0cor = k0_corr_factor(p, q, solar_phase, tilt)  # ; // k0_paral is corrected by a correction factor
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
