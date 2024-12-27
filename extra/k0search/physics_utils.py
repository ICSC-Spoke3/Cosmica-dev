from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d

from extra.k0search.files_utils import get_lis
from extra.k0search.isotopes import ISOTOPES


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


def lin_log_interpolation(vx, vy, vx_new):
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    vx_new = np.asarray(vx_new)
    lvx, lvy = np.log10(vx), np.log10(vy)
    lvx_new = np.log10(vx_new)
    lvy_interp = interp1d(lvx, lvy, bounds_error=False, fill_value='extrapolate')(lvx_new)
    return 10 ** lvy_interp


def beta_eval(t, t0):
    tt = t + t0
    t2 = tt + t0
    return np.sqrt(t * t2) / tt


def spectra_backward_energy(modulation_matrix_dict_isotope, lis_isotope, t0):
    """
    Evaluate the modulated specra for a single isotope in case of SDE Monte Carlo in Rigidity
    :param modulation_matrix_dict_isotope:
    :param lis_isotope:
    :param t0:
    :return:
    """
    lis_isotope_tkin, lis_isotope_flux = lis_isotope
    boundary_distribution = modulation_matrix_dict_isotope['BoundaryDistribution']
    input_energy = np.asarray([a for a in modulation_matrix_dict_isotope['InputEnergy']])
    n_generated_particle = np.asarray([a for a in modulation_matrix_dict_isotope['NGeneratedParticle']])
    outer_energy = modulation_matrix_dict_isotope['OuterEnergy']

    # Legacy loading
    # boundary_distribution = modulation_matrix_dict_isotope['BounduaryDistribution']
    # input_energy = np.asarray([a for a in modulation_matrix_dict_isotope['InputEnergy']])
    # n_generated_particle = np.asarray([a for a in modulation_matrix_dict_isotope['NGeneratedPartcle']])
    # outer_energy = modulation_matrix_dict_isotope['OuterEnergy']

    lis_isotope_interp = lin_log_interpolation(lis_isotope_tkin, lis_isotope_flux, input_energy)

    is_object_array = outer_energy.dtype == 'object'

    un_norm_flux = np.zeros(len(input_energy))
    for indexTDet in range(len(input_energy)):
        if is_object_array:
            lis_isotope_flux_outer = lin_log_interpolation(lis_isotope_tkin, lis_isotope_flux, outer_energy[indexTDet])
            oenk = outer_energy[indexTDet]
        else:
            lis_isotope_flux_outer = lin_log_interpolation(lis_isotope_tkin, lis_isotope_flux, outer_energy)
            oenk = outer_energy

        for indexTLIS_isotope in range(len(lis_isotope_flux_outer)):
            lis_isotope_energy = oenk[indexTLIS_isotope]
            # Compute $\frac{J_{LIS}(T_j)}{\beta(T_j)}\cdot \exp(L_j)$
            un_norm_flux[indexTDet] += (
                    lis_isotope_flux_outer[indexTLIS_isotope] *
                    boundary_distribution[indexTDet][indexTLIS_isotope] /
                    beta_eval(lis_isotope_energy, t0)
            )

    # Compute $J_{mod}(T) = \frac{\beta(T)}{N_{ev}} \sum_{j=1}^{N_{ev}} \frac{J_{LIS}(T_j)}{\beta(T_j)}\cdot \exp(L_j)$
    j_mod = [beta_eval(t, t0) / n_part * un_flux
             for t, un_flux, n_part in zip(input_energy, un_norm_flux, n_generated_particle)]

    if input_energy[0] > input_energy[-1]:
        return input_energy[::-1], j_mod[::-1], lis_isotope_interp[::-1]
    else:
        return input_energy, j_mod, lis_isotope_interp


#TODO: refactor below

# ========= Flux Conveyyrsion from Tkin --> Rigi ===================
def Tkin2Rigi_FluxConversione(Xval, Spectra, MassNumber=1., Z=1.):
    Rigi = np.array([en_to_rig(T, MassNumber=MassNumber, Z=Z) for T in Xval])
    Flux = np.array([Flux * dT_dR(T=T, R=R, MassNumber=MassNumber, Z=Z) for T, R, Flux in zip(Xval, Rigi, Spectra)])
    return (Rigi, Flux)


# ========= Flux conversion factor from Rigidity to Tkin ==========
def dT_dR(T=1, R=1, MassNumber=1., Z=1.):
    MassNumber = float(MassNumber)
    Z = float(Z)
    T0 = 0.931494061
    if np.fabs(Z) == 1.:
        T0 = 0.938272046
    if MassNumber == 0.:
        T0 = 5.11e-4
        MassNumber = 1.
    return Z * Z / (MassNumber * MassNumber) * R / (T + T0)


def EvaluateModulation(Ion, IonLIS, Modulation_Matrix_dict, output_in_energy=True):
    """
        Evaluate the modulation of cosmic rays for a given ion species.
        Parameters:
        Ion (str): The ion species to evaluate.
        IonLIS (array-like): The local interstellar spectrum (LIS) for the ion species.
        Modulation_Matrix_dict (dict): A dictionary containing modulation matrices for different isotopes.
        output_in_energy (bool, optional): If True, the output is evaluated in kinetic energy. If False, convert to rigidity. Default is True.
        Returns:
        tuple: A tuple containing:
            - SimEnRig (array-like): The energy or rigidity binning.
            - SimFlux (array-like): The modulated flux.
            - SimLIS (array-like): The local interstellar spectrum (LIS) after modulation.
    """
    IsotopesList = ISOTOPES.get(Ion, [])
    for iIsot in range(len(IsotopesList)):
        Z, A, T0, Isotope = IsotopesList[iIsot]
        print(Isotope)

        # print(LIS)
        lis_spectrum = get_lis(IonLIS, Z, A)
        EnergyBinning, J_Mod, J_LIS = spectra_backward_energy(Modulation_Matrix_dict[Isotope], lis_spectrum, T0)
        if iIsot == 0:
            SimEnRig = np.copy(EnergyBinning)
            SimFlux = np.zeros_like(EnergyBinning)
            SimLIS = np.zeros_like(EnergyBinning)
        # print(J_LIS,J_Mod)
        if not output_in_energy:
            SimEnRig, J_Mod = Tkin2Rigi_FluxConversione(EnergyBinning, J_Mod, MassNumber=A, Z=Z)
            SimEnRig, J_LIS = Tkin2Rigi_FluxConversione(EnergyBinning, J_LIS, MassNumber=A, Z=Z)

        SimFlux += J_Mod
        SimLIS += J_LIS

    return (SimEnRig, SimFlux, SimLIS)
