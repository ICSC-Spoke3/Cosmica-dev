import numpy as np

from .files_utils import get_lis
from .isotopes import find_isotope
from .physics_utils import lin_log_interpolation, rig_to_en, en_to_rig_flux, beta_eval


def spectra(RawMatrixFile, LIS, T0, A, Z):
    LIS_Tkin, LIS_Flux = LIS

    BounduaryDistribution = RawMatrixFile['BoundaryDistribution']
    InputEnRig = np.asarray([a for a in RawMatrixFile['InputEnergy']])
    NGeneratedPartcle = np.asarray([a for a in RawMatrixFile['NGeneratedParticle']])
    OuterEnRig = RawMatrixFile['OuterEnergy']

    assert OuterEnRig.dtype == 'object'

    ILIS = lin_log_interpolation(LIS_Tkin, LIS_Flux, rig_to_en(InputEnRig, A, Z))
    # OuterEnRig = np.array([rig_to_en(x, A, Z) for x in OuterEnRig], dtype=object) \
    #     if is_object_array else rig_to_en(OuterEnRig, A, Z)

    UnNormFlux = np.zeros(len(InputEnRig))
    for indexTDet in range(len(InputEnRig)):
        new_OuterEnRig = rig_to_en(np.asarray([a for a in OuterEnRig[indexTDet]]), A, Z)
        _, OLIS = en_to_rig_flux(new_OuterEnRig, lin_log_interpolation(LIS_Tkin, LIS_Flux, new_OuterEnRig), A, Z)

        for indexTLIS in range(len(OLIS)):
            EnRigLIS = OuterEnRig[indexTDet][indexTLIS]
            UnNormFlux[indexTDet] += BounduaryDistribution[indexTDet][indexTLIS] * OLIS[
                indexTLIS] / EnRigLIS ** 2 / beta_eval(rig_to_en(EnRigLIS, A, Z), T0)

    J_Mod = [UnFlux / Npart * beta_eval(rig_to_en(R, A, Z), T0) * R ** 2 for R, UnFlux, Npart in
             zip(InputEnRig, UnNormFlux, NGeneratedPartcle)]

    if InputEnRig[0] > InputEnRig[-1]:
        EnRigBinning = np.array(InputEnRig[::-1])  # Energy or rigidity depending on the simulation output unit
        J_Mod = np.array(J_Mod[::-1])  # Energy or rigidity depending on the simulation output unit
        LIS = np.array(ILIS[::-1])  # Always in energy unit
    else:
        EnRigBinning = np.array(InputEnRig[:])  # Energy or rigidity depending on the simulation output unit
        J_Mod = np.array(J_Mod[:])  # Energy or rigidity depending on the simulation output unit
        LIS = np.array(ILIS[:])  # Always in energy unit

    return EnRigBinning, J_Mod, LIS, np.array(BounduaryDistribution)


def evaluate_modulation(outputs, ion_lis):
    """
    Evaluate the modulation of cosmic rays for a given ion species.
    :param ion:
    :param ion_lis:
    :param modulation_matrix:
    :param output_in_energy:
    :return:
    """

    isotopes_list = [find_isotope(iso) for iso in outputs.keys()]
    sim_en_rig, sim_flux, sim_lis = None, None, None

    for z, a, t0, isotope in isotopes_list:
        lis_spectrum = get_lis(ion_lis, z, a)
        energy_binning, j_mod, j_lis, _ = spectra(outputs[isotope], lis_spectrum, z, a, t0)
        # print('input', outputs[isotope]['InputEnergy'])
        # print('spectra', energy_binning)

        # If the first isotope, initialize the output
        if sim_en_rig is None:
            sim_en_rig = energy_binning
            sim_flux = np.zeros_like(energy_binning)
            sim_lis = np.zeros_like(energy_binning)

        # sim_en_rig, j_od = en_to_rig_flux(energy_binning, j_mod, a, z)
        # sim_en_rig, j_lis = en_to_rig_flux(energy_binning, j_lis, a, z)

        # Sum the fluxes and LIS
        sim_flux += j_mod
        sim_lis += j_lis

    return sim_en_rig, sim_flux, sim_lis
