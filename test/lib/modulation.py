import numpy as np

from .files_utils import get_lis
from .isotopes import find_isotope
from .physics_utils import lin_log_interpolation, rig_to_en, en_to_rig_flux, beta_eval


def evaluate_spectra(RawMatrixFile, LIS, T0, A, Z):
    lis_en, lis_flux = LIS

    input_rig = np.asarray([a for a in RawMatrixFile['InputEnergy']])
    n_particles = np.asarray([a for a in RawMatrixFile['NGeneratedParticle']])
    outer_rig = RawMatrixFile['OuterEnergy']
    boundary_distribution = RawMatrixFile['BoundaryDistribution']

    assert outer_rig.dtype == 'object'

    lis_flux_interp = lin_log_interpolation(lis_en, lis_flux, rig_to_en(input_rig, A, Z))

    un_norm_flux = np.zeros(len(input_rig))
    for index_rig in range(len(input_rig)):
        new_OuterEnRig = rig_to_en(np.asarray([a for a in outer_rig[index_rig]]), A, Z)
        _, OLIS = en_to_rig_flux(new_OuterEnRig, lin_log_interpolation(lis_en, lis_flux, new_OuterEnRig), A, Z)

        for indexTLIS in range(len(OLIS)):
            EnRigLIS = outer_rig[index_rig][indexTLIS]
            un_norm_flux[index_rig] += boundary_distribution[index_rig][indexTLIS] * OLIS[
                indexTLIS] / EnRigLIS ** 2 / beta_eval(rig_to_en(EnRigLIS, A, Z), T0)

    J_Mod = [UnFlux / Npart * beta_eval(rig_to_en(R, A, Z), T0) * R ** 2 for R, UnFlux, Npart in
             zip(input_rig, un_norm_flux, n_particles)]

    return input_rig.copy(), J_Mod.copy(), lis_flux_interp.copy()


def evaluate_spectra_multiple(outputs, ion_lis):
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
        energy_binning, j_mod, j_lis = evaluate_spectra(outputs[isotope], lis_spectrum, t0, a, z)

        if sim_en_rig is None:
            sim_en_rig = energy_binning
            sim_flux = np.zeros_like(energy_binning)
            sim_lis = np.zeros_like(energy_binning)

        sim_flux += j_mod
        sim_lis += j_lis

    return sim_en_rig, sim_flux, sim_lis

def evaluate_modulations(ion_lis, *results):
    def assign_or_assert(s, d):
        if d is None:
            return s
        else:
            assert np.allclose(s, d, rtol=1e-5), 'Mismatching axis'
            return d

    rig, lis, fluxes = None, None, []
    for result in results:
        rig_, flux_, lis_ = evaluate_spectra_multiple(result, ion_lis)
        rig = assign_or_assert(rig_, rig)
        fluxes.append(flux_)
        lis = assign_or_assert(lis_, lis)

    return np.c_[rig, lis, *fluxes]