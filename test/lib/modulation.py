import numpy as np

from .files_utils import get_lis
from .isotopes import find_isotope
from .physics_utils import lin_log_interpolation, rig_to_en, en_to_rig, d_rig_to_en


def evaluate_spectra(RawMatrixFile, LIS, A, Z):
    lis_en, lis_flux_en = LIS

    input_rig = np.asarray([a for a in RawMatrixFile['InputEnergy']])
    n_particles = np.asarray([a for a in RawMatrixFile['NGeneratedParticle']])
    outer_rig = RawMatrixFile['OuterEnergy']
    boundary_distribution = RawMatrixFile['BoundaryDistribution']

    assert outer_rig.dtype == 'object'

    lis_rig = en_to_rig(lis_en, A, Z)
    lis_flux_en_in = lin_log_interpolation(lis_rig, lis_flux_en, input_rig)

    un_norm_flux = np.zeros(len(input_rig))
    for index_rig in range(len(input_rig)):
        lis_flux_en_out = lin_log_interpolation(lis_rig, lis_flux_en, outer_rig[index_rig])

        for outer_rig_bin, boundary_bin, lis_flux_bin in zip(outer_rig[index_rig], boundary_distribution[index_rig],
                                                             lis_flux_en_out):
            un_norm_flux[index_rig] += boundary_bin * lis_flux_bin / outer_rig_bin ** 2

    conv_coeff = d_rig_to_en(rig_to_en(input_rig, A, Z), input_rig, A, Z)
    J_Mod = conv_coeff * [UnFlux / Npart * R ** 2 for R, UnFlux, Npart in zip(input_rig, un_norm_flux, n_particles)]
    lis_flux_rig_in = conv_coeff * lis_flux_en_in
    return input_rig.copy(), J_Mod.copy(), lis_flux_rig_in.copy()


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
        energy_binning, j_mod, j_lis = evaluate_spectra(outputs[isotope], lis_spectrum, a, z)

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
