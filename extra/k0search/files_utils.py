import pickle

import numpy as np
from os.path import join as pjoin

import astropy.io.fits as pyfits


def load_simulation_list(list_path: str, output_dict: dict, debug=False):
    """
    Load the list of simulations
    :param list_path: path to list file
    :param output_dict: dictionary to store the simulation results
    :param debug: if True, more verbose output will be printed
    :return: list of simulations
    """

    # Counters for excluded simulations (Added by me)
    excluded_sims_filter = 0
    excluded_sims_type = 0

    sim_list = []
    if debug:
        print(f"Loading simulations for: {list_path}")

    for line in open(list_path).readlines():
        if line.startswith("#"):
            continue

        single_sim = [x.strip() for x in line.replace("\t", "").split("|")[:8]]
        if debug:
            print(f"Parsed simulation: {single_sim}")

        # if debug is true, it excludes all files obtaining a final empty list, so I inibited it (MG)
        # if debug:
        # if False:
        #     if (float(single_sim[3]) < 20180101 or float(single_sim[3]) > 20180102):
        #         excluded_sims_filter += 1
        #     continue

        if "Electron" in single_sim[1] or "Positron" in single_sim[1]:
            if debug:
                excluded_sims_type += 1
            continue

        print(f"Adding simulation: {single_sim}")
        sim_list.append(single_sim)

        ions = single_sim[1].strip()
        if ions not in output_dict:
            output_dict[ions] = {}
            if debug:
                print(f"Added new ion type to OUTPUT_DICT: {ions}")

    if debug:
        print(f"Loaded simulation list:")
        for sim in sim_list:
            print(sim)
        print(f"Excluded simulations due to filter: {excluded_sims_filter}")
        print(f"Excluded simulations due to particle type: {excluded_sims_type}")

    return sim_list


def load_heliospheric_parameters(pastpar_path: str, frcpar_path: str, debug=False):
    """
    Load the Heliospheric parameters files
    :param pastpar_path: all parameters file
    :param frcpar_path: frcst parameters file
    :param debug: if True, more verbose output will be printed
    :return: matrix of parameters
    """

    h_par = np.loadtxt(pastpar_path)  # carrington rotations in decreasing order of time (most recent to least)

    frc_heliospheric_parameters = np.loadtxt(frcpar_path)
    h_par = np.append(frc_heliospheric_parameters, h_par, axis=0)
    if debug:
        print(" ----- HeliosphericParameters loaded ----")
    return h_par


def load_experimental_data(exp_path: str, file: str, rig_range=(3, 11), debug=False):
    """
    Load the experimental data and filter in rigidity range
    :param exp_path: directory of experimental data
    :param file: name of file
    :param rig_range: range of rigidity to filter in
    :param debug: if True, more verbose output will be printed
    :return: matrix of experimental data (rig, flux, flux_inf, flux_sup)
    """
    rig_low, rig_high = rig_range
    fname = pjoin(exp_path, file)
    exp_data = np.loadtxt(fname, usecols=(0, 1, 2, 3))
    if debug:
        print(f'Loaded experimental data from {fname}: {exp_data.shape}')
    exp_data = exp_data[(rig_low <= exp_data[:, 0]) & (exp_data[:, 0] <= rig_high)]
    if debug:
        print(f'Filtered experimental data: {exp_data.shape}')
    return exp_data


# TODO: refactor/rewrite
def load_lis(input_lis='../LIS/ProtonLIS_ApJ28nuclei.gz'):
    """Load LIS from a fits file
    InputLISFile: string, path to the fits file
    Returns: dictionary, containing the LIS for all species
        ParticleFlux[Z][A][K] contains the particle flux for all considered species
        galprop convention wants that for same combination of Z,A,K
        firsts are secondaries, latter Primary
    """
    # --------- Load LIS
    ERsun = 8.33
    """Open Fits File and Store particle flux in a dictionary 'ParticleFlux'
    We store it as a dictionary containing a dictionary, containing an array, where the first key is Z, the second key is A and then we have primaries, secondaries in an array
    - The option GALPROPInput select if LIS is a galprop fits file or a generic txt file
    """

    galdefid = input_lis
    hdulist = pyfits.open(galdefid)  # open fits file
    # hdulist.info()
    data = hdulist[0].data  # assign data structure di data
    # Find out which indices to interpolate over for Rsun
    Rsun = ERsun  # Earth position in the Galaxy
    R = (np.arange(int(hdulist[0].header["NAXIS1"]))) * hdulist[0].header["CDELT1"] + hdulist[0].header["CRVAL1"]
    inds = []
    weights = []
    if (R[0] > Rsun):
        inds.append(0)
        weights.append(1)
    elif (R[-1] <= Rsun):
        inds.append(-1)
        weights.append(1)
    else:
        for i in range(len(R) - 1):
            if (R[i] <= Rsun and Rsun < R[i + 1]):
                inds.append(i)
                inds.append(i + 1)
                weights.append((R[i + 1] - Rsun) / (R[i + 1] - R[i]))
                weights.append((Rsun - R[i]) / (R[i + 1] - R[i]))
                break

    # print("DEBUGLINE:: R=",R)
    # print("DEBUGLINE:: weights=",weights)
    # print("DEBUGLINE:: inds=",inds)

    # Calculate the energy for the spectral points.. note that Energy is in MeV
    energy = 10 ** (float(hdulist[0].header["CRVAL3"]) + np.arange(int(hdulist[0].header["NAXIS3"])) * float(
        hdulist[0].header["CDELT3"]))

    # Parse the header, looking for Nuclei definitions
    ParticleFlux = {}

    Nnuclei = hdulist[0].header["NAXIS4"]
    for i in range(1, Nnuclei + 1):
        id = "%03d" % i
        Z = int(hdulist[0].header["NUCZ" + id])
        A = int(hdulist[0].header["NUCA" + id])
        K = int(hdulist[0].header["NUCK" + id])

        # print("id=%s Z=%d A=%d K=%d"%(id,Z,A,K))
        # Add the data to the ParticleFlux dictionary
        if Z not in ParticleFlux:
            ParticleFlux[Z] = {}
        if A not in ParticleFlux[Z]:
            ParticleFlux[Z][A] = {}
        if K not in ParticleFlux[Z][A]:
            ParticleFlux[Z][A][K] = []
            # data structure
            #    - Particle type, identified by "id", the header allows to identify which particle is
            #    |  - Energy Axis, ":" takes all elements
            #    |  | - not used
            #    |  | |  - distance from Galaxy center: inds is a list of position nearest to Earth position (Rsun)
            #    |  | |  |
        d = ((data[i - 1, :, 0, inds].swapaxes(0, 1)) * np.array(weights)).sum(
            axis=1)  # real solution is interpolation between the nearest solution to Earh position in the Galaxy (inds)
        ParticleFlux[Z][A][K].append(1e7 * d / energy ** 2)  # 1e7 is conversion from [cm^2 MeV]^-1 --> [m^2 GeV]^-1
        # print (Z,A,K)
        # print ParticleFlux[Z][A][K]

    ## ParticleFlux[Z][A][K] contains the particle flux for all considered species  galprop convention wants that for same combiantion of Z,A,K firsts are secondaries, latter Primary
    energy = energy / 1e3  # convert energy scale from MeV/n to GeV/n
    # if A>1:
    #  energy = energy/float(A)
    hdulist.close()
    # LISSpectra = [ 0 for T in energy]
    LIS_Tkin = energy
    LIS_Flux = ParticleFlux
    return (LIS_Tkin, LIS_Flux)


def load_simulation_output(file_name, debug=False):
    """
    Load a simulation output file histograms
    :param file_name: path to file
    :param debug:
    :return:
    """
    input_energy = []  # Energy Simulated
    n_registered_particle = []  # number of simulated energy per input bin
    n_bins_outer_energy = []  # number of bins used for the output distribution
    outer_energy = []  # Bin center of output distribution
    energy_distribution_at_boundary = []  # Energy distribution at heliosphere boundary
    warning_list = []

    with open(file_name) as f:
        lines = list(filter(lambda l: not l.startswith("#"), f.readlines()))
    n_bins = int(lines[0])

    # Read bins specifications
    for spec in map(str.split, lines[1::2]):
        e_gen, n_part_gen, n_part_reg, n_bin_out, bin_low, bin_amp = spec[:6]
        input_energy.append(float(e_gen))
        if debug and n_part_gen != n_part_reg:
            warning_list.append(
                f'WARNING: registered particle for Energy {e_gen} ({n_part_reg}) is different from injected ({n_part_gen})')
        n_registered_particle.append(int(n_part_reg))
        n_bins_outer_energy.append(int(n_bin_out))
        bin_low, bin_amp = map(float, (bin_low, bin_amp))
        bins = bin_low + np.arange(int(n_bin_out)) * bin_amp
        outer_energy.append((10 ** bins + 10 ** (bins + bin_amp)) / 2)

    # Read bins distributions
    for dist, ie, nboe in zip(map(str.split, lines[2::2]), input_energy, n_bins_outer_energy):
        if len(dist) != nboe:
            warning_list.append(
                f'WARNING: The number of saved bins for energy {ie} ({len(dist)}) is different from expected ({nboe})')
        energy_distribution_at_boundary.append(list(map(float, dist)))

    assert 2 * n_bins == len(lines) - 1 and n_bins == len(input_energy)

    return {
        'InputEnergy': np.asarray(outer_energy, object),
        'NGeneratedPartcle': np.asarray(n_registered_particle, object),
        'OuterEnergy': np.asarray(outer_energy, object),
        'BounduaryDistribution': np.asarray(energy_distribution_at_boundary, object)
    }, warning_list
