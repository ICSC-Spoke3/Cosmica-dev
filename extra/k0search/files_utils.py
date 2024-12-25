import numpy as np
from os.path import join as pjoin


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
