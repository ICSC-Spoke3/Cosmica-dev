import datetime
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

from extra.k0search.isotopes import find_ion_or_isotope
from extra.k0search.physics_utils import rig_to_en


def create_input_file(k0vals, h_par, exp_data, input_dir, sim_el, tot_npart_per_bin=1200,
                      n_heliosphere_regions=15, force_execute=False, debug=False):
    """
    Generates input files for the Cosmica simulation, validating parameters and handling file creation.
    
    param: k0vals (array): Array of K0 values to simulate.
    param: h_par (array): Array of heliosphere parameters used in the simulation.
    param: exp_data (array): Experimental data relevant to the simulation.
    param: input_path (str): Directory where input files will be stored.
    param: sim_el (list): Simulation metadata including name, ions, dates, and positions.
    param: tot_npart_per_bin (int): Total number of particles per bin (default=1200).
    param: n_heliosphere_regions (int): Number of heliosphere regions considered (default=15).
    param: force_execute (bool): If True, overwrites existing files (default=False).
    param: debug (bool): If True, enables debug output (default=False).
    return: A tuple containing the list of input file paths and the corresponding output file names.
    """

    # Unpack simulation metadata
    sim_name, ions, file_name, init_date, end_date, rad, lat, lon = sim_el

    # Ensure dates are ints
    init_date, end_date = int(init_date), int(end_date)

    # Make coordinates lists if not already and convert to float
    rad, lat, lon = map(lambda x: list(map(float, x)),
                        map(lambda x: x if isinstance(x, list) else [x], (rad, lat, lon)))

    # Parse ion names into a list
    ions = [ion.strip() for ion in ions.split(',')]

    # Check whether the simulation uses kinetic energy (tko) or rigidity
    tko = "Rigidity" not in file_name

    # Extract start and end dates from the heliosphere parameters
    cr_ini, cr_end = h_par[:, 0], h_par[:, 1]

    # Validate integration dates
    if int(end_date) > int(cr_end[0]):
        print(f"WARNING: End date {end_date} exceeds the range of available parameters {int(cr_end[0])}.")
        return []
    if int(init_date) < int(cr_ini[-n_heliosphere_regions]):
        print(
            f"WARNING: Start date {init_date} is earlier than the first available parameter {int(cr_ini[-n_heliosphere_regions])}.")
        return []

    # Generate the list of CRs (Cosmic Rays) and parameters to simulate
    cr_ord = np.arange(len(cr_ini))
    mask1 = (cr_ini <= end_date) & (cr_end > init_date)  # if between start and end
    mask2 = (cr_ini <= end_date) & (
            cr_end[cr_ord - (n_heliosphere_regions - 1)] > init_date)  # if between (start - num regions) and end
    mask3 = np.append((cr_ord - (n_heliosphere_regions - 1) >= 0) & (
            cr_ini[cr_ord - (n_heliosphere_regions - 1)] < init_date), 1)  # if before (start - num regions)
    mask3 = cr_ord <= np.argwhere(mask3)[0, 0]  # remove all after first True
    cr_list = cr_ini[mask1 & mask3]
    cr_list_param = cr_ini[mask2 & mask3]

    if not len(cr_list):
        print("WARNING: No valid CRs found for simulation.")
        return []

    # Expand position vectors if they are scalar values
    np_rad = np.full(len(cr_list), rad[0]) if len(rad) == 1 else np.array(rad)
    np_lat = np.full(len(cr_list), np.radians(90. - lat[0])) if len(lat) == 1 else np.radians(90. - np.array(lat))
    np_lon = np.full(len(cr_list), np.radians(lon[0])) if len(lon) == 1 else np.radians(np.array(lon))

    # Initialize dict to store file names
    sims_dict = defaultdict(list)

    # Loop over K0 values to generate input files
    for k0val in k0vals:
        ions_str = '' if len(ions) == 1 else '-'.join(ions) + '_'
        k0val_str = f"{k0val:.6e}".replace('.', "_")  # Convert K0 value to a formatted string
        tk0_str = '_TKO' if tko else ''
        sim_meta_str = ions_str + k0val_str + tk0_str

        # Loop over ions to generate input files for each isotope
        for ion in ions:
            isotopes = find_ion_or_isotope(ion)
            if not isotopes:
                print(f"WARNING: {ion} not found in isotopes dictionary.")
                continue

            # Loop through each isotope to construct simulation files
            for isotope in isotopes:
                # Create a unique simulation name
                sim_name = f"{isotope[3]}_{sim_meta_str}_{cr_list[-1]:.0f}_{cr_list[0]:.0f}_r{rad[0] * 100:05.0f}_lat{lat[0] * 100:05.0f}"
                input_file_path = pjoin(input_dir, f"{sim_name}.txt")

                # Check if the file already exists or needs to be recreated
                if not os.path.exists(input_file_path) or force_execute:
                    # Open the file for writing
                    with open(input_file_path, 'w') as f:
                        # Write metadata and simulation details
                        f.write(f"# File generated on {datetime.date.today()}\n")
                        f.write(f"OutputFilename: {sim_name}\n")
                        f.write(f"Particle_NucleonRestMass: {isotope[2]}\n")
                        f.write(f"Particle_MassNumber: {isotope[1]}\n")
                        f.write(f"Particle_Charge: {isotope[0]}\n")

                        # Write energy bins, converting if necessary
                        tcentr = exp_data[:, 0] if tko else rig_to_en(exp_data[:, 0], isotope[1], isotope[0])
                        tcentr_str = ','.join(f"{x:.3e}" for x in tcentr)
                        if len(tcentr_str) >= 2000:
                            raise ValueError("Energy inputs exceed allowed 2000 characters.")
                        f.write(f"Tcentr: {tcentr_str}\n")

                        # Write source position details
                        f.write(f"SourcePos_theta: {','.join(f'{x:.5f}' for x in np_lat)}\n")
                        f.write(f"SourcePos_phi: {','.join(f'{x:.5f}' for x in np_lon)}\n")
                        f.write(f"SourcePos_r: {','.join(f'{x:.5f}' for x in np_rad)}\n")

                        # Write particle generation and heliosphere parameters
                        f.write(f"Npart: {tot_npart_per_bin}\n")
                        f.write(f"Nregions: {n_heliosphere_regions}\n")

                        # Add heliospheric parameters for each CR period
                        for hp in h_par:
                            if hp[0] in cr_list_param:
                                f.write(
                                    "HeliosphericParameters: {:.6e}, {:.3f}, {:.2f}, {:.2f}, {:.3f}, {:.3f}, {:.0f}, {:.0f}, {:.3f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(
                                        k0val if np.where(cr_list_param == hp[0])[0].item() == 0 else 0.,
                                        *hp[[2, 3, 4, 12, 6, 7, 8, 11, 13, 14, 15, 16]]
                                    ))

                        # Add heliosheat parameters for each CR period
                        for i, hp in enumerate(h_par):
                            if hp[0] in cr_list:
                                f.write("HeliosheatParameters: {:.5e}, {:.2f}\n".format(
                                    3.e-05, h_par[i + n_heliosphere_regions - 1, 3]
                                ))
                        f.close()

                    # Append file paths and names to the list
                    sims_dict[(ion, k0val)].append(sim_name)

    return dict(sims_dict)


def run_cosmica(cosmica_path: str, sim_name: str, input_dir: str, output_dir: str, verbosity=0, debug=False):
    """
    Executes cosmica simulation.
    :param cosmica_path: Path to cosmica executable.
    :param sim_name: Path to input file.
    :param input_dir: Directory where the output will be saved.
    :param output_dir: Directory where the output will be saved.
    :param verbosity: Verbosity level for the simulation.
    :param debug: Enable debug output if True.
    :return: Return code and output of the simulation.
    """

    verbosity_str = ('-' + 'v' * verbosity) if verbosity else ''
    command = f"cd {output_dir} && {cosmica_path} {verbosity_str} -i {pjoin(input_dir, sim_name)}.txt > {sim_name}.log 2>&1"
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
        return result.returncode, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error during simulation execution for {sim_name}: {e}")
        if debug:
            print(e.output)
        return e.returncode, e.output


def submit_sims(sim_el, cosmica_path, results_path, k0_array, h_par, exp_data, max_workers=1, debug=False):
    """
    Creates simulations and executes them locally, generating input and output files.
    
    param: sim_el (list): Simulation parameters including simulation name, ions, etc.
    param: cosmica_path (str): Path to cosmica executable.
    param: results_path (str): Path to store results.
    param: k0_array (list): Array of K0 values to simulate.
    param: h_par (list): Parameters for the simulation.
    param: exp_data (dict): Experimental data for the simulation.
    param: debug (bool): Enable debug output if True.
    return: Path to the input directory and list of output file names.
    """

    sim_name, ions, file_name, init_date, final_date, radius, latitude, longitude = sim_el
    ions_str = '-'.join(i.strip() for i in ions.split(','))

    sim_dir = pjoin(results_path, f'{sim_name}_{ions_str}')
    input_dir = pjoin(sim_dir, 'inputs')
    output_dir = pjoin(sim_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    # Generate input and output file names for the simulation
    sims_dict = create_input_file(
        k0_array, h_par, exp_data, input_dir, sim_el, force_execute=True, debug=debug
    )

    # return sims_dict, output_dir  # TODO: REMOVE

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        job = lambda f: run_cosmica(cosmica_path, f, input_dir, output_dir, verbosity=2)
        futures = [executor.submit(job, f) for ff in sims_dict.values() for f in ff]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass  # Wait for all to finish

    return sims_dict, output_dir
