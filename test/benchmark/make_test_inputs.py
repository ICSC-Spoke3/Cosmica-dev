import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from math import ceil

from lib.files_utils import load_simulation_list, load_heliospheric_parameters, load_experimental_data
from lib.isotopes import find_ion_or_isotope

yaml.Dumper.ignore_aliases = lambda *args: True


class InlineList(list):
    @staticmethod
    def inline_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(InlineList, InlineList.inline_list_representer)


def create_input_file(nparts, nk0, random_seed, sim_el, input_dir, h_par, rigidities, n_heliosphere_regions=15):
    """
    Generates input files for the Cosmica simulation, validating parameters and handling file creation.

    param: k0vals (array): Array of K0 values to simulate.
    param: h_par (array): Array of heliosphere parameters used in the simulation.
    param: rigidities (array): Experimental data relevant to the simulation.
    param: input_path (str): Directory where input files will be stored.
    param: sim_el (list): Simulation metadata including name, ions, dates, and positions.
    param: tot_npart_per_bin (int): Total number of particles per bin (default=1200).
    param: n_heliosphere_regions (int): Number of heliosphere regions considered (default=15).
    param: force_execute (bool): If True, overwrites existing files (default=False).
    param: debug (bool): If True, enables debug output (default=False).
    return: A tuple containing the list of input file paths and the corresponding output file names.
    """

    # Unpack simulation metadata
    sim_name, ion, file_name, init_date, end_date, rad, lat, lon = sim_el

    # Ensure dates are ints
    init_date, end_date = int(init_date), int(end_date)

    # Make coordinates lists if not already and convert to float
    rad, lat, lon = map(lambda x: list(map(float, x)),
                        map(lambda x: x.split(','), (rad, lat, lon)))

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

    isotopes = find_ion_or_isotope(ion)
    sim_name = f'{"_".join(iso[3] for iso in isotopes)}_{init_date}_{end_date}_{nparts}_{nk0}_{random_seed}'
    folder_name = input_dir / sim_name
    folder_name.mkdir(exist_ok=True)

    for ink0 in range(nk0):
        # Loop through each isotope to construct simulation files
        for isotope in isotopes:
            input_file_path = folder_name / f"{isotope[3]}_{ink0}.txt"
            output_name = f'{isotope[3]}_{init_date}_{end_date}_{nparts}_{nk0}_{random_seed}_{ink0}'

            # Open the file for writing
            with open(input_file_path, 'w') as f:
                # Write metadata and simulation details
                f.write(f"# File generated on {datetime.date.today()}\n")
                f.write(f"RandomSeed: {random_seed}\n")
                f.write(f"OutputFilename: {output_name}\n")
                f.write(f"Particle_NucleonRestMass: {isotope[2]}\n")
                f.write(f"Particle_MassNumber: {isotope[1]}\n")
                f.write(f"Particle_Charge: {isotope[0]}\n")

                # Write energy bins, converting if necessary
                tcentr = rigidities
                tcentr_str = ','.join(f"{x:.3e}" for x in tcentr)
                f.write(f"Tcentr: {tcentr_str}\n")

                # Write source position details
                f.write(f"SourcePos_theta: {','.join(f'{x:.5f}' for x in np_lat)}\n")
                f.write(f"SourcePos_phi: {','.join(f'{x:.5f}' for x in np_lon)}\n")
                f.write(f"SourcePos_r: {','.join(f'{x:.5f}' for x in np_rad)}\n")

                # Write particle generation and heliosphere parameters
                f.write(f"Npart: {nparts * len(rigidities)}\n")
                f.write(f"Nregions: {n_heliosphere_regions}\n")

                # Add heliospheric parameters for each CR period
                for hp in h_par:
                    if hp[0] in cr_list_param:
                        f.write(
                            "HeliosphericParameters: {:.6e}, {:.3f}, {:.2f}, {:.2f}, {:.3f}, {:.3f}, {:.0f}, {:.0f}, {:.3f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(
                                0.0 if np.where(cr_list_param == hp[0])[0].item() == 0 else 0.,
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
                sims_dict[(ion, 0.0)].append(sim_name)

            hpp = []
            for hp in h_par:
                if hp[0] in cr_list_param:
                    hpp.append(hp[[2, 3, 4, 12, 6, 7, 8, 11, 13, 14, 15, 16]].tolist())
            hpp = list(map(list, zip(*hpp)))

            hsp = []
            for i, hp in enumerate(h_par):
                if hp[0] in cr_list:
                    hsp.append((3.e-05, h_par[i + n_heliosphere_regions - 1, 3].tolist()))
            hsp = list(map(list, zip(*hsp)))

            yml = {
                'random_seed': random_seed,
                'output_path': sim_name,
                # 'energies': InlineList(rig_to_en(exp_data[:, 0], isotopes[0][1], isotopes[0][0]).tolist()),
                'rigidities': InlineList(rigidities),
                'isotopes': {nm.lower(): {
                    'nucleon_rest_mass': t0,
                    'mass_number': a,
                    'charge': z,
                } for z, a, t0, nm in isotopes},
                'sources': {
                    'r': InlineList(np_rad.tolist()),
                    'th': InlineList(np_lat.tolist()),
                    'phi': InlineList(np_lon.tolist()),
                },
                'relative_bin_amplitude': 0.00855,
                'n_particles': nparts,  # TODO: maybe * 10
                'n_regions': n_heliosphere_regions,
                'dynamic': {'heliosphere': {'k0': [InlineList([0.0] * len(hpp[0]))] * nk0}},
                'static': {
                    'heliosphere': {
                        k: InlineList(hpp[i])
                        for i, k in
                        enumerate(('ssn', 'v0', 'tilt_angle', 'smooth_tilt', 'b_field', 'polarity', 'solar_phase',
                                   'nmcr', 'ts_nose', 'ts_tail', 'hp_nose', 'hp_tail'))
                    },
                    'heliosheat': {
                        k: InlineList(hsp[i])
                        for i, k in enumerate(('k0', 'v0'))
                    }
                }
            }
            yml['n_particles'] = int(ceil(yml['n_particles'] / len(yml['sources']['r'])))
            with open(folder_name / f'{yml['output_path']}.yaml', 'w') as f:
                yaml.dump(yml, f, sort_keys=False, width=float("inf"))

    return dict(sims_dict)


def make_input_from_sim(data_dir, input_dir, sim_el):
    p_past_par = data_dir / 'heliospheric_parameters' / 'ParameterListALL_v12.txt'
    p_frct_par = data_dir / 'heliospheric_parameters' / 'Frcst_param.txt'
    p_exp = data_dir / 'experimental'

    h_par = load_heliospheric_parameters(p_past_par, p_frct_par)
    exp_data_file = next(p_exp.glob(f'*_{sim_el[3]}_*.dat'))
    exp = load_experimental_data(exp_data_file, (2, 3, 4, 5), rig_range=(0, 11))
    rigidities = exp[:, 0].tolist()

    for npart in (10, 100, 300, 500):
        for nk0 in (1, 10, 30, 50):
            for rnd in (42, 69, 123):
                create_input_file(npart, nk0, rnd, sim_el, input_dir, h_par, rigidities)


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data'
    p_inputs = data_dir / 'benchmark' / 'inputs'
    p_sims = data_dir / 'benchmark' / 'Simulations.list'

    sim_list = load_simulation_list(str(p_sims))
    for sim_el in sim_list:
        print(sim_el)
        make_input_from_sim(data_dir, p_inputs, sim_el)
