import os
import subprocess
import datetime

import numpy as np

from extra.k0search.isotopes import ISOTOPES
from extra.k0search.physics_utils import rig_to_en
from extra.k0search.utils import first_value_with_key_in


def create_input_file(k0vals, h_par, exp_data, input_path, sim_el, tot_npart_per_bin=1200,
                      n_heliosphere_regions=15, force_execute=False, debug=False):
    """
    Create an input file for Cosmica (I have very little understanding of this code)

    :param k0vals:
    :param h_par:
    :param exp_data:
    :param input_path:
    :param sim_el:
    :param tot_npart_per_bin:
    :param n_heliosphere_regions:
    :param force_execute:
    :param debug:
    :return:
    """
    sim_name, ions, file_name, init_date, end_date, rad, lat, lon = sim_el
    ions = [i.strip() for i in ions.split(',')]
    rad, lat, lon = [rad], [lat], [lon]
    # - lista di uscita
    input_file_names = []
    output_file_names = []
    # - verifica se il file di ingresso è in rigidità o Energia Cinetica
    tko = False if ("Rigidity" in file_name) else True

    # - ottieni la lista delle CR da simulare
    cr_ini = h_par[:, 0]
    cr_end = h_par[:, 1]
    ######################################################################
    # controlla se il periodo di integrazione rientra all'interno dei parametri temporali disponibili
    # se non lo fosse, si potrebbe cmq usare i parametri di forecast, ma questo caso va trattato (va aperta la lista FRCSTPARAMETERLIST e fatto una join con l'attuale, attenzione all'overlapping)
    if int(end_date) > int(cr_end[0]):
        print(
            f"WARNING:: End date for integration time ({end_date}) is after last available parameter ({int(cr_end[0])})")
        print(
            f"WARNING:: in questo caso la lista delle CR va integrata con la lista di forecast -- CASO DA SVILUPPARE skipped simulation")
        return []
    ######################################################################
    if int(init_date) < int(cr_ini[-n_heliosphere_regions]):
        print(
            f"WARNING:: Initial date for integration time ({init_date}) is before first available parameter ({int(cr_ini[-15])}) (including 15 regions)")
        print("           --> skipped simulation")
        return []

    cr_list = []
    cr_list_param = []
    # print(f"{init_date} - {end_date}")
    for iCR in range(len(cr_ini)):
        # print(f"{int(cr_ini[iCR])} {int(end_date)} {int(cr_end[iCR])} {int(init_date)}")
        # se sono in un range di CR tra init_date e end_date
        if int(cr_ini[iCR]) <= int(end_date) and int(cr_end[iCR]) > int(init_date):
            cr_list.append(cr_ini[iCR])

        # se sono in un range di CR tra init_date(-numero regioni) e end_date
        if int(cr_ini[iCR]) <= int(end_date) and int(cr_end[iCR - (n_heliosphere_regions - 1)]) > int(init_date):
            cr_list_param.append(cr_ini[iCR])

        # se sono ad un CR antecedente init_date(-numero regioni)
        if iCR - (n_heliosphere_regions - 1) >= 0:
            if int(cr_ini[iCR - (n_heliosphere_regions - 1)]) < int(init_date):
                break

    if len(cr_list) <= 0:
        print(f"WARNING:: CR list empty {init_date} {end_date}")
        return []

    # - controlla se i vettori posizioni in ingresso sono lunghi quanto cr_list,
    #   nel caso non lo fossero significa che la posizione è la stessa per tutte le simulazioni
    if len(rad) != len(cr_list):
        if len(rad) != 1:
            print("ERROR: Source array dimension != CR list selected")
            exit(1)
        np_rad = np.ones(len(cr_list)) * float(rad[0])
        np_lat = np.ones(len(cr_list)) * (90. - float(lat[0])) / 180. * np.pi
        np_lon = np.ones(len(cr_list)) * (float(lon[0]) / 180. * np.pi)
    else:
        np_rad = np.array([float(x) for x in rad])
        np_lat = np.array([(90. - float(x)) / 180. * np.pi for x in lat])
        np_lon = np.array([float(x) / 180. * np.pi for x in lon])

    if len(ions) > 1:
        addstring = f"_{'-'.join(ions)}"
    else:
        addstring = ''

    ##############################
    # K0 scan
    # k0scanMin=5e-5 #k0RefVal-7*k0RefVal_rel*k0RefVal
    # k0scanMax=6e-4 #k0RefVal+2*k0RefVal_rel*k0RefVal
    # k0vals=np.linspace(k0scanMin, k0scanMax, num=600, endpoint=True)
    ##############################
    base_add_string = addstring
    for k0val in k0vals:
        k0valstr = f"{k0val:.6e}".replace('.', "_")
        addstring = f"{base_add_string}_{k0valstr}"
        if debug: print(f"K0 to be simulated {k0valstr}")
        for ion in ions:
            # - ottieni la lista degli isotopi da simulare
            isotopes_list = first_value_with_key_in(ISOTOPES, ion)
            if debug: print(isotopes_list)
            if len(isotopes_list) <= 0:
                print(f"################################################################")
                print(f"WARNING:: {ion} not found in Isotopes_dict, please Check")
                print(f"################################################################")
                return []
            # - cicla sulle varie combinazioni

            for Isotopes in isotopes_list:
                # - crea nome  input file
                simulation_name_key = f"{Isotopes[3]}{addstring}{'_TKO' if tko else ''}_{cr_list[-1]:.0f}_{cr_list[0]:.0f}_r{float(rad[0]) * 100:05.0f}_lat{float(lat[0]) * 100:05.0f}"

                input_file_name = f"Input_{simulation_name_key}.txt"
                # - verifica se il file esiste, se non esiste or FORCE_EXECUTE=True allora crea il file e appendilo alla lista in input_file_names
                if not os.path.isfile(f"{input_path}/{input_file_name}") or force_execute:
                    target = open(f"{input_path}/{input_file_name}", "w")
                    target.write(f"#File generated on {datetime.date.today()}\n")

                    ############################ OutputFilename
                    target.write(f"OutputFilename: {simulation_name_key}\n")
                    ############################ particle to be simulated
                    target.write(f"# particle to be simulated\n")
                    target.write(f"Particle_NucleonRestMass: {Isotopes[2]}\n")
                    target.write(f"Particle_MassNumber: {Isotopes[1]}\n")
                    target.write(f"Particle_Charge: {Isotopes[0]}\n")

                    ############################ Load Energy Bins
                    target.write(f"# .. Generation energies -- NOTE the row cannot exceed 2000 chars\n")

                    tcentr = exp_data[:, 0] if tko else rig_to_en(exp_data[:, 0], Isotopes[1], Isotopes[0])
                    if tcentr.size > 1:
                        tcentr = ','.join(f"{x:.3e}" for x in tcentr)
                    tcentr = f"Tcentr: {tcentr} \n"
                    if len(tcentr) >= 2000:
                        print("ERROR: too much Energy inputs (exceeding allowed 2000 characters)")
                        exit(1)
                    target.write(tcentr)

                    ############################ Source (detector) position
                    target.write(f"# .. Source Position {len(np_lat)}\n")
                    str_source_the = f"SourcePos_theta: {','.join(f'{x:.5f}' for x in np_lat)} \n"
                    str_source_phi = f"SourcePos_phi: {','.join(f'{x:.5f}' for x in np_lon)} \n"
                    str_source_rad = f"SourcePos_r: {','.join(f'{x:.5f}' for x in np_rad)}  \n"
                    if len(str_source_the) >= 2000 or len(str_source_phi) >= 2000 or len(str_source_rad) >= 2000:
                        print("ERROR: too much source points (exceeding allowed 2000 characters)")
                        exit(1)
                    target.write(str_source_the)
                    target.write(str_source_phi)
                    target.write(str_source_rad)

                    ############################ Source (detector) position
                    target.write(f"# .. Number of particle to be generated\n")
                    target.write(f"Npart: {tot_npart_per_bin}\n")
                    ############################ Heliosphere Parameters
                    target.write(f"# .. Heliosphere Parameters\n")
                    target.write(f"Nregions: {n_heliosphere_regions}\n")
                    target.write(
                        f"# from {cr_list[0]} to {cr_list[-1]} - Total {len(cr_list_param)}({len(cr_list_param) - n_heliosphere_regions + 1}+{n_heliosphere_regions - 1}) input parameters periods\n")
                    target.write(
                        f"# . region 0 :            k0,    ssn,      V0, TiltAngle,SmoothTilt, Bfield, Polarity, SolarPhase, NMCR, Rts_nose, Rts_tail, Rhp_nose, Rhp_tail\n")
                    for hp in h_par:
                        if hp[0] in cr_list_param:
                            target.write(
                                "HeliosphericParameters: {:.6e},\t{:.3f},\t{:.2f},\t{:.2f},\t{:.3f},\t{:.3f},\t{:.0f},\t{:.0f},\t{:.3f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f}\n".format(
                                    k0val if cr_list_param.index(hp[0]) == 0 else 0.,  # k0 set only for first row
                                    *hp[[2, 3, 4, 12, 6, 7, 8, 11, 13, 14, 15, 16]]
                                )
                            )
                    target.write(f"# . heliosheat        k0,     V0,\n")
                    for i, hp in enumerate(h_par):
                        if hp[0] in cr_list:
                            target.write("HeliosheatParameters: {:.5e},\t{:.2f} \n".format(
                                3.e-05, h_par[i + n_heliosphere_regions - 1, 3]))
                    target.close()
                    input_file_names.append(f"{input_path}/{input_file_name}")
                    output_file_names.append(f"{simulation_name_key}")
    return input_file_names, output_file_names


def submit_sims(sim_el, results_path, k0_array, h_par, exp_data, debug=False):
    """
    Creates simulations and executes them locally or on a cluster.
    Returns the list of output files generated after the simulation.

    Args:
        sim_el (list): Simulation parameters including simulation name, ions, etc.
        k0_array (list): Array of K0 values to simulate.
        rig_range (list, optional): Range of values for simulation. Defaults to None.
        debug (bool): If True, print debug information. Defaults to False.

    Returns:
        tuple: Path to the input directory and list of output file names.
    """

    sim_name, ions, file_name, init_date, final_date, radius, latitude, longitude = sim_el
    ions_str = '-'.join(i.strip() for i in ions.split(','))
    is_tko = "Rigidity" not in file_name

    input_path = os.path.join(results_path, f"{sim_name}_{ions_str}")
    os.makedirs(input_path, exist_ok=True)

    # Generate input and output file names for the simulation
    input_file_list, output_file_list = create_input_file(k0_array, h_par, exp_data, input_path, sim_el,
                                                          force_execute=True, debug=debug)


    # FAXSIMILE COMMANDS
    #
    # nvcc   --ptxas-options=-v   --resource-usage   -rdc=true   -Xcompiler -fopenmp  --use_fast_math   -I ./Cosmica_1D-rigi/headers/ -o ./Cosmica_1D-rigi/exefiles/Cosmica  ./Cosmica_1D-rigi/kernel_test.cu ./Cosmica_1D-rigi/sources/DiffusionModel.cu ./Cosmica_1D-rigi/sources/GenComputation.cu ./Cosmica_1D-rigi/sources/GPUManage.cu ./Cosmica_1D-rigi/sources/HeliosphereLocation.cu ./Cosmica_1D-rigi/sources/HeliosphereModel.cu ./Cosmica_1D-rigi/sources/HeliosphericPropagation.cu ./Cosmica_1D-rigi/sources/HelModLoadConfiguration.cu ./Cosmica_1D-rigi/sources/HistoComputation.cu ./Cosmica_1D-rigi/sources/Histogram.cu ./Cosmica_1D-rigi/sources/LoadConfiguration.cu ./Cosmica_1D-rigi/sources/MagneticDrift.cu ./Cosmica_1D-rigi/sources/SDECoeffs.cu ./Cosmica_1D-rigi/sources/SolarWind.cu
    #
    # ./Cosmica_1D-en/exefiles/Cosmica -vv -i ./Cosmica_1D-en/runned_tests/AMS-02_PRL2015/Input_Proton_TKO_20110509_20131121_r00100_lat00000.txt >./Cosmica_1D-en/runned_tests/AMS-02_PRL2015/run/run_AMS-02_PRL2015_Proton_Proton_0.out 2>&1


    # if input_file_list:
    #     # If there are input files, prepare and run the simulation
    #     CreateRun(
    #         input_path,
    #         f"{sim_name}_{ions_str}{'_TKO' if is_tko else ''}",
    #         EXE_full_path,
    #         input_file_list,
    #     )
    #
    #     # Execute the simulation using Bash commands
    #     bash_commands = [
    #         f"cd {input_path}",
    #         f"./run_simulations.sh",
    #         f"cd {start_dir}"
    #     ]
    #     try:
    #         # Execute the commands in a shell
    #         result = subprocess.run(
    #             "; ".join(bash_commands),
    #             shell=True,
    #             check=True,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             text=True
    #         )
    #
    #         if debug:
    #             print("Command output:")
    #             print(result.stdout)
    #
    #         if result.stderr:
    #             print("Command error:")
    #             print(result.stderr)
    #
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error during simulation execution: {e}")
    #         if debug:
    #             print(e.output)

    return input_path, output_file_list
