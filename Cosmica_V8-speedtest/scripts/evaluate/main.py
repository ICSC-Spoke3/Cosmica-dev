import sys
from glob import glob
from os.path import join as pjoin, dirname, basename

import numpy as np

from files_utils import load_experimental_data, load_lis
from fitness import evaluate_output
# from physics_utils import initialize_output_dict
from files_utils import load_simulation_list, load_heliospheric_parameters

DEBUG = True
NK0 = 3  # Number of K0 to test
APP = "all"
ROOTDIR = pjoin(dirname(dirname(dirname(dirname(__file__)))), 'extra', 'simfiles')

if __name__ == "__main__":
    sim_name = sys.argv[1]
    sim_iso = sys.argv[2]
    out_file = sys.argv[3:]

    psims = pjoin(ROOTDIR, f'Simulations_{APP}.list')

    ppastpar = pjoin(ROOTDIR, 'heliospheric_parameters', 'ParameterListALL_v12.txt')
    pfrcpar = pjoin(ROOTDIR, 'heliospheric_parameters', 'Frcst_param.txt')
    pexp = pjoin(ROOTDIR, 'experimental_data')
    presults = pjoin(ROOTDIR, 'results')
    pcosmica = pjoin(dirname(dirname(ROOTDIR)), 'Cosmica_1D', 'Cosmica_1D-en', 'exefiles', 'Cosmica')
    plis = pjoin(ROOTDIR, 'GALPROP_LIS_Esempio')
    pfit = pjoin(ROOTDIR, 'figures', 'output.txt')
    pplots = pjoin(ROOTDIR, 'figures')

    sim_list = load_simulation_list(psims)
    sim_el = next(filter(lambda s: s[0] == sim_name and s[1] == sim_iso, sim_list))

    h_par = load_heliospheric_parameters(ppastpar, pfrcpar)

    lis = load_lis(plis)

    print(f'Running simulation: {sim_el}')

    exp_data = load_experimental_data(pexp, sim_el[2], (3, 11), DEBUG)

    print(pjoin(pplots, basename(out_file[0]) + '.png'))
    rmse = evaluate_output(out_file, exp_data, lis, plot_path=pjoin(dirname(__file__), basename(out_file[0]) + '.png'))
    print(f'RMSE: {rmse}')

    # print(sims_dict)
    # rmses = []
    # for (ion, k0), sim_names in sims_dict.items():
    #     sim_spec = f'{ion}_{k0:.6e}'.replace('.', "_")
    #     outputs = [sorted(glob(pjoin(output_dir, f'{fn}*.dat')))[-1] for fn in sim_names]
    #     rmses.append(f'{sim_spec},{rmse:.3f}')
    #     print(f'{sim_spec} RMSE: {rmse}')
    # with open(pfit, 'w') as f:
    #     f.write('fname,rmse\n')
    #     f.write('\n'.join(rmses))
