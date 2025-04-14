from glob import glob
from os.path import join as pjoin, dirname, basename

import numpy as np

from extra.k0search.files_utils import load_experimental_data, load_lis
from extra.k0search.fitness import evaluate_output
# from extra.k0search.physics_utils import initialize_output_dict
from extra.k0search.tasks import submit_sims
from extra.k0search.files_utils import load_simulation_list, load_heliospheric_parameters
from extra.k0search.grid import k0_grid_from_estimate

DEBUG = True
NK0 = 3  # Number of K0 to test
APP = "FD006"
ROOTDIR = pjoin(dirname(dirname(__file__)), 'simfiles')

if __name__ == "__main__":
    psims = pjoin(ROOTDIR, f'Simulations_{APP}.list')
    ppastpar = pjoin(ROOTDIR, 'heliospheric_parameters', 'ParameterListALL_v12.txt')
    pfrcpar = pjoin(ROOTDIR, 'heliospheric_parameters', 'Frcst_param.txt')
    pexp = pjoin(ROOTDIR, 'experimental_data')
    presults = pjoin(ROOTDIR, 'results')
    pcosmica = pjoin(dirname(dirname(ROOTDIR)), 'Cosmica_1D', 'Cosmica_1D-en', 'exefiles', 'Cosmica')
    plis = pjoin(ROOTDIR, 'GALPROP_LIS_Esempio')
    pfit = pjoin(ROOTDIR, 'figures', 'output.txt')
    pplots = pjoin(ROOTDIR, 'figures')

    sim_list = load_simulation_list(psims, DEBUG)

    h_par = load_heliospheric_parameters(ppastpar, pfrcpar)

    lis = load_lis(plis)

    sim_el = sim_list[14]
    print(f'Running simulation: {sim_el}')
    k0_arr, k0_ref, k0_ref_err = k0_grid_from_estimate(int(sim_el[3]), NK0, h_par, DEBUG)
    # k0_arr = np.linspace(5e-5, 6e-4, 1)

    exp_data = load_experimental_data(pexp, sim_el[2], (3, 11), DEBUG)

    sims_dict, output_dir = submit_sims(sim_el, pcosmica, presults, k0_arr, h_par, exp_data, max_workers=2, debug=DEBUG)
    print(sims_dict)
    rmses = []
    for (ion, k0), sim_names in sims_dict.items():
        sim_spec = f'{ion}_{k0:.6e}'.replace('.', "_")
        outputs = [sorted(glob(pjoin(output_dir, f'{fn}*.dat')))[-1] for fn in sim_names]
        rmse = evaluate_output(outputs, exp_data, lis, plot_path=pjoin(pplots, sim_spec + '.png'))
        rmses.append(f'{sim_spec},{rmse:.3f}')
        print(f'{sim_spec} RMSE: {rmse}')
    with open(pfit, 'w') as f:
        f.write('fname,rmse\n')
        f.write('\n'.join(rmses))
