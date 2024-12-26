from os.path import join as pjoin, dirname

from extra.k0search.files_utils import load_experimental_data
from extra.k0search.physics_utils import initialize_output_dict
from extra.k0search.tasks import submit_sims
from files_utils import load_simulation_list, load_heliospheric_parameters
from grid import k0_grid_from_estimate

DEBUG = True
NK0 = 400  #Number of K0 to test
SLEEP_TIME = 60
APP = "FD006"
ROOTDIR = pjoin(dirname(dirname(__file__)), 'simfiles')

if __name__ == "__main__":
    psims = pjoin(ROOTDIR, f'Simulations_{APP}.list')
    ppastpar = pjoin(ROOTDIR, 'heliospheric_parameters', 'ParameterListALL_v12.txt')
    pfrcpar = pjoin(ROOTDIR, 'heliospheric_parameters', 'Frcst_param.txt')
    pexp = pjoin(ROOTDIR, 'experimental_data')
    presults = pjoin(ROOTDIR, 'results')
    pcosmica = pjoin(dirname(dirname(ROOTDIR)), 'Cosmica_1D', 'Cosmica_1D-rigi', 'exefiles', 'Cosmica')

    output_dict = {}
    sim_list = load_simulation_list(psims, output_dict, DEBUG)

    h_par = load_heliospheric_parameters(ppastpar, pfrcpar)

    sim_el = sim_list[0]
    k0_arr, k0_ref, k0_ref_err = k0_grid_from_estimate(int(sim_el[3]), NK0, h_par, DEBUG)
    exp_data = load_experimental_data(pexp, sim_el[2], (3, 11), DEBUG)
    initialize_output_dict(sim_el, exp_data, output_dict, k0_ref, k0_ref_err, DEBUG)

    output_dir_path, output_file_list = submit_sims(sim_el, pcosmica, presults, k0_arr, h_par, exp_data, max_workers=2, debug=DEBUG)
    print(len(output_file_list))
