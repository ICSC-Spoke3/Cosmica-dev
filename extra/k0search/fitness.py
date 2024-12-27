from os.path import basename

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from extra.k0search.files_utils import load_simulation_output
from extra.k0search.physics_utils import evaluate_modulation


def evaluate_output(output_path, experimental_data, lis, output_in_energy=False, plot_path=None):
    """

    :param output_path:
    :param experimental_data:
    :param lis:
    :param output_in_energy:
    :param plot_path:
    :return:
    """
    output, _ = load_simulation_output(output_path)
    ion = basename(output_path).split('_')[0]

    sim_en_rig, sim_j_mod, j_lis = evaluate_modulation(ion, lis, output, output_in_energy)
    exp_en_rig, exp_j_mod, exp_inf, exp_sup = experimental_data.T

    lower_bound = max(exp_en_rig.min(), sim_en_rig.min())
    upper_bound = min(exp_en_rig.max(), sim_en_rig.max())

    space = np.linspace(lower_bound, upper_bound, 100)
    sim_interp = interp1d(sim_en_rig, sim_j_mod)(space)
    exp_interp = interp1d(exp_en_rig, exp_j_mod)(space)
    rmse = np.sqrt(np.square(np.subtract(sim_interp, exp_interp)).mean())

    if plot_path is not None:  # TODO: nice plots!
        plt.plot(sim_en_rig, sim_j_mod, label='cosmica', color='blue')
        plt.plot(sim_en_rig, j_lis, label='lis', color='yellow')
        plt.errorbar(exp_en_rig, exp_j_mod, [exp_inf, exp_sup], marker='x' ,label='experimental data', color='red')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(plot_path)
        plt.close()

    return rmse
