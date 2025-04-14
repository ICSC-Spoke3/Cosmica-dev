from glob import glob
from os.path import join as pjoin, dirname
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from lib.files_utils import load_simulation_outputs_yaml, load_simulation_output, load_simulation_outputs, \
    load_simulation_list, load_lis
from lib.modulation import evaluate_modulation
from lib.files_utils import load_experimental_data

# Setting rc params for all plots

plt.rcdefaults()

rc_params = {
    'axes.titlesize': 20,
    'axes.labelsize': 15,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1,
    'axes.grid': True,
    'figure.titlesize': 30,
    'axes.prop_cycle': plt.cycler(color=plt.cm.Dark2.colors),
}

plt.rcParams.update(rc_params)

colors = ['silver', 'skyblue', 'royalblue', 'blue', 'navy']
color_positions = [0.0, 0.25, 0.5, 0.75, 1]  # range of each color
cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(color_positions, colors)), N=1000)


# cmap = 'inferno'


def evaluate_output(outputs, experimental_data, lis, plot_path=None):
    """
    Evaluate the output of a simulation and compare it with experimental data.
    :param output_path: path to the output file
    :param experimental_data: experimental data
    :param lis: LIS data
    :param rig_in: if the output is in energy
    :param plot_path: path to save the plot, if None the plot is not saved
    :return: RMSE between the simulation and the experimental data
    """

    # outputs = [load_simulation_outputs(o, y) for o, y in zip(outputs, yamls)]

    # sim_en_rig, sim_j_mod, j_lis = evaluate_modulation(outputs, lis)
    mods = [evaluate_modulation(o, lis) for o in outputs]
    exp_en_rig, exp_j_mod = experimental_data.T[:2]

    rmses = []
    for i, (sim_en_rig, sim_j_mod, j_lis) in enumerate(mods):
        assert np.allclose(sim_en_rig, exp_en_rig, rtol=0.02 * sim_en_rig), i

        # print(np.subtract(sim_j_mod, exp_j_mod))
        rmse = np.sqrt(np.square(np.subtract(sim_j_mod, exp_j_mod)).mean())
        rmses.append(rmse)

    diffs = np.abs((mods[0][1] - mods[1][1]) / mods[0][1])
    # print(diffs)
    print(diffs.max(), diffs.mean())

    if plot_path is not None:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot simulation and LIS
        for i, (sim_en_rig, sim_j_mod, j_lis) in enumerate(mods):
            ax.plot(sim_en_rig, sim_j_mod, label=fr'$\text{{Simulated: Cosmica {i}}}$', linewidth=2, color=f'C{i}')

        ax.plot(sim_en_rig, j_lis, label=fr'$\text{{Local Interstellar Spectrum (LIS) {i}}}$',
                color='navy', linestyle='--', linewidth=1.5)

        # Plot experimental data with error bars
        # ax.errorbar(exp_en_rig, exp_j_mod, yerr=[exp_inf, exp_sup], fmt='o',
        #             label=r'$\text{Experimental Data}$', color='crimson', markersize=5, capsize=4, elinewidth=1)
        ax.scatter(exp_en_rig, exp_j_mod, marker='o',
                   label=r'$\text{Experimental Data}$', color='crimson', s=5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\text{Energy Rigidity (GV)}$', fontsize=16)
        ax.set_ylabel(
            r'$\text{Flux } \left(\frac{1}{\mathrm{GeV}/n \, \mathrm{m}^2 \, \mathrm{sr} \, \mathrm{s}}\right)$',
            fontsize=16)
        ax.set_title(
            rf'$\text{{Comparison of }} {list(outputs[0].keys())} \text{{ Simulation with Experimental Data}}$',
            fontsize=20,
            pad=20)

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        ax.grid(visible=True, which='major', linestyle='-', linewidth=0.75, alpha=0.8)  # Prominent major grid
        ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)  # Subtle minor grid

        ax.legend(loc='upper right', fontsize=12, frameon=True)
        plt.subplots_adjust(right=0.8)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return rmses, diffs


if __name__ == "__main__":
    ROOTDIR = pjoin(dirname(__file__), 'data')
    plis = pjoin(ROOTDIR, 'LIS_Default2020_Proton')
    pinputs = pjoin(ROOTDIR, 'inputs')

    # poutputs_a = pjoin(ROOTDIR, 'outputs', 'v6')
    poutputs_a = pjoin(ROOTDIR, 'outputs', 'v8')
    poutputs_b = pjoin(ROOTDIR, 'outputs', 'v6.1')
    # poutputs_b = pjoin(ROOTDIR, 'outputs', 'v8.1')

    pexp = pjoin(ROOTDIR, 'raw')
    psims = pjoin(ROOTDIR, f'Simulations.list')
    pplots = pjoin(dirname(__file__), 'plots')

    sim_list = load_simulation_list(psims)
    lis = load_lis(plis)

    # outputs_a = glob(pjoin(poutputs_a, '*.dat'))
    outputs_a = glob(pjoin(poutputs_a, '*.yaml'))
    outputs_b = glob(pjoin(poutputs_b, '*.dat'))
    # outputs_b = glob(pjoin(poutputs_b, '*.yaml'))

    diffs = []
    for sim_name, ions, file_name, init_date, end_date, rad, lat, lon in sim_list:
        print(sim_name, init_date)
        if outputs_a[0].endswith('.dat'):
            proton_res_a = next(filter(lambda f: init_date in f and 'Proton' in f, outputs_a), None)
            deuteron_res_a = next(filter(lambda f: init_date in f and 'Deuteron' in f, outputs_a), None)
            res_a = load_simulation_outputs([proton_res_a, deuteron_res_a])
        else:
            proton_deuteron_res_a = next(filter(lambda f: init_date in f, outputs_a), None)
            res_a = load_simulation_outputs(proton_deuteron_res_a, yaml=True)

        if outputs_b[0].endswith('.dat'):
            proton_res_b = next(filter(lambda f: init_date in f and 'Proton' in f, outputs_b), None)
            deuteron_res_b = next(filter(lambda f: init_date in f and 'Deuteron' in f, outputs_b), None)
            res_b = load_simulation_outputs([proton_res_b, deuteron_res_b])
        else:
            proton_deuteron_res_b = next(filter(lambda f: init_date in f, outputs_b), None)
            res_b = load_simulation_outputs(proton_deuteron_res_b, yaml=True)

        exp_data = load_experimental_data(pexp, file_name, (0, 100), ncols=2)
        rmse, diff = evaluate_output([res_a, res_b], exp_data, lis, pjoin(pplots, f'{sim_name}.png'))
        diffs.append(diff)
        print(rmse)
        print()
    diffs = np.array(diffs)
    print(diffs.mean(), diffs.max())
