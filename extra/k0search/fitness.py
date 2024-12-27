from os.path import basename

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from scipy.interpolate import interp1d

from extra.k0search.files_utils import load_simulation_output
from extra.k0search.physics_utils import evaluate_modulation

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


def evaluate_output(output_path, experimental_data, lis, output_in_energy=False, plot_path=None):
    """
    Evaluate the output of a simulation and compare it with experimental data.
    :param output_path: path to the output file
    :param experimental_data: experimental data
    :param lis: LIS data
    :param output_in_energy: if the output is in energy
    :param plot_path: path to save the plot, if None the plot is not saved
    :return: RMSE between the simulation and the experimental data
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

    if plot_path is not None:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot simulation and LIS
        ax.plot(sim_en_rig, sim_j_mod, label=r'$\text{Simulated: Cosmica}$', color='royalblue', linewidth=2)
        ax.plot(sim_en_rig, j_lis, label=r'$\text{Local Interstellar Spectrum (LIS)}$',
                color='navy', linestyle='--', linewidth=1.5)

        # Plot experimental data with error bars
        ax.errorbar(exp_en_rig, exp_j_mod, yerr=[exp_inf, exp_sup], fmt='o',
                    label=r'$\text{Experimental Data}$', color='crimson', markersize=5, capsize=4, elinewidth=1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\text{Energy Rigidity (GV)}$', fontsize=16)
        ax.set_ylabel(
            r'$\text{Flux } \left(\frac{1}{\mathrm{GeV}/n \, \mathrm{m}^2 \, \mathrm{sr} \, \mathrm{s}}\right)$',
            fontsize=16)
        ax.set_title(rf'$\text{{Comparison of }} {ion} \text{{ Simulation with Experimental Data}}$', fontsize=20,
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

    return rmse
