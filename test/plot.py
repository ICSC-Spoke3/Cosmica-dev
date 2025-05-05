from glob import glob
from os.path import join as pjoin, dirname

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import gridspec

from lib.files_utils import load_experimental_data
from lib.files_utils import load_simulation_outputs, \
    load_simulation_list, load_lis
from lib.modulation import evaluate_modulations

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

def plot_fluxes(results, raw_results, results_labels=(), raw_results_labels=(), title=None, plot_path=None):
    rig, lis_flux = results.T[:2]
    fluxes = results.T[2:]

    assert len(fluxes) == len(results_labels), 'Missing labels'
    assert len(raw_results) == len(raw_results_labels), 'Missing raw labels'

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])

    if title is not None:
        fig.suptitle(title, fontsize=20)

    # AXIS 1
    ax1.plot(rig, lis_flux, label=fr'$\text{{Local Interstellar Spectrum (LIS)}}$',
             color='navy', linestyle='--', linewidth=1.5)

    for i, (flux, label) in enumerate(zip(fluxes, results_labels)):
        ax1.plot(rig, flux, label=fr'$\text{{Simulated: {label}}}$', linewidth=2, color=f'C{i}')

    for i, (raw, label) in enumerate(zip(raw_results, raw_results_labels)):
        assert np.allclose(rig, raw[:, 0], rtol=0.02 * rig), raw.shape[1] in (2, 4)
        if raw.shape[1] == 2:
            ax1.scatter(raw[:, 0], raw[:, 1], label=fr'$\text{{{label}}}$',
                        marker='x', color=f'C{i + len(results)}', s=300)
        else:
            ax1.errorbar(raw[:, 0], raw[:, 1], yerr=[raw[:, 2], raw[:, 3]], label=fr'$\text{{{label}}}$',
                         fmt='o', color='crimson', markersize=5, capsize=4, elinewidth=1)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\text{Energy Rigidity (GV)}$', fontsize=16)
    ax1.set_ylabel(r'$\text{Flux } \left(\frac{1}{\mathrm{GeV}/n \, \mathrm{m}^2 \, \mathrm{sr} \, \mathrm{s}}\right)$',
                   fontsize=16)

    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=10)

    ax1.grid(visible=True, which='major', linestyle='-', linewidth=0.75, alpha=0.8)
    ax1.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    ax1.legend(loc='upper right', fontsize=12, frameon=True)

    # AXIS 2
    norm = raw_results[1][:, 1]
    ax21.plot(rig, (lis_flux - norm) / norm, label=fr'$\text{{Local Interstellar Spectrum (LIS)}}$',
              color='navy', linestyle='--', linewidth=1.5)
    ax22.plot(rig, lis_flux - norm, label=fr'$\text{{Local Interstellar Spectrum (LIS)}}$',
              color='navy', linestyle='--', linewidth=1.5)

    for i, (raw, label) in enumerate(zip(raw_results, raw_results_labels)):
        if raw.shape[1] == 2:
            ax21.scatter(raw[:, 0], (raw[:, 1] - norm) / norm, label=fr'$\text{{{label}}}$',
                         marker='x', color=f'C{i + len(results)}', s=300)
            ax22.scatter(raw[:, 0], raw[:, 1] - norm, label=fr'$\text{{{label}}}$',
                         marker='x', color=f'C{i + len(results)}', s=300)
        else:
            ax21.errorbar(raw[:, 0], (raw[:, 1] - norm) / norm, yerr=[raw[:, 2] / norm, raw[:, 3] / norm],
                          label=fr'$\text{{{label}}}$',
                          fmt='o', color='crimson', markersize=5, capsize=4, elinewidth=1)
            ax22.errorbar(raw[:, 0], raw[:, 1] - norm, yerr=[raw[:, 2], raw[:, 3]], label=fr'$\text{{{label}}}$',
                          fmt='o', color='crimson', markersize=5, capsize=4, elinewidth=1)

    for i, (flux, label) in enumerate(zip(fluxes, results_labels)):
        ax21.scatter(rig, (flux - norm) / norm, label=fr'$\text{{Simulated: {label}}}$', color=f'C{i}')
        ax22.scatter(rig, flux - norm, label=fr'$\text{{Simulated: {label}}}$', color=f'C{i}')

    ax21.set_title('Relative')
    ax22.set_title('Absolute')
    ax21.set_ylim((-.1, .1))
    ax22.set_ylim((-100, 100))
    ax21.set_xscale('log')
    ax22.set_xscale('log')

    plt.subplots_adjust(right=0.8)
    plt.tight_layout()

    if plot_path is not None:
        plt.savefig(plot_path, dpi=300)
        plt.close()
    else:
        plt.show()


def evaluate_output(outputs, experimental_data, raw_data, lis, labels, plot_path=None):
    """
    Evaluate the output of a simulation and compare it with experimental data.
    :param output_path: path to the output file
    :param experimental_data: experimental data
    :param lis: LIS data
    :param rig_in: if the output is in energy
    :param plot_path: path to save the plot, if None the plot is not saved
    :return: RMSE between the simulation and the experimental data
    """

    results = evaluate_modulations(lis, *outputs)
    rig, lis_flux = results.T[:2]
    fluxes = results.T[2:]
    exp_en_rig, exp_j_mod, exp_inf, exp_sup = experimental_data.T
    raw_en_rig, raw_j_mod, = raw_data.T

    assert np.allclose(rig, exp_en_rig, rtol=0.02 * rig)

    rmses = []
    for i, flux in enumerate(fluxes):
        rmse = np.sqrt(np.square(np.subtract(flux, exp_j_mod)).mean())
        rmses.append(rmse)

    diffs = np.abs((fluxes[0] - fluxes[1]) / fluxes[0])
    print('diff', diffs.mean(), diffs.max())
    for i, f in enumerate(fluxes):
        err = np.abs(raw_j_mod - f) / raw_j_mod
        print(f'err_{i}', err.mean(), err.max())

    plot_fluxes(results, (raw_data, experimental_data), labels[0], labels[1], 'Comparison',
                plot_path)

    return rmses, diffs


def get_out(outputs, init_date):
    if outputs[0].endswith('.dat'):
        proton_res = next(filter(lambda f: init_date in f and 'Proton' in f, outputs), None)
        deuteron_res = next(filter(lambda f: init_date in f and 'Deuteron' in f, outputs), None)
        if not all([proton_res, deuteron_res]):
            return None
        res = load_simulation_outputs([proton_res, deuteron_res])
    else:
        proton_deuteron_res = next(filter(lambda f: init_date in f, outputs), None)
        if not proton_deuteron_res:
            return None
        res = load_simulation_outputs(proton_deuteron_res, yaml=True)
    return res


if __name__ == "__main__":
    ROOTDIR = pjoin(dirname(__file__), 'data')
    plis = pjoin(ROOTDIR, 'LIS_Default2020_Proton')
    pinputs = pjoin(ROOTDIR, 'inputs')

    poutputs = [
        pjoin(ROOTDIR, 'outputs', 'v6', '*.dat'),
        pjoin(ROOTDIR, 'outputs', 'v6.1', '*.dat'),
        pjoin(ROOTDIR, 'outputs', 'v8', '*.yaml'),
        pjoin(ROOTDIR, 'outputs', 'v8.1', '*.yaml'),
        pjoin(ROOTDIR, 'outputs', 'v8s', '*.dat'),
    ]
    labels = (('V6', 'V6 (random)', 'V8', 'V8 (many)', 'V8 (sep)'), ('HelMod', 'Experimental'))

    pexp = pjoin(ROOTDIR, 'outfile')
    praw = pjoin(ROOTDIR, 'helmod')
    psims = pjoin(ROOTDIR, f'Simulations.list')
    pplots = pjoin(dirname(__file__), 'plots')

    sim_list = load_simulation_list(psims)
    lis = load_lis(plis)

    outputs = [sorted(glob(p), reverse=True) for p in poutputs]

    diffs = []
    for sim_name, ions, file_name, init_date, end_date, rad, lat, lon in sim_list:
        print(sim_name, init_date)
        results = [get_out(o, init_date) for o in outputs]
        if not all(results):
            print(results)
            continue

        exp_data = load_experimental_data(pexp, file_name, cols=(2, 3, 4, 5), rig_range=(0, 100), to_rig=(1, 1))
        raw_data = load_experimental_data(praw, file_name, cols=(0, 1), rig_range=(0, 100))
        rmse, diff = evaluate_output(results, exp_data, raw_data, lis, labels, pjoin(pplots, f'{sim_name}.png'))
        diffs.append(diff)
        print(rmse)
        print()
    diffs = np.array(diffs)
    print(diffs.mean(), diffs.max())
