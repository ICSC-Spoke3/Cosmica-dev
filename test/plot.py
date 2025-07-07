import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional, Any

import yaml

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from test.lib.files_utils import SimulationList, LisLoader, SimulationOutput, ExperimentalData, ModulationResult

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

def plot_fluxes(results: list[ModulationResult], raw_results: list[ExperimentalData], results_labels=(),
                raw_results_labels=(), title=None, plot_path=None):
    assert len(results) == len(results_labels), 'Missing labels'
    assert len(raw_results) == len(raw_results_labels), 'Missing raw labels'

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    ax1, ax2 = axs.flatten()

    if title is not None:
        fig.suptitle(title, fontsize=20)

    # AXIS 1
    ax1.plot(*results[0].lis_flux, label=fr'$\text{{Local Interstellar Spectrum (LIS)}}$',
             color='navy', linestyle='--', linewidth=1.5)

    for i, (res, label) in enumerate(zip(results, results_labels)):
        ax1.plot(*res.rig_flux, label=fr'$\text{{Simulated: {label}}}$', linewidth=2, color=f'C{i}')

    for i, (raw, label) in enumerate(zip(raw_results, raw_results_labels)):
        if raw.limits is None:
            ax1.scatter(*raw.rig_flux, label=fr'$\text{{{label}}}$',
                        marker='x', color=f'C{i + len(results)}', s=300)
        else:
            ax1.errorbar(*raw.rig_flux, yerr=list(raw.limits), label=fr'$\text{{{label}}}$',
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
    norm = raw_results[1].flux
    ax2.plot(results[0].rigidity, (results[0].lis - norm) / norm,
             label=fr'$\text{{Local Interstellar Spectrum (LIS)}}$',
             color='navy', linestyle='--', linewidth=1.5)

    for i, (raw, label) in enumerate(zip(raw_results, raw_results_labels)):
        if raw.limits is None:
            ax2.scatter(raw.rigidity, (raw.flux - norm) / norm, label=fr'$\text{{{label}}}$',
                        marker='x', color=f'C{i + len(results)}', s=300)
        else:
            ax2.errorbar(raw.rigidity, (raw.flux - norm) / norm, yerr=[raw.limits[0] / norm, raw.limits[1] / norm],
                         label=fr'$\text{{{label}}}$',
                         fmt='o', color='crimson', markersize=5, capsize=4, elinewidth=1)

    for i, (res, label) in enumerate(zip(results, results_labels)):
        ax2.scatter(res.rigidity, (res.flux - norm) / norm, label=fr'$\text{{Simulated: {label}}}$', color=f'C{i}')

    ax2.set_title('Relative')
    ax2.set_ylim((-.1, .1))
    ax2.set_xscale('log')

    plt.subplots_adjust(right=0.8)
    plt.tight_layout()

    return fig, ax1, ax2, norm


def evaluate_output(outputs: SimulationOutput, experimental_data: ExperimentalData, raw_data: ExperimentalData,
                    lis_loader: LisLoader, labels: tuple[list[str], list[str]], plot_path: Optional[Path] = None):
    """
    Evaluate the output of a simulation and compare it with experimental data.
    :param output_path: path to the output file
    :param experimental_data: experimental data
    :param lis_loader: LIS data
    :param rig_in: if the output is in energy
    :param plot_path: path to save the plot, if None the plot is not saved
    :return: RMSE between the simulation and the experimental data
    """

    results = [o.trim(0, 11) for o in outputs.modulate(lis_loader)]

    assert np.allclose(results[0].rigidity, experimental_data.rigidity, rtol=0.02 * results[0].rigidity)
    assert np.allclose(results[0].rigidity, raw_data.rigidity, rtol=0.02 * results[0].rigidity)
    for i, (r1, r2) in enumerate(zip(results[:-1], results[1:])):
        assert np.allclose(r1.rigidity, r2.rigidity, rtol=0.02 * r1.rigidity), (i, i + 1)

    rmses = []
    for i, res in enumerate(results):
        rmse = np.sqrt(np.square(np.subtract(res.flux, experimental_data.flux)).mean())
        rmses.append(rmse)

    # diffs = np.abs((fluxes[0] - fluxes[1]) / fluxes[0])
    # print('diff', diffs.mean(), diffs.max())
    # for i, f in enumerate(fluxes):
    #     err = np.abs(raw_j_mod - f) / raw_j_mod
    #     print(f'err_{i}', err.mean(), err.max())

    fig, ax1, ax2, norm = plot_fluxes(
        results,
        [raw_data, experimental_data],
        labels[0], labels[1],
        'Comparison', plot_path
    )

    # norm = lin_log_interpolation(rig, norm, results[:, 0])
    #
    # ax1.plot(results[:, 0], results[:, 1], label=fr'$\text{{LISS}}$',)
    # ax1.plot(results[:, 0], results[:, 2], label=fr'$\text{{V6}}$',)
    # ax1.plot(results[:, 0], results[:, 3], label=fr'$\text{{V8}}$',)
    # ax1.legend(loc='upper right', fontsize=12, frameon=True)
    #
    # ax2.scatter(results[:, 0], (results[:, 1]-norm)/norm, label=fr'$\text{{LISS}}$',)
    # ax2.scatter(results[:, 0], (results[:, 2]-norm)/norm, label=fr'$\text{{V6}}$',)
    # ax2.scatter(results[:, 0], (results[:, 3]-norm)/norm, label=fr'$\text{{V8}}$',)

    if plot_path is not None:
        plt.savefig(plot_path, dpi=300)
        plt.close()
    else:
        plt.show()

    return rmses, diffs


def match_file(files: list[Path], *vals: Any) -> Optional[Path]:
    return next(filter(lambda f: all((str(v).lower() in f.name.lower() for v in vals)), files), None)


def get_out(outputs: list[Path], init_date: int) -> SimulationOutput:
    if outputs[0].suffix == '.dat':
        proton_res = match_file(outputs, init_date, 'proton')
        deuteron_res = match_file(outputs, init_date, 'deuteron')
        with open(proton_res, 'r') as fp, open(deuteron_res, 'r') as fd:
            return SimulationOutput.from_txt([{'proton': fp.read(), 'deuteron': fd.read()}])

    proton_deuteron_res = match_file(outputs, init_date)
    with open(proton_deuteron_res, 'r') as f:
        return SimulationOutput.from_yaml(yaml.load(f, Loader=yaml.SafeLoader))


if __name__ == "__main__":
    data_dir = Path(__file__).parent / 'data'
    p_sims = data_dir / 'Simulations.list'
    p_lis = data_dir / 'LIS_Default2020_Proton'
    p_inputs = data_dir / 'inputs'
    p_exp = (data_dir / 'experimental').glob('*.dat')
    p_raw = (data_dir / 'helmod').glob('*.txt')
    p_plots = data_dir / 'plots'

    p_outputs = [
        (data_dir / 'outputs' / 'v6').glob('*.dat'),
        (data_dir / 'outputs' / 'v8').glob('*.yaml'),
        # (data_dir / 'outputs' / 'v8s').glob('*.dat'),
        (data_dir / 'outputs' / 'v6.1').glob('*.dat'),
        # (data_dir / 'outputs' / 'v8.1').glob('*.yaml'),
        # (data_dir / 'outputs' / 'v8.1s').glob('*.dat'),
        (data_dir / 'outputs' / 'v8m').glob('*.yaml'),
        # (data_dir / 'outputs' / 'v8.1m').glob('*.yaml'),
    ]
    labels = (['V6 (1)', 'V8 (1)', 'V6 (2)', 'V8 (1, mul)'],
              ['HelMod', 'Experimental'])
    # labels = (['V6 (1)', 'V8 (1)', 'V8 (1, sep)', 'V6 (2)', 'V8 (2)', 'V8 (2, sep)', 'V8 (1, mul)', 'V8 (2, mul)'],
    #           ['HelMod', 'Experimental'])
    # labels = (('V6', 'V6 (random)', 'V8', 'V8 (many)', 'V8 (sep)'), ('HelMod', 'Experimental'))

    sim_list = SimulationList.from_listfile(p_sims)
    lis_loader = LisLoader(p_lis)

    outputs = [sorted(p, reverse=True) for p in p_outputs]
    experimental = sorted(p_exp, reverse=True)
    helmod = sorted(p_raw, reverse=True)

    diffs = []
    for sim in sim_list:
        print(sim)
        init_date = sim.period[0]
        results = SimulationOutput.from_outputs(*[get_out(o, init_date) for o in outputs])

        exp_data = ExperimentalData.from_data(match_file(experimental, init_date), (2, 3, 4, 5))
        raw_data = ExperimentalData.from_data(match_file(helmod, init_date), (0, 1))

        rmse, diff = evaluate_output(results, exp_data, raw_data, lis_loader, labels, p_plots / f'{sim.name}.png')
        diffs.append(diff)
        print(rmse)
        print()
    # diffs = np.array(diffs)
    # print(diffs.mean(), diffs.max())
