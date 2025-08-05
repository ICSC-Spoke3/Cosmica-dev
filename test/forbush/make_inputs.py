import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test.lib.files_utils import HeliosphericParameters, SimulationExperimentItem, ExperimentalData, \
    SimulationInput, SimulationList

import numpy as np
import yaml

yaml.Dumper.ignore_aliases = lambda *args: True


class InlineList(list):
    @staticmethod
    def inline_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(InlineList, InlineList.inline_list_representer)


def make_single_input(data_dir: Path, folder_name: str, sim: SimulationExperimentItem,
                      heliospheric_parameters: HeliosphericParameters, n_particles: int, n_k0: int, rnd: int):
    p_inputs = data_dir / 'forbush' / 'inputs' / folder_name
    p_exp = data_dir / 'forbush' / 'experimental'

    isotopes = sim.ions[0].isotopes

    isotopes_str = '_'.join(i.name for i in isotopes)
    sim_spec_str = f'{sim.period[0]}_{sim.period[1]}_{n_particles}_{n_k0}_{rnd}'
    full_spec_str = f'{isotopes_str}_{sim_spec_str}'

    p_inputs.mkdir(parents=True, exist_ok=True)

    experimental_data = ExperimentalData.from_data(p_exp / sim.experimental_data_path, (0, 1, 2, 3))
    rigidities = experimental_data.rig_flux.rigidity
    sphere, sheat = heliospheric_parameters.in_period(sim.period, 15)

    dynamic = SimulationInput.DynamicParameters.DynamicHeliosphere([np.full(len(sphere), 0.)] * n_k0)

    static_sphere = SimulationInput.StaticParameters.StaticHeliosphere(
        *sphere[:, [2, 3, 4, 12, 6, 7, 8, 11, 13, 14, 15, 16]].T)
    static_sheat = SimulationInput.StaticParameters.StaticHeliosheat(np.full(len(sheat), 3.e-05), sheat[:, 3])

    inpt = SimulationInput(
        random_seed=rnd,
        output_path=full_spec_str,
        rigidities=rigidities,
        isotopes=isotopes,
        sources=sim.sources,
        n_particles=n_particles,
        n_regions=15,
        dynamic=SimulationInput.DynamicParameters(dynamic),
        static=SimulationInput.StaticParameters(static_sphere, static_sheat)
    )

    with open(p_inputs / f'{full_spec_str}.yaml', 'w') as f:
        yaml.dump(inpt.to_dict(), f, sort_keys=False, width=float("inf"))

    for iso, txt in inpt.to_txt(lambda _, i, __: f'{i.name}_{sim_spec_str}')[0].items():
        with open(p_inputs / f'{iso.name}_{sim_spec_str}.txt', 'w') as f:
            f.write(txt)


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data'
    p_sims = data_dir / 'forbush'
    p_past_par = data_dir / 'heliospheric_parameters' / 'ParameterListALL_v12.txt'
    p_frct_par = data_dir / 'heliospheric_parameters' / 'Frcst_param.txt'

    heliospheric_parameters = HeliosphericParameters.from_files(p_past_par, p_frct_par)

    p_sim_lists = p_sims.glob('Simulations_FD*.list')
    for p_sim_list in p_sim_lists:
        sim_list = SimulationList.from_listfile(p_sim_list)
        for sim in sim_list:
            folder_name = f'{str(p_sim_list.stem.split('_')[-1])}/{sim.name}'
            print(folder_name)
            make_single_input(data_dir, folder_name, sim, heliospheric_parameters, 4096, 1, 42)
