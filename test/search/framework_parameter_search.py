import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Optional, Callable

from cmaes import CMA
from fstpso import FuzzyPSO

from test.lib.files_utils import LisLoader, SimulationPredictionItem, SimulationInput, HeliosphericParameters, \
    SimulationExperimentItem, ExperimentalData, SimulationOutput, ModulationResult
from test.lib.isotopes import IONS


import subprocess

import numpy as np
from pathlib import Path

import yaml

yaml.Dumper.ignore_aliases = lambda self, data: True


def run_cosmica(inpt: SimulationInput, cosmica_executable: Path, log_file: Path, output_dir: Path,
                cuda_devices: str = '0,1') -> Optional[
    SimulationOutput]:
    try:
        command = [
            str(cosmica_executable),
            "-v",
            "debug",
            "--stdin",
            "--stdout",
            "--log_file",
            str(log_file),
            "-o",
            str(output_dir) + '/',
        ]
        print(f"Executing command: {' '.join(command)}")

        input_string = yaml.dump(inpt.to_dict())

        process = subprocess.run(command, env={'CUDA_VISIBLE_DEVICES': cuda_devices},
                                 input=input_string, capture_output=True, text=True)

        if process.returncode != 0:
            return None

        return SimulationOutput.from_yaml(yaml.load(process.stdout, Loader=yaml.SafeLoader))

    except FileNotFoundError:
        print(f"Error: Cosmica executable not found at {cosmica_executable}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def mistery_function_next_k0list(fitness_score: list[float], k0_list: list[float]):
    """
    Generate the next k0 list based on the fitness score and current k0 list.
    
    Args:
        fitness_score (float): The fitness score from the previous evaluation.
        k0_list (list): The current list of k0 values.
        
    Returns:
        list: A new list of k0 values for the next iteration.
    """
    assert len(fitness_score) == len(k0_list)

    # Placeholder
    print(f"=== Placeholder for mistery_function_next_k0list ===")
    next_k0_list = [k + 0.1 for k in k0_list]
    print(f"Next k0 list based on fitness score {fitness_score}: {next_k0_list}")
    return next_k0_list


def fitness_fn(results: list[ModulationResult], experimental_data: ExperimentalData,
               metric_fn: Optional[Callable[[ModulationResult, ExperimentalData], float]] = None) -> list[float]:
    losses: list[float] = []
    for result in results:
        if metric_fn is not None:
            losses.append(metric_fn(result, experimental_data))
        else:
            losses.append(float(np.sqrt(np.square(np.mean(result.flux - experimental_data.flux)))))

    return losses


def base_input(data_dir: Path, sim: SimulationExperimentItem, rnd: int = 42, n_part: int = 4096) -> SimulationInput:
    p_past_par = data_dir / 'heliospheric_parameters' / 'ParameterListALL_v12.txt'
    p_frct_par = data_dir / 'heliospheric_parameters' / 'Frcst_param.txt'
    heliospheric_parameters = HeliosphericParameters.from_files(p_past_par, p_frct_par)

    isotopes = sim.ions[0].isotopes

    p_exp = data_dir / 'experimental' / sim.experimental_data_path
    experimental_data = ExperimentalData.from_data(p_exp, (2, 3, 4, 5))
    rigidities = experimental_data.rig_flux.rigidity

    sphere, sheat = heliospheric_parameters.in_period(sim.period, 15)

    dynamic = SimulationInput.DynamicParameters.DynamicHeliosphere([np.full(len(sphere), 0.)])

    static_sphere = SimulationInput.StaticParameters.StaticHeliosphere(
        *sphere[:, [2, 3, 4, 12, 6, 7, 8, 11, 13, 14, 15, 16]].T)
    static_sheat = SimulationInput.StaticParameters.StaticHeliosheat(np.full(len(sheat), 3.e-05), sheat[:, 3])

    return SimulationInput(
        random_seed=rnd,
        output_path='search',
        rigidities=rigidities,
        isotopes=isotopes,
        sources=sim.sources,
        n_particles=n_part,
        n_regions=15,
        dynamic=SimulationInput.DynamicParameters(dynamic),
        static=SimulationInput.StaticParameters(static_sphere, static_sheat)
    )


def generate_input(base: SimulationInput, k0s: list[float]) -> SimulationInput:
    n_reg = len(base.static.heliosphere.v0)
    return base._replace(
        dynamic=SimulationInput.DynamicParameters(
            SimulationInput.DynamicParameters.DynamicHeliosphere(
                # k0=[np.full(n_reg, k0) for k0 in k0s],
                k0=[np.array([k0] + [0.0] * (n_reg - 1)) for k0 in k0s],
            ),
        ),
    )


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data'
    p_cosmica = Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / 'exefiles' / 'Cosmica'
    p_out = data_dir / 'search' / 'output'

    p_lis = data_dir / 'LIS_Default2020_Proton'

    lis_loader = LisLoader(p_lis)

    sim = SimulationExperimentItem(
        name='search',
        ions=[IONS.get('proton')],
        period=(20180929, 20181025),
        sources=(np.array([1.0]), np.array([1.5707963267948966]), np.array([0.0])),
        experimental_data_path='Rigidity_Proton_AMS-02_PRL1272021271102_20180929_20181025.dat',
    )

    template = base_input(data_dir, sim)

    p_exp = data_dir / 'experimental' / sim.experimental_data_path
    exp_data = ExperimentalData.from_data(p_exp, (2, 3, 4, 5), rig_range=(0, 11))

    population_size = 5
    optimizer = CMA(mean=np.full(2, 0.000325), sigma=1)


    for iteration in range(3):
        k0_list = [optimizer.ask() for _ in range(population_size)]

        iter_folder = p_out / f'iteration_{iteration}'
        iter_folder.mkdir(parents=True, exist_ok=True)

        print(f"\nIteration {iteration + 1}")
        inpt = generate_input(template, [float(k0[0]) for k0 in k0_list])

        # Run Cosmica with the generated input
        print("Running Cosmica with the generated input...")
        out = run_cosmica(inpt, p_cosmica, iter_folder / 'log.log', iter_folder, cuda_devices='1')

        if out is None:
            print("Failed to load simulation outputs.")
            continue

        results = out.modulate(lis_loader)

        fitness = fitness_fn(results, exp_data)
        print(len(k0_list), len(fitness), fitness)

        optimizer.tell([(k0, fit) for k0, fit in zip(k0_list, fitness)])
        # k0_list = mistery_function_next_k0list(fitness, k0_list)
        print(f"Next k0 list: {k0_list}")
