import sys
from pathlib import Path
from typing import Optional, Callable, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cmaes import CMA
import nevergrad as ng

from test.lib.files_utils import (
    LisLoader, SimulationPredictionItem, SimulationInput, HeliosphericParameters,
    SimulationExperimentItem, ExperimentalData, SimulationOutput, ModulationResult, estimate_k0
)
from test.lib.isotopes import IONS

import subprocess
import numpy as np
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
    """
        Calculate the fitness score for the given results against the experimental data.
        Args:
            results (list[ModulationResult]): List of modulation results from the simulation.
            experimental_data (ExperimentalData): The experimental data to compare against.
            metric_fn (Optional[Callable[[ModulationResult, ExperimentalData], float]]): A custom metric function to calculate the loss.
        Returns:
            list[float]: A list of fitness scores (losses) for each result.
    """
    losses: list[float] = []
    for result in results:
        if metric_fn is not None:
            losses.append(metric_fn(result, experimental_data))
        else:
            losses.append(float(np.sqrt(np.square(np.mean(result.flux - experimental_data.flux)))))

    return losses


def base_input(data_dir: Path, sim: SimulationExperimentItem, rnd: int = 42, n_part: int = 4096) -> SimulationInput:
    """
        Generate a base input for the simulation with given parameters.
        Args:
            data_dir (Path): The directory where the data files are located.
            sim (SimulationExperimentItem): The simulation experiment item containing the simulation parameters.
            rnd (int): Random seed for reproducibility.
            n_part (int): Number of particles to simulate.
        Returns:
            SimulationInput: A SimulationInput object with the base parameters set.
    """
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
    """
        Generate a new SimulationInput based on the base input and a list of k0 values.
        Args:
            base (SimulationInput): The base simulation input.
            k0s (list[float]): A list of k0 values to set in the dynamic parameters.
        Returns:
            SimulationInput: A new SimulationInput with updated dynamic parameters.
    """
    n_reg = len(base.static.heliosphere.v0)
    return base._replace(
        dynamic=SimulationInput.DynamicParameters(
            SimulationInput.DynamicParameters.DynamicHeliosphere(
                # k0=[np.full(n_reg, k0) for k0 in k0s],
                k0=[np.array([k0] + [0.0] * (n_reg - 1)) for k0 in k0s],
            ),
        ),
    )

def run_optimization(sim: SimulationExperimentItem, data_dir: Path, cosmica_path: Path, p_out: Path, lis_loader: LisLoader, names: List[str], population_size: int = 1, n_iterations: int = 3) -> Dict[str, Dict]:
    """
        Run the optimization process for the given simulation experiment.
        Args:
            sim (SimulationExperimentItem): The simulation experiment item containing the simulation parameters.
            data_dir (Path): The directory where the data files are located.
            cosmica_path (Path): The path to the Cosmica executable.
            p_out (Path): The output directory for the results.
            lis_loader (LisLoader): The loader for LIS data.
            names (List[str]): List of optimization algorithm names to use.
            population_size (int): Number of individuals in the population for each iteration.
            n_iterations (int): Number of iterations to run the optimization.
        Returns:
            Dict[str, Dict]: A dictionary containing the best parameters found for each optimization algorithm.
    """
    best_params = {}
    template = base_input(data_dir, sim)
    p_exp = data_dir / 'experimental' / sim.experimental_data_path
    exp_data = ExperimentalData.from_data(p_exp, (2, 3, 4, 5), rig_range=(0, 11))

    # Nevergrad parametrization
    lr = ng.p.Scalar(lower=5e-5, upper=6e-4)
    parametrization = ng.p.Instrumentation(lr)

    for name in names:
        # Initialization with estimated k0
        initial_k0 = estimate_k0(template)[0][0]
        print(f"Estimated initial k0: {initial_k0}")
        inpt = generate_input(template, [float(initial_k0)])
        iter_folder = p_out / 'initial'
        iter_folder.mkdir(parents=True, exist_ok=True)
        out = run_cosmica(inpt, cosmica_path, iter_folder / 'log_initial.log', iter_folder, cuda_devices='1')
        if out is None:
            raise RuntimeError(f"Cosmica run failed for initial k0={initial_k0}")
        results = out.modulate(lis_loader)
        fit = fitness_fn(results, exp_data)[0]

        init_param = parametrization.spawn_child()
        init_param.value = ((initial_k0,), {})

        evaluated_k0 = [initial_k0]
        evaluated_loss = [fit]

        optim = ng.optimizers.registry[name](
            parametrization=parametrization,
            budget=population_size,
            num_workers=population_size
        )
        optim.tell(init_param, fit)

        # Main optimization loop
        for iteration in range(n_iterations):
            k0_list = [optim.ask() for _ in range(population_size)]
            print(f"Current k0 list: {[k0.value[0][0] for k0 in k0_list]}")
            iter_folder = p_out / f'iteration_{iteration}'
            iter_folder.mkdir(parents=True, exist_ok=True)
            print(f"\nIteration {iteration + 1}")

            params = [float(k0.value[0][0]) for k0 in k0_list]
            inpt = generate_input(template, params)
            out = run_cosmica(inpt, cosmica_path, iter_folder / f'log.log', iter_folder, cuda_devices='1')
            if out is None:
                raise RuntimeError(f"Cosmica run failed for k0={params}")
            results = out.modulate(lis_loader)
            fitness = fitness_fn(results, exp_data)
            for k0, fit in zip(k0_list, fitness):
                evaluated_k0.append(float(k0.value[0][0]))
                evaluated_loss.append(fit)
                optim.tell(k0, fit)

        best_params[name] = {
            "best_x": float(np.min(evaluated_k0)),
            "best_loss": float(np.min(evaluated_loss)),
            "steps": evaluated_k0,
            "corr_loss": evaluated_loss,
        }
    return best_params


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    cosmica_path = Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / 'exefiles' / 'Cosmica'
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
    names = ["CMA"]
    best_params = run_optimization(
        sim=sim,
        data_dir=data_dir,
        cosmica_path=cosmica_path,
        p_out=p_out,
        lis_loader=lis_loader,
        names=names,
        population_size=1,
        n_iterations=3
    )
    print(best_params)


if __name__ == "__main__":
    main()