import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Callable, List, Dict

import pandas as pd
from fastparquet import write

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import nevergrad as ng

from test.lib.files_utils import (
    LisLoader, SimulationPredictionItem, SimulationInput, HeliosphericParameters,
    SimulationExperimentItem, ExperimentalData, SimulationOutput, ModulationResult, estimate_k0
)
from test.lib.isotopes import IONS
from test.lib import metrics

import subprocess
import numpy as np
import yaml

yaml.Dumper.ignore_aliases = lambda self, data: True



########################################################################
# Version 1.0:
# Version 2.0: Enhanced fitness evaluation: if the result lies within the experimental sensitivity bounds, it is not penalized
#
#
#######################################################################

def run_cosmica(inpt, base_command: list[str | Path] | str | Path, log_file: Path, output_dir: Path,
                cuda_devices: str = '1,3,5,7') -> Optional[
    SimulationOutput]:
    try:

        command = list(map(str, base_command)) if isinstance(base_command, list) else [str(base_command)]


        command += [
            "-v",
            "debug",
            "-i", str(inpt),
            "--stdout",
            "--log_file",
            str(log_file),
            "-o",
            str(output_dir) + '/',
            ]

        print(f"Executing command: {' '.join(command)}")


        process = subprocess.run(command, env={'CUDA_VISIBLE_DEVICES': cuda_devices}, capture_output=True, text=True)
        #process = subprocess.run(command, env={'CUDA_VISIBLE_DEVICES': cuda_devices}, capture_output=True, text=True)

        if process.returncode != 0:
            return None

        file_output = output_dir / 'output.yaml'

        if not file_output.exists():
            file_output.touch()

        with open(file_output, 'w') as f:
            f.write(process.stdout)

        lines = process.stdout.splitlines()
        # filtoro l output fino ad histograms
        start_index = next((i for i, line in enumerate(lines) if line.strip().startswith("histograms:")), None)

        if start_index is not None:
            relevant_output = "\n".join(lines[start_index:])
        else:
            print("Warning: 'histograms:' section not found in output")
            relevant_output = process.stdout  # fallback to all output


        # Write filtered output
        with open(file_output, 'w') as f:

            f.write(relevant_output)

        return SimulationOutput.from_yaml(yaml.load(relevant_output, Loader=yaml.SafeLoader))

    except FileNotFoundError:
        print(f"Error: Cosmica executable not found at {base_command}")
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
        lower,upper = experimental_data.limits

        if lower < result < upper:
            losses.append(0)
            continue

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


def get_mean_fitness(template,init_pop,k_validation, file_input, cosmica_path, iter_folder, lis_loader, metric_fn, exp_data, initial_k0)->List[float]:

    all_fitness = []
    starting_seed = template.random_seed

    for i in range(k_validation):

        new_template = template._replace(random_seed=starting_seed + i)

        inpt = generate_input(new_template, init_pop)

        with open(file_input, 'w') as f:
            yaml.dump(inpt.to_dict(), f, Dumper=yaml.Dumper)


        print(f"K-{i} run")
        out = run_cosmica(file_input, cosmica_path, iter_folder / 'log_initial.log', iter_folder)

        if out is None:
            raise RuntimeError(f"Cosmica run failed for initial k0={initial_k0}")

        results = out.modulate(lis_loader)

        print("Find results")
        fitness_list = fitness_fn(results, exp_data, metric_fn=metric_fn)

        all_fitness.append(fitness_list)

    all_fitness = np.array(all_fitness)

    print(f"all Fitness: {all_fitness}")

    mean_fitness = np.mean(all_fitness, axis=0)
    print(f"Mean Fitness{mean_fitness}")

    return mean_fitness.tolist()



def run_optimization(sim: SimulationExperimentItem, data_dir: Path, cosmica_path: list[str | Path] | str | Path, p_out: Path, lis_loader: LisLoader, names: List[str],
                     population_size: int = 10, n_iterations: int = 3,random_seed :int = 42,n_part:int=4096,
                        min_improvement: float = 0.001,epochs=1,max_patience = 10,metric_fn = metrics.mean_relative_error,k_validation = 1):
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
            :param n_part:
            :param random_seed:
            :param names:
            :param lis_loader:
            :param sim:
            :param p_out:
            :param metrics:
            :param epochs:
            :param n_iterations:
            :param cosmica_path:
            :param data_dir:
            :param population_size:
            :param max_patience:
            :param min_improvement:
    """
    best_params = {}

    file_input = Path(__file__).parent / 'inputs' / 'test.yaml'

    if not file_input.exists():
        file_input.touch()

    template = base_input(data_dir, sim, rnd=random_seed, n_part=n_part)
    p_exp = data_dir / 'experimental' / sim.experimental_data_path
    exp_data = ExperimentalData.from_data(p_exp, (2, 3, 4, 5), rig_range=(0, 11))
    parquet_dir = Path(__file__).parent / 'parquet' / 'k0_search'


    for epoch in range(epochs):
        print(f"Epoch - {epoch}")
        random_seed = int(time.time())
        random.seed(random_seed)
        np.random.seed(random_seed)


        for name in names:


            start_run = time.time()
            iteration_counter = 0

            best_actual_fitness = np.inf
            best_actual_x = 0

            best_global_fitness = np.inf
            best_global_step = 0

            best_fitness_per_epoch = []
            best_x_per_epoch = []
            times_elapsed_per_epoch = []

            patience = max_patience

            lr = ng.p.Scalar(lower=5e-5, upper=6e-4)
            parametrization = ng.p.Instrumentation(lr)

            initial_k0 = estimate_k0(template)[0][0]
            delta = initial_k0 * 0.10
            iter_folder = p_out / 'initial'
            iter_folder.mkdir(parents=True, exist_ok=True)

            # introduce controlled variation to initial k0
            init_pop = [initial_k0]
            for _ in range(population_size - 1):
                init_pop.append(initial_k0 + random.uniform(-delta, delta))

            print(f"Estimated initial k0: {init_pop}")



            start_time_epoch = time.time()

            fitness = get_mean_fitness(template,init_pop,k_validation, file_input, cosmica_path, iter_folder, lis_loader, metric_fn, exp_data, initial_k0)

            evaluated_k0 = []
            evaluated_loss = []

            init_params = []

            for initial_k0, fit in zip(init_pop, fitness):
                evaluated_k0.append(float(initial_k0))
                evaluated_loss.append(fit)
                init_param = parametrization.spawn_child()
                init_param.value = ((initial_k0,), {})
                init_params.append(init_param)

                if fit < best_global_fitness:
                    best_global_fitness = fit
                    best_global_step = initial_k0
                    best_actual_fitness = best_global_fitness
                    best_actual_x = initial_k0

            best_fitness_per_epoch.append(best_actual_fitness)

            best_x_per_epoch.append(best_actual_x)

            optim = ng.optimizers.registry[name](
                parametrization=parametrization,
                budget=population_size,
                num_workers=population_size
            )

            for initial_k0, fit in zip(init_params, fitness):
                optim.tell(initial_k0, fit)

            times_elapsed_per_epoch.append(time.time() - start_time_epoch)

            for iteration in range(n_iterations):

                start_time_epoch = time.time()

                best_actual_fitness = np.inf
                best_actual_x = 0

                iteration_counter +=1

                k0_list = [optim.ask() for _ in range(population_size)]
                print(f"Current k0 list: {[k0.value[0][0] for k0 in k0_list]}")

                iter_folder = p_out / f'iteration_{iteration}'
                iter_folder.mkdir(parents=True, exist_ok=True)

                print(f"\nIteration {iteration + 1}")

                params = [float(k0.value[0][0]) for k0 in k0_list]


                fitness = get_mean_fitness(template,params,k_validation, file_input, cosmica_path, iter_folder, lis_loader, metric_fn, exp_data, params)


                for k0, fit in zip(k0_list, fitness):
                    evaluated_k0.append(float(k0.value[0][0]))
                    evaluated_loss.append(fit)
                    optim.tell(k0, fit)

                    if fit < best_actual_fitness:
                        best_actual_fitness = fit
                        best_actual_x = k0.value[0][0]
                        if best_actual_fitness < best_global_fitness:
                            best_global_fitness = best_actual_fitness
                            best_global_step = k0.value[0][0]

                best_fitness_per_epoch.append(best_actual_fitness)
                best_x_per_epoch.append(best_actual_x)

                if len(best_fitness_per_epoch) >= 2:
                    print(best_fitness_per_epoch[-2])
                    print(best_fitness_per_epoch[-1])

                    improvement = best_fitness_per_epoch[-2] - best_fitness_per_epoch[-1]
                    if improvement < min_improvement:
                        patience -= 1
                        print(f"No significant improvement ({improvement:.3e}). Remaining patience: {patience}")
                        if patience == 0:
                            print(f"Early stopping: improvement less than {min_improvement}")
                            break
                    else:
                        patience = max_patience

                times_elapsed_per_epoch.append(time.time() - start_time_epoch)


            best_params[name] = {
                "best_x": best_global_step,
                "best_loss": best_global_fitness,
                "steps": evaluated_k0,
                "corr_loss": evaluated_loss,
                "seed": random_seed,
                "time_elapsed": time.time() - start_run ,
                "algorithm": name,
                "iterations": iteration_counter,
                "n_part": n_part,
                "experimental_data_path": sim.experimental_data_path,
                "eval_metric": metric_fn.__name__,
                "best_per_epoch":best_fitness_per_epoch,
                "best_k0_per_epoch":best_x_per_epoch,
                "max_iterations":n_iterations,
                "population":population_size,
                "k_validation":k_validation,
                "times_elapsed_per_epoch" : times_elapsed_per_epoch,
                "version":"v2"
            }

        save_output_in_parquet(parquet_dir,best_params)


def save_output_in_parquet(path_file_parquet_store_res: Path, best_params: Dict):

    os.makedirs(os.path.dirname(str(path_file_parquet_store_res)), exist_ok=True)

    df_new = pd.DataFrame.from_dict(best_params, orient='index').reset_index(drop=True)

    if os.path.exists(path_file_parquet_store_res):
        df_existing = pd.read_parquet(path_file_parquet_store_res)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_parquet(path_file_parquet_store_res, index=False, engine="pyarrow")



def main():

    data_dir = Path(__file__).parent.parent / 'data'

    cosmica_path = [sys.executable, Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / 'launch_docker.py']

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


    names = ["CMA","PSO","DE","TwoPointsDE","TBPSA","RandomSearch"]

    run_optimization(
        sim=sim,
        data_dir=data_dir,
        cosmica_path=cosmica_path,
        p_out=p_out,
        lis_loader=lis_loader,
        names=names,
        population_size=10,
        n_iterations=50,
        epochs=1,
        n_part=4096,
        max_patience=10,
        min_improvement = 0.00001,
        k_validation = 4
    )



if __name__ == "__main__":
    main()