import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Optional, Callable, List, Dict
from fastparquet import write
from test.lib.fstpso_ask_tell.fstpso import FuzzyPSO
from test.lib.optimizer_lib.optimizer import Optimizer, OptimizerQueue


from cmaes import CMA
import nevergrad as ng
import pandas as pd

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

        process = subprocess.run(command,
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
                k0=[np.array(k0 + [0.0] * (n_reg - 1)) for k0 in k0s],
            ),
        ),
    )


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
    cosmica_path = Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / 'exefiles' / 'Cosmica'

    sim = SimulationExperimentItem(
        name='search',
        ions=[IONS.get('proton')],
        period=(20180929, 20181025),
        sources=(np.array([1.0]), np.array([1.5707963267948966]), np.array([0.0])),
        experimental_data_path='Rigidity_Proton_AMS-02_PRL1272021271102_20180929_20181025.dat',
    )

    template = base_input(data_dir, sim, rnd=42, n_part=4096)
    p_exp = data_dir / 'experimental' / sim.experimental_data_path
    exp_data = ExperimentalData.from_data(p_exp, (2, 3, 4, 5), rig_range=(0, 11))
    parquet_dir = Path(__file__).parent / 'parquet' / 'k0_search_v2'
    p_out = data_dir / 'search' / 'output'
    p_lis = data_dir / 'LIS_Default2020_Proton'
    lis_loader = LisLoader(p_lis)

    N_PART = 4096

    def get_results(k0_list,random_seed=42):
        template = base_input(data_dir, sim, rnd=random_seed, n_part=N_PART)
        inpt = generate_input(template, k0_list)
        iter_folder = p_out / 'initial'
        iter_folder.mkdir(parents=True, exist_ok=True)

        out = run_cosmica(inpt, cosmica_path, iter_folder / 'log_initial.log', iter_folder, cuda_devices='1')
        if out is None:
            raise RuntimeError(f"Cosmica run failed for initial k0={k0_list}")
        return out.modulate(lis_loader)


    def get_initial_k0():
        return estimate_k0(inpt=template)[0][0]

    #  ----------- FPSO ------------

    space_dim = 1
    LOWER_BOUND =5e-5
    UPPER_BOUND =6e-4
    MAX_ITERATIONS = 3
    MAX_ITER_WO_NEW_GLOBAL_BEST = 10
    POPULATION_SIZE = 2
    initial_k0_value = get_initial_k0()

    optimizer_fuzzy_pso = FuzzyPSO()
    optimizer_fuzzy_pso.set_search_space([[LOWER_BOUND,UPPER_BOUND]]*space_dim)
    optimizer_fuzzy_pso.InitCreateParticles(POPULATION_SIZE, space_dim)

    optimizer_fuzzy_pso._prepare_for_optimization(max_iter=MAX_ITERATIONS,max_iter_without_new_global_best=MAX_ITER_WO_NEW_GLOBAL_BEST,  max_FEs = None)

    optimizer_FUZZYPSO = Optimizer(optimizer=optimizer_fuzzy_pso,name="Fuzzy-PSO",eval_metric=metrics.mean_relative_error,
                                   save_results_ended=True,external_stopping_criteria=optimizer_fuzzy_pso.TerminationCriterion,
                                   parquet_dir=parquet_dir,max_iterations=MAX_ITERATIONS,budget=POPULATION_SIZE)
    # --------------- DE AND CMA -------

    vector_param = ng.p.Array(shape=(1,))
    print(initial_k0_value)
    vector_param.value = [initial_k0_value]  # must provide a default value
    vector_param.set_bounds(lower=5e-5, upper=6e-4)

    parametrization = ng.p.Instrumentation(vector_param)
    optim = ng.optimizers.registry["CMA"](
        parametrization=parametrization,
        budget=POPULATION_SIZE,
        num_workers=POPULATION_SIZE
    )


    optim_DE = ng.optimizers.registry["DE"](
        parametrization=parametrization,
        budget=POPULATION_SIZE,
        num_workers=POPULATION_SIZE
    )
    optimizer_CMA = Optimizer(optimizer=optim,name="CMA",min_increment=1e-4,max_iterations=1000,eval_metric=metrics.mean_relative_error,
                              save_results_ended=True,max_size_best_candidates=10,budget=1000,parquet_dir=parquet_dir)


    delta = initial_k0_value * 0.10
    # generate initial population
    # must work with 1D vector
    init_pop = [[initial_k0_value]]
    for _ in range(POPULATION_SIZE - 1):
        init_pop.append([initial_k0_value + random.uniform(-delta, delta)])


    optim_queue = OptimizerQueue(optimizers_list=[optimizer_FUZZYPSO])


    K_VALIDATION = 1

    optim_queue.start(epochs=1,
                      procedure=get_results,
                      initial_value_parameter=init_pop,
                      generate_random_seed_each_epoch=True,
                      real_data=exp_data,
                      experimental_data_str = "Rigidity_Proton_AMS-02_PRL1272021271102_20180929_20181025.dat",
                      k_validation = K_VALIDATION)


if __name__ == "__main__":
    main()