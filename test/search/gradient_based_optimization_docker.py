import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Callable, List, Dict

import nevergrad.optimization
import pandas as pd
from babel.plural import range_list_node
from fastparquet import write
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test.lib.physics_utils import FluxVec, RigidityVec
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
# Version 3.0: multiGPU bug fix
#
#######################################################################

VERSION = "v3"

file_input = Path(__file__).parent / 'inputs' / 'test.yaml'

def run_cosmica(inpt, base_command: list[str | Path] | str | Path, log_file: Path, output_dir: Path,
                cuda_devices: str = '1,3,5,7') -> Optional[
    SimulationOutput]:
    try:

        with open(file_input, 'w') as f:
            yaml.dump(inpt.to_dict(), f, Dumper=yaml.Dumper)

        command = list(map(str, base_command)) if isinstance(base_command, list) else [str(base_command)]


        command += [
            "-v",
            "debug",
            "-i", str(file_input),
            "--stdout",
            "--log_file",
            str(log_file),
            "-o",
            str(output_dir) + '/',
            ]

        print(f"Executing command: {' '.join(command)}")


        #process = subprocess.run(command, capture_output=True, text=True)
        process = subprocess.run(command, env={'CUDA_VISIBLE_DEVICES': cuda_devices}, capture_output=True, text=True)

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
        #lower,upper = experimental_data.limits

        #for i,flux in enumerate(result.flux):
        #if lower[i] < flux < upper[i]:
        #       # if flux lies inside uncertainty interval
                # then it's evaluated as the exact value
        #       result.flux[i] = experimental_data.flux[i]

        if metric_fn is not None:
            losses.append(metric_fn(result, experimental_data))
        else:
            losses.append(float(np.sqrt(np.square(np.mean(result.flux - experimental_data.flux)))))

    return losses


def base_input(data_dir: Path, sim: SimulationExperimentItem, rnd: int = 42, n_part: int = 4096,rig_range=(0,11)) -> SimulationInput:
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
    experimental_data = ExperimentalData.from_data(p_exp, (2, 3, 4, 5),rig_range)
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


def generate_input(base: SimulationInput, k0s: list[float],rigidities:RigidityVec = None,random_seed = None) -> SimulationInput:
    """
        Generate a new SimulationInput based on the base input and a list of k0 values.
        Args:
            base (SimulationInput): The base simulation input.
            k0s (list[float]): A list of k0 values to set in the dynamic parameters.
        Returns:
            SimulationInput: A new SimulationInput with updated dynamic parameters.
            :param rigidities:
    """
    n_reg = len(base.static.heliosphere.v0)

    base = base._replace(
        dynamic=SimulationInput.DynamicParameters(
            SimulationInput.DynamicParameters.DynamicHeliosphere(
                k0=[np.array([k0] + [0.0] * (n_reg - 1)) for k0 in k0s],
            ),
        )
    )

    # Now, update the new 'base' object again if needed
    if rigidities is not None:
        base = base._replace(rigidities=rigidities)

    # And again for the random seed
    if random_seed is not None:
        base = base._replace(random_seed=random_seed)

    return base


def save_output_in_parquet(path_file_parquet_store_res: Path, best_params: Dict):

    os.makedirs(os.path.dirname(str(path_file_parquet_store_res)), exist_ok=True)

    df_new = pd.DataFrame.from_dict(best_params, orient='index').reset_index(drop=True)

    if os.path.exists(path_file_parquet_store_res):
        df_existing = pd.read_parquet(path_file_parquet_store_res)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        print("non exists")
        df_combined = df_new

    df_combined.to_parquet(path_file_parquet_store_res,  index=False, engine="pyarrow")

def generate_initial_entry(number_of_entries:int,generation_entry_method:str|Callable,min_max_interval:tuple):

    lower_bound , upper_bound = min_max_interval

    if isinstance(generation_entry_method,str):
        if generation_entry_method == "uniform":
            return [random.uniform(a=lower_bound,b=upper_bound) for _ in range(number_of_entries)]
        elif generation_entry_method == "log":
            list = []
            for i in range(number_of_entries):
                # WARNING
                # there is no check if boundaries are < 0
                minimo = math.log(lower_bound)
                massimo = math.log(upper_bound)
                list.append(minimo+(massimo-minimo)*random.random())
            return list


    elif isinstance(generation_entry_method,Callable):
        return [ generation_entry_method(a=lower_bound,b=upper_bound) for _ in range(number_of_entries)]

    # dummy method if nothing correspond to sampling methods

    return [(i - lower_bound) % upper_bound for i in range(number_of_entries)]



def rigidity_analysis(sampling: int = 6,
                      min_max_interval: tuple = [-5, 5],
                      number_of_entries: int = 10,
                      generation_entry_method: str | Callable = "uniform",
                      random_seed: int = 42,
                      sim: SimulationExperimentItem = None,
                      data_dir: Path = "",
                      cosmica_path: list[str | Path] | str | Path = "",
                      p_out: Path = "",
                      p_exp:Path = "",
                      parquet_dir:Path = "",
                      file_input:Path ="",
                      lis_loader: LisLoader = None,
                      max_iter = 100,
                      rig_range:tuple= (0,1.816),
                      n_part = 16384):

    experimental_data = ExperimentalData.from_data(p_exp, (2, 3, 4, 5), rig_range=rig_range)
    template = base_input(data_dir, sim, rnd=random_seed, n_part=n_part)

    rigidities = experimental_data.rig_flux.rigidity
    num_rigidities = len(rigidities)

    random_seed_i = random_seed
    for turn in range(sampling):

        results_for_this_turn = {}

        print(f"--- Starting Sampling Turn {turn + 1}/{sampling} ---")

        # This dictionary will hold all results for the current turn


        random_seed_i += turn

        random.seed(random_seed_i)
        np.random.seed(random_seed_i)

        initial_candidates = generate_initial_entry(number_of_entries=number_of_entries,
                                                    generation_entry_method=generation_entry_method,
                                                    min_max_interval=min_max_interval)
        print(f"{initial_candidates}")
        for bitmask in range(1, 1 << num_rigidities):

            selected_rigidities = RigidityVec([rig for i, rig in enumerate(rigidities) if bitmask & (1 << i)])

            experimental_data_tmp = ExperimentalData(rigidity=selected_rigidities,
                                                     flux = FluxVec([flux for i, flux in enumerate(experimental_data.flux) if bitmask & (1 << i)]))

            iter_folder = p_out / f'turn_{turn}_bitmask_{bitmask}'
            iter_folder.mkdir(parents=True, exist_ok=True)

            print(f"  Testing Bitmask {bitmask:0{num_rigidities}b} with {len(selected_rigidities)} rigidities...")

            def f(k0):
                inpt = generate_input(template, np.exp(k0), rigidities=selected_rigidities,random_seed=random_seed)

                with open(file_input, 'w') as f:
                    yaml.dump(inpt.to_dict(), f, Dumper=yaml.Dumper)

                sim_results = run_cosmica(inpt, base_command=cosmica_path, log_file=iter_folder / 'log.log',
                                          output_dir=iter_folder)
                results = sim_results.modulate(lis_loader)

                fitness = fitness_fn(results=results, experimental_data=experimental_data_tmp, metric_fn=metrics.rmse)[0]

                print(f"k0:{math.exp(k0)} RMSE{fitness}")
                #gradient =

                return fitness#, gradient


            log_bounds = [(math.log(5e-5), math.log(6e-4))]

            for i, k0_candidate in enumerate(initial_candidates):
                res = minimize(f, k0_candidate, method="L-BFGS-B", options={"maxiter": max_iter},bounds=log_bounds)

                results_for_this_turn[f"{bitmask}-{turn}-{i}"] = {
                    "turn": turn,                               # From outer loop
                    "seed": random_seed_i,                      # From outer loop
                    "bitmask": f"{bitmask:0{num_rigidities}b}",  # From middle loop
                    "selected_rigidities": list(selected_rigidities), # From middle loop
                    "initial_k0": k0_candidate,                 # From inner loop
                    "optimized_k0": res.x[0] if res.x.size > 0 else None,
                    "final_fitness": res.fun,
                    "success": res.success,
                    "message": res.message,
                    "n_iterations": res.nit,
                    "g(X)" :  "log"
                }

        print(results_for_this_turn)
        save_output_in_parquet(parquet_dir / "rigidity_analysis_data", results_for_this_turn)

        print(f"--- Turn {turn + 1} complete. Results saved to {parquet_dir} ---\n")



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

    p_exp = data_dir / 'experimental' / sim.experimental_data_path

    parquet_dir = Path(__file__).parent / 'parquet'
    parquet_dir.mkdir(parents=True, exist_ok=True)


    if not file_input.exists():
        file_input.touch()

    rigidity_analysis(sampling=6,
                      min_max_interval=(5e-5,6e-4),
                      number_of_entries=6,
                      generation_entry_method="log",
                      random_seed=42,
                      sim = sim,
                      data_dir = data_dir,
                      cosmica_path= cosmica_path,
                      p_out = p_out,
                      p_exp = p_exp,
                      parquet_dir = parquet_dir,
                      file_input = file_input,
                      lis_loader = lis_loader,
                      max_iter=100,
                      rig_range = (0,1.816))







if __name__ == "__main__":
    main()