import math
import os
import random
import sys
import time
from copy import deepcopy
from email.policy import default
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple
import click

import nevergrad.optimization
import pandas as pd
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


def run_cosmica(inpt, base_command: list[str | Path] | str | Path, log_file: Path, output_dir: Path,
                cuda_devices: str = '0,1') -> Optional[
    SimulationOutput]:
    try:

        #with open(file_input, 'w') as f:
        #    yaml.dump(inpt.to_dict(), f, Dumper=yaml.Dumper)

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


        #process = subprocess.run(command, capture_output=True, text=True)
        process = subprocess.run(command, env={'CUDA_VISIBLE_DEVICES': cuda_devices}, capture_output=True, text=True)

        if process.returncode != 0:
            print(process.returncode)
            print("QUI")
            return None


        output_dir.mkdir(parents=True, exist_ok=True)
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
        print("QUI")
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
    experimental_data = ExperimentalData.from_data(p_exp, (0,1,2, 3),rig_range)
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


def generate_input(base: SimulationInput, k0s: List[List[float]],rigidities:RigidityVec = None,random_seed = None,is_k0_log_base = False) -> SimulationInput:
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


    list_k0 = []


    for k0 in k0s:
        # abbiamo k0 perpendicolare e il rapporto tra il k0_perpendicolare/K0_parallelo = k0[1]
        copy_k0 = deepcopy(k0)
        if len(k0) == 2:
            if is_k0_log_base:
                copy_k0[1] = copy_k0[0] - copy_k0[1]
            else:
                copy_k0[1] = copy_k0[0] / copy_k0[1]

        list_k0.append([copy_k0] + [[0.0, 0.0]] * (n_reg - 1))


    base = base._replace(
        dynamic=SimulationInput.DynamicParameters(
            SimulationInput.DynamicParameters.DynamicHeliosphere(
                k0=list_k0,
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

def generate_initial_entry(number_of_entries:int,generation_entry_method:str|Callable,min_max_interval:List[tuple]):
    if isinstance(generation_entry_method,str):
        if generation_entry_method == "uniform":

            def uniform(n_dim_interval:List[tuple]):
                return [random.uniform(a=interval[0],b=interval[1]) for interval in n_dim_interval]
            return [ uniform(min_max_interval) for _ in range(number_of_entries)]

        elif generation_entry_method == "log":

            def log_gen(n_dim_interval:List[tuple]):

                multi_dim_entity = []

                for interval in n_dim_interval:
                    minimo = math.log(interval[0])
                    massimo = math.log(interval[1])
                    multi_dim_entity.append(minimo+(massimo-minimo)*random.random())
                return multi_dim_entity

            # WARNING
            # there is no check if boundaries are < 0
            return [log_gen(min_max_interval) for _ in range(number_of_entries)]

    elif isinstance(generation_entry_method,Callable):
        return [ generation_entry_method(min_max_interval) for _ in range(number_of_entries)]

    # dummy method if nothing correspond to sampling methods
    def dummy(n_dim_interval,i):
        return [(i - interval[0]) % interval[1] for interval in n_dim_interval]

    return [dummy(min_max_interval,i) for i in range(number_of_entries)]

def get_mean_fitness(template,init_pop,k_validation, file_input_:Path, cosmica_path, iter_folder, lis_loader, metric_fn, exp_data, initial_k0,rigidities,random_seed):

    all_fluxes = fluxes_mean(cosmica_path, file_input_, init_pop, initial_k0, iter_folder, k_validation, lis_loader,
                             random_seed, rigidities, template)

    fitness = fitness_fn(all_fluxes[0], exp_data, metric_fn=metric_fn)[0]

    return fitness,all_fluxes[0]

def get_list_fitness(template,init_pop,k_validation, file_input_:Path, cosmica_path, iter_folder, lis_loader, metric_fn, exp_data, initial_k0,rigidities,random_seed):

    all_fluxes = fluxes_mean(cosmica_path, file_input_, init_pop, initial_k0, iter_folder, k_validation, lis_loader,
                             random_seed, rigidities, template)

    fitness = fitness_fn(all_fluxes[0], exp_data, metric_fn=metric_fn)
    return fitness,all_fluxes[0]



def fluxes_mean(cosmica_path, file_input_, init_pop, initial_k0, iter_folder, k_validation, lis_loader, random_seed,
                rigidities, template)-> List[List[ModulationResult]]:
    all_fluxes = []
    starting_seed = random_seed
    for i in range(k_validation):

        inpt = generate_input(template, init_pop, rigidities=rigidities, random_seed=starting_seed + i)

        with open(file_input_, 'w') as f:
            yaml.dump(inpt.to_dict(), f, Dumper=yaml.Dumper)

        print(f"K-{i} run")
        out = run_cosmica(file_input_, cosmica_path, iter_folder / 'log_initial.log', iter_folder)

        if out is None:
            raise RuntimeError(f"Cosmica run failed for initial k0={initial_k0}")

        results = out.modulate(lis_loader)


        print(f"Risultati simulazion {results}")

        all_fluxes.append(results)
    flux_matrix = []
    for row in all_fluxes:
        flux_matrix.append([mod_result.flux for mod_result in row])
    mean_flux = np.mean(flux_matrix, axis=0)

    # using first row since lis and rig are the same for each row[i]
    # mean was evaluated per column index

    for i, res in enumerate(all_fluxes[0]):
        all_fluxes[0][i] = res._replace(flux=FluxVec(deepcopy(mean_flux[i])))

    return all_fluxes


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
                      rig_range:dict=None,
                      n_part = 16384,
                      **kwargs):

    if rig_range is None:
        rig_range = {"helium":(0, 2.83),"proton":(0, 1.816)}


    particle_to_test = kwargs.get('particle_to_test',['helium','proton'])
    particle_name = kwargs.get("particle_name","proton")

    print(f"rig_range {rig_range}")
    print(f"particle_to_test {particle_to_test}")
    print(f"particle_name {particle_name}")

    if particle_name not in particle_to_test:
        return


    rig_range_interval = rig_range.get(particle_name)

    experimental_data = ExperimentalData.from_data(p_exp, (0, 1, 2, 3), rig_range=rig_range_interval)
    template = base_input(data_dir, sim, rnd=random_seed, n_part=n_part)

    rigidities = experimental_data.rig_flux.rigidity
    num_rigidities = len(rigidities)


    parquet_file_name = kwargs.get('parquet_name','rigidity_analysis_data')


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
            if not iter_folder.exists():
                print(f"Warning: folder not created: {iter_folder}")

            print(f"  Testing Bitmask {bitmask:0{num_rigidities}b} with {len(selected_rigidities)} rigidities...")

            steps = []

            def f(k0):

                steps = []

                fitness,fluxes = get_mean_fitness(template=template,init_pop = np.exp(k0),
                                           k_validation = 1, file_input_ = file_input,
                                           cosmica_path = cosmica_path, iter_folder = iter_folder,
                                           lis_loader = lis_loader,
                                           metric_fn = metrics.rmse,
                                           exp_data = experimental_data_tmp,
                                           initial_k0 = np.exp(k0),rigidities=selected_rigidities,
                                           random_seed=random_seed)
                steps.append([k0,fitness])

                print(f"k0:{math.exp(k0)} RMSE{fitness}")
                return fitness


            log_bounds = [(math.log(5e-5), math.log(6e-4))]

            for i, k0_candidate in enumerate(initial_candidates):
                try:

                    res = minimize(f, k0_candidate, method="L-BFGS-B", options={"maxiter": max_iter},bounds=log_bounds)

                    results_for_this_turn[f"{bitmask}-{turn}-{i}"] = {
                        "turn": turn,
                        "seed": random_seed_i,
                        "bitmask": f"{bitmask:0{num_rigidities}b}",
                        "selected_rigidities": list(selected_rigidities),
                        "initial_k0": k0_candidate,
                        "optimized_k0": res.x[0] if res.x.size > 0 else None,
                        "final_fitness": res.fun,
                        "success": res.success,
                        "message": res.message,
                        "n_iterations": res.nit,
                        "g(X)" :  "log",
                        "data_dir" : str(p_exp),
                        "steps" : steps
                    }
                except Exception as e:
                    print(f"Optimization failed for candidate {i} in turn {turn}: {e}")

                    results_for_this_turn[f"{turn}-{i}"] = {
                        "turn": turn,
                        "seed": random_seed_i,
                        "bitmask": f"{bitmask:0{num_rigidities}b}",
                        "selected_rigidities": list(selected_rigidities),
                        "initial_k0": k0_candidate,
                        "optimized_k0": 0,
                        "final_fitness": 0,
                        "success": False,
                        "message": f"Optimization failed for candidate {i} in turn {turn}: {e}",
                        "n_iterations": 0,
                        "g(X)" :  "log",
                        "data_dir" : str(p_exp),
                        "steps":steps
                    }
                finally:
                    steps = []

        print(results_for_this_turn)
        save_output_in_parquet(parquet_dir / parquet_file_name, results_for_this_turn)

        print(f"--- Turn {turn + 1} complete. Results saved to {parquet_dir} ---\n")


def run_all_files(procedure:Callable,**kwargs):

    selected_forbush = kwargs.get('forbush_name','FD006')
    docker_launcher = kwargs.get('launch_docker','launch_docker.py')

    print(f"{docker_launcher}")
    print(f"{selected_forbush}")

    root = Path(__file__).parent.parent / 'data' / 'forbush'
    inputs_root = root / 'inputs' / selected_forbush
    experimental_root = root / 'experimental'

    yaml_files = sorted(inputs_root.rglob("*.yaml"))
    print(f"Found {len(yaml_files)} .yaml files to process")

    data_dir = Path(__file__).parent.parent / 'data'

    env_docker = kwargs.get('env_docker',True)
    if env_docker:

        cosmica_path = [sys.executable, Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / docker_launcher]

        #cosmica_path = [
        #    sys.executable,
        #
        #    Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / 'launch_docker_unive.py'
        #]
    else:
        cosmica_path = Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / 'exefiles' / 'Cosmica'

    p_out = data_dir / 'search' / 'output'
    p_lis = data_dir / 'LIS_Default2020_Proton'
    lis_loader = LisLoader(p_lis)

    # parquet output dir
    parquet_dir = Path(__file__).parent / 'parquet'
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # loop over each input file
    for idx, file_input_ in enumerate(yaml_files, start=1):
        # extract the date from folder name: AMS-02Daily_20110801 → 20110801

        file_input = file_input_

        date_str = file_input.parent.name.split("_")[1]
        type_name = file_input.name.split('_')[0]

        if type_name == "helium":
            # matching experimental file
            experimental_data_path = experimental_root / f"Rigidity_AMS-02Daily_{date_str}_Helium.dat"
        else:
            experimental_data_path = experimental_root / f"Rigidity_AMS-02Daily_{date_str}_Proton.dat"

        if not experimental_data_path.exists():
            print(f"Skipping {file_input}, missing experimental file {experimental_data_path}")
            continue

        print(f"\n[{idx}/{len(yaml_files)}] Running optimization:")
        print(f"  input = {file_input}")
        print(f"  exp   = {experimental_data_path}")

        # simulation setup
        sim = SimulationExperimentItem(
            name='search',
            ions=[IONS.get(type_name)],
            period=(int(date_str), int(date_str)+1),
            sources=(np.array([1.0]), np.array([1.5707963267948966]), np.array([0.0])),
            experimental_data_path=str(experimental_data_path)
        )

        # full experimental path
        p_exp = data_dir / 'experimental' / sim.experimental_data_path

        # safeguard in case input file doesn’t exist
        if not file_input.exists():
            file_input.touch()


        procedure(sim=sim,
                  data_dir=data_dir,
                  cosmica_path=cosmica_path,
                  p_out=p_out,
                  p_exp=p_exp,
                  parquet_dir=parquet_dir,
                  file_input=file_input,
                  lis_loader=lis_loader,
                  particle_name=type_name,
                  **kwargs)

def optimize(sampling: int = 6,
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
             max_iter = 20,
             n_part = 16384,
             methods = ["L-BFGS-B","Powell"],
             **kwargs):

    experimental_data = ExperimentalData.from_data(p_exp, (0, 1, 2, 3))
    template = base_input(data_dir, sim, rnd=random_seed, n_part=n_part)

    parquet_file_name = kwargs.get('parquet_name','gradient_d_res')

    rigidities = experimental_data.rig_flux.rigidity

    random_seed_i = random_seed

    initial_candidates: List|None = kwargs.get('initial_candidates',None)

    print(f"Candidati : {initial_candidates}")

    for method in methods:

        initial_candidates_i = initial_candidates

        for turn in range(sampling):

            results_for_this_turn = {}

            print(f"--- Starting Sampling Turn {turn + 1}/{sampling} ---")

            random_seed_i += turn
            random.seed(random_seed_i)
            np.random.seed(random_seed_i)

            if initial_candidates_i is None or turn > 1:
                initial_candidates = generate_initial_entry(number_of_entries=number_of_entries,
                                                        generation_entry_method=generation_entry_method,
                                                        min_max_interval=min_max_interval)
            else:
                print("Non generati")

                if len(initial_candidates) != sampling:
                    print("Errore passati candidati minori rispetto ai turni")
                    return

            print(f"{initial_candidates}")

            experimental_data_tmp = ExperimentalData(rigidity=rigidities,flux = experimental_data.flux)
            iter_folder = p_out / f'turn_{turn}'
            iter_folder.mkdir(parents=True, exist_ok=True)
            if not iter_folder.exists():
                print(f"Warning: folder not created: {iter_folder}")


            steps_k0 = []
            fitness_k0 = []

            def f(k0):

                k0 = k0[0]
                val = np.exp(k0)


                fitness,fluxes = get_mean_fitness(template=template,init_pop = [val],
                                               k_validation = 1, file_input_ = file_input,
                                               cosmica_path = cosmica_path, iter_folder = iter_folder,
                                               lis_loader = lis_loader,
                                               metric_fn = metrics.rmse,
                                               exp_data = experimental_data_tmp,
                                               initial_k0 = val,rigidities=rigidities,
                                               random_seed=random_seed_i)
                print(f"k0:{val} RMSE{fitness}")
                steps_k0.append(val)
                fitness_k0.append(fitness)


                return fitness

            log_bounds = [(math.log(5e-5), math.log(6e-4))]
            for i, k0_candidate in enumerate(initial_candidates):

                steps_k0 = []
                fitness_k0 = []

                try:

                    res = minimize(f, k0_candidate, method=method, options={"maxiter": max_iter},bounds=log_bounds)

                    results_for_this_turn[f"{turn}-{i}"] = {
                        "turn": turn,
                        "seed": random_seed_i,
                        "initial_k0": k0_candidate,
                        "optimized_k0": res.x[0] if res.x.size > 0 else None,
                        "final_fitness": res.fun,
                        "success": res.success,
                        "message": res.message,
                        "n_iterations": res.nit,
                        "g(X)" :  "log",
                        "data_dir" : str(p_exp),
                        "method" : method,
                        "steps":steps_k0,
                        "fitness_each_step" : fitness_k0
                    }

                    print("numero di iterazioni eseguite")
                    print(res.nit)
                except Exception as e:
                    print(f"Optimization failed for candidate {i} in turn {turn}: {e}")

                    results_for_this_turn[f"{turn}-{i}"] = {
                        "turn": 0,
                        "seed": random_seed_i,
                        "initial_k0": k0_candidate,
                        "optimized_k0": 0,
                        "final_fitness": 0,
                        "success": False,
                        "message": f"Optimization failed for candidate {i} in turn {turn}: {e}",
                        "n_iterations": 0,
                        "g(X)" :  "log",
                        "data_dir" : str(p_exp),
                        "method" : method,
                        "steps":steps_k0,
                        "fitness_each_step" : fitness_k0
                    }


            save_output_in_parquet(parquet_dir / parquet_file_name, results_for_this_turn)

            print(f"--- Turn {turn + 1} complete. Results saved to {parquet_dir} ---\n")



def launch_simulation(k0_array:list,output_file,bitmask:list,launch_docker,exp_datapath,forbush,random_seed = 42,n_part = 65536,**kwargs):
    print(k0_array)

    particle_to_test = kwargs.get('particle_to_test',['proton'])
    root = Path(__file__).parent.parent / 'data' / 'forbush'


    # extract forbush

    inputs_root = root / 'inputs' / forbush
    experimental_root = root / 'experimental'

    p = Path(exp_datapath)
    parts = p.parts
    try:
        # ho usato due macchine diverse per i test, alcuni puntano al percoso di vm diverse
        idx = parts.index('experimental')
        rest_parts = parts[idx+1:]
        rest = Path(*rest_parts) if rest_parts else Path()
        print(f"Da {exp_datapath}")
        exp_datapath = (Path(experimental_root) / rest).as_posix()
        print(f"A percorso {exp_datapath}")

    except ValueError:
        None
        print("Errore nella creazione del path, experimental non presente")



    data_dir = Path(__file__).parent.parent / 'data'

    env_docker = kwargs.get('env_docker',True)
    if env_docker:
        cosmica_path = [sys.executable, Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / launch_docker]
    else:
        cosmica_path = Path(__file__).parent.parent.parent / 'Cosmica_V8-speedtest' / 'exefiles' / 'Cosmica'

    p_out = data_dir / 'search' / 'output'
    p_lis = data_dir / 'LIS_Default2020_Proton'
    lis_loader = LisLoader(p_lis)

    # parquet output dir
    parquet_dir = Path(__file__).parent / 'parquet'
    parquet_dir.mkdir(parents=True, exist_ok=True)

    exp_datapath = Path(exp_datapath)

    filename = exp_datapath.name

    fname = Path(filename).stem

    parts = fname.split("_")

    # Extract the pieces
    date = parts[2]
    element = parts[3]

    if element.lower() not in particle_to_test:
        print(f"Particella {element} non da testare")
        return

    # simulation setup
    sim = SimulationExperimentItem(
        name='search',
        ions=[IONS.get(element.lower())],
        period=(int(date), int(date)+1),
        sources=(np.array([1.0]), np.array([1.5707963267948966]), np.array([0.0])),
        experimental_data_path=str(exp_datapath)
    )

    # full experimental path
    base_input_name = date + "_" + date + "_4096_1_42.yaml"

    subfolder = (
        f"helium_heli3_{base_input_name}"
        if element.lower() == "helium"
        else f"proton_deuteron_{base_input_name}"
    )

    # Build the full path
    file_input = inputs_root / f"AMS-02Daily_{date}" / subfolder
    iter_folder = p_out / f'godness_estimation'
    experimental_data = ExperimentalData.from_data(exp_datapath, (0, 1, 2, 3))
    template = base_input(data_dir, sim, rnd=random_seed, n_part=n_part)


    rigidities = experimental_data.rig_flux.rigidity

    fitness_res = get_list_fitness(template=template,init_pop = k0_array,
                               k_validation = 1, file_input_ = file_input,
                               cosmica_path = cosmica_path, iter_folder = iter_folder,
                               lis_loader = lis_loader,
                               metric_fn = metrics.rmse,
                               exp_data = experimental_data,
                               initial_k0 = k0_array,rigidities=rigidities,
                               random_seed=random_seed)


    results_for_this_turn = {}
    turn = kwargs.get("turn","ND")

    for i,fitness in enumerate(fitness_res):

        results_for_this_turn[f"{turn}-{i}"] = {
            "turn": turn,
            "seed": random_seed,
            "tested_k0": k0_array[i],
            "final_fitness": fitness,
            "data_dir" : str(exp_datapath),
            "bitmask": bitmask[i]
            }
    save_output_in_parquet(parquet_dir / output_file, results_for_this_turn)



@click.command()
@click.option("--forbush", default="FD006", help="Which forbush event to run")
@click.option("--launch-docker", default="launch_docker.py", help="Docker launcher script")
@click.option("--optimize_flag/--no-optimize_flag", default=True, help="Run optimization or rigidity analysis")
@click.option("--outfile", help="Output file")
def start_script(forbush, launch_docker, optimize_flag,outfile):

    min_max_interval = (5e-5, 6e-4)
    number_of_entries = 6
    generation_entry_method = "log"

    if optimize_flag:


        initial_candidates = generate_initial_entry(number_of_entries=number_of_entries,
                                                    generation_entry_method=generation_entry_method,
                                                    min_max_interval=min_max_interval)


        run_all_files(procedure=optimize,
                      sampling=6,
                      min_max_interval=min_max_interval,
                      number_of_entries=number_of_entries,
                      #generation_entry_method="log",
                      random_seed=42,
                      max_iter=20,
                      forbush_name = forbush,
                      launch_docker=launch_docker,
                      parquet_name = outfile,
                      methods = ["Powell"],
                      initial_candidates = initial_candidates)
    else:
        run_all_files(procedure=rigidity_analysis,
                      sampling=6,
                      min_max_interval=(5e-5,6e-4),
                      number_of_entries=6,
                      generation_entry_method="log",
                      random_seed=42,
                      max_iter=100,
                      rig_range = {"helium":(0, 2.83),"proton":(0, 1.816)},
                      forbush_name = forbush,
                      launch_docker=launch_docker,
                      parquet_name = outfile,
                      particle_to_test = ['helium'])



if __name__ == "__main__":
    start_script()

