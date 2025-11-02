import threading
import time
import json
from pathlib import Path
from typing import Callable

from gradient_based_optimization_docker import base_input, generate_initial_entry, get_mean_fitness, \
    save_output_in_parquet, run_all_files, get_list_fitness
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


# ----- Optimizer Container -----
class OptimizerContainer:
    def __init__(self, id: int, p_exp: str,random_seed:int):
        self.id = id
        self.p_exp = p_exp

        self.steps = []
        self.fitness = []
        self.fluxes = []
        self.ended = False
        self.has_error = False

        self.pushed_result = None
        self.event = threading.Event()
        self.optimizer_result = None
        self.random_seed = random_seed

    def get_k0_to_test(self)->List | None:
        return self.steps[-1] if len(self.steps) > 0 else None

    def wait_for_result(self, k0):
        self.steps.append(k0)
        print(k0)
        # aspetta che il risultato venga pushato
        self.event.wait()
        self.event.clear()

        print(f"pushed_res {self.pushed_result}")
        fitness, fluxes = self.pushed_result
        self.fitness.append(fitness)
        self.fluxes.append(fluxes)
        self.pushed_result = None
        return fitness

    def push_result(self, fitness_flux: Tuple[float, any]):
        self.pushed_result = fitness_flux
        self.event.set()  # segnala al thread in attesa

    def start(self, init_k0, log_bounds):
        try:
            # Qui lanciamo l'ottimizzazione usando minimize o altro
            # esempio semplificato: ottimizza iterativamente aspettando risultati
            self.optimizer_result = minimize(self.wait_for_result, init_k0,
                                      method="Powell",
                                      options={"maxiter": 6},
                                      bounds=log_bounds)

            print(f"TERMINATO {self.id}")
            self.ended = True
        except Exception as e:
            print(f"Container {self.id} error: {e} , {init_k0}")
            self.has_error = True
            self.ended = True

    def is_ended(self):
        return self.ended

    def n_iteration(self):
        return len(self.steps)

    def get_res(self):

        if not self.has_error:

            print("Stampo")
            return {
                "turn": 0,
                "seed": self.random_seed,
                "initial_k0": self.steps[0],
                "optimized_k0": self.optimizer_result.x[0] if self.optimizer_result.x.size > 0 else None,
                "final_fitness": self.optimizer_result.fun,
                "success": self.optimizer_result.success,
                "message": self.optimizer_result.message,
                "n_iterations": self.optimizer_result.nit,
                "g(X)" :  "log",
                "data_dir" : self.p_exp,
                "method" : "Powell",
                "steps":self.steps,
                "fitness_each_step" : self.fitness
            }
        else:
            return {
                "turn": 0,
                "seed": self.random_seed,
                "initial_k0": [0],
                "optimized_k0": [0],
                "final_fitness": 0,
                "success": False,
                "message": f"Optimization failed for candidate {id} ",
                "n_iterations": 0,
                "g(X)" :  "log",
                "data_dir" : self.p_exp,
                "method" : "Powell",
                "steps":self.steps,
                "fitness_each_step" : self.fitness
            }


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
             n_part = 16384,
             methods = ["L-BFGS-B","Powell"],
             **kwargs):

    experimental_data = ExperimentalData.from_data(p_exp, (0, 1, 2, 3))
    template = base_input(data_dir, sim, rnd=random_seed, n_part=n_part)

    parquet_file_name = kwargs.get('parquet_name','gradient_d_res')

    rigidities = experimental_data.rig_flux.rigidity

    initial_candidates: List|None = kwargs.get('initial_candidates',None)

    final_initial_candidates =  []

    for i in range(sampling):
        for j in range(len(initial_candidates)):
            final_initial_candidates.append(initial_candidates[j])


    experimental_data_tmp = ExperimentalData(rigidity=rigidities,flux = experimental_data.flux)
    iter_folder = p_out / f'turn_{0}'
    iter_folder.mkdir(parents=True, exist_ok=True)
    if not iter_folder.exists():
        print(f"Warning: folder not created: {iter_folder}")


    class OptimizerContainerList:
        def __init__(self):
            self.container_list = []
            self.finished_optimizer = []
            self.to_do_list = []
            self.p_exp = p_exp

        def start_optimization(self, init_pop: List[List[float]], log_bounds):
            # Creazione dei container se non esistono
            self.container_list = []
            for i, k0 in enumerate(init_pop):
                optimizer_container = OptimizerContainer(id=i, p_exp=str(self.p_exp),random_seed=random_seed)
                self.container_list.append(optimizer_container)

            # Avvia i container in thread separati
            threads = []
            for i, container in enumerate(self.container_list):

                t = threading.Thread(
                    target=container.start,
                    args=(deepcopy(init_pop[i]), log_bounds)
                )
                t.start()
                threads.append(t)

            # Loop principale: simulazione unica
            active_containers : List[OptimizerContainer] = [c for c in self.container_list if not c.is_ended()]
            n_iteration = 1

            while active_containers:
                # Prendi i k0 da testare per tutti i container attivi

                to_test_k0 = [c.get_k0_to_test() for i, c in enumerate(active_containers)]
                to_test_k0 = [np.exp(k0) for k0 in to_test_k0]
                # Esegui la simulazione unica
                fitness_list, fluxes_list = get_list_fitness(
                    template=template,
                    init_pop=to_test_k0,
                    k_validation=1,
                    file_input_=file_input,
                   cosmica_path=cosmica_path,
                    iter_folder=iter_folder,
                   lis_loader=lis_loader,
                   metric_fn=metrics.rmse,
                   exp_data=experimental_data_tmp,
                   initial_k0=to_test_k0,
                   rigidities=rigidities,
                   random_seed=random_seed
                )
                #fitness_list = [12.92908532070397, 5.601976597307415, 1.3434649857662369, 2.1702433176649025, 1.2306916766847524, 1.234347892631663, 1.2289230794503565, 1.230405051052975, 1.2307812406371608, 1.2294588979317207, 1.2321314040621694, 1.21922003838464, 1.2277869059001827, 1.2295719636217788, 1.2272439575249536, 1.2322289083671303, 4.648641111311211, 5.601976597307415, 1.3434649857662369, 2.1702433176649025, 1.2306916766847524, 1.234347892631663, 1.2289230794503565, 1.230405051052975, 1.2307812406371608, 1.2294588979317207, 1.2321314040621694, 1.21922003838464, 1.2277869059001827, 1.2295719636217788, 1.229306955757998, 1.2277209521833108, 1.3434649857662369, 5.601976597307415, 2.1702433176649025, 1.2306916766847524, 1.234347892631663, 1.2289230794503565, 1.230405051052975, 1.2307812406371608, 1.2294588979317207, 1.2321314040621694, 1.21922003838464, 1.2277869059001827, 1.2295719636217788, 1.229306955757998, 1.228648769286966, 1.228058452504755, 1.2263386736790947, 1.227445412651336, 1.220980037016337, 1.222976401769953, 1.2222380892409528, 1.2262293469861099, 1.2242912946888072, 1.2222358015090034, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464, 1.21922003838464]
                # Push dei risultati ai container attivi

                for c,fitness, flux in zip(active_containers, fitness_list, fluxes_list):
                    c.push_result((fitness, flux))

                # aspetta che tutti abbiano ricevuto la fitness

                n_iteration += 1

                self.busy_wait_update(active_containers=active_containers,n_iteration=n_iteration)

                # Aggiorna la lista dei container attivi
                active_containers = [c for c in self.container_list if not c.is_ended()]

            # Attende tutti i thread
            for t in threads:
                t.join()

            # Salva i risultati in parquet
            save_results = {}
            for i, optimizer in enumerate(self.container_list):
                save_results[i] = optimizer.get_res()

            save_output_in_parquet(parquet_dir / parquet_file_name, save_results)

        def busy_wait_update(self,active_containers,n_iteration):
            updated_all = False

            while not updated_all:
                updated_all = True
                print("SONO IN ATTESA CHE TUTTI FINISCANO")

                for opt in active_containers:
                    updated_all = updated_all and ( opt.n_iteration() == n_iteration or opt.is_ended() )
                time.sleep(0.1)

    log_min_max_bounds = [(np.log(interval[0]),np.log(interval[1])) for interval in min_max_interval]
    OptimizerContainerList().start_optimization(final_initial_candidates,log_bounds=log_min_max_bounds)
    print("END")

@click.command()
@click.option("--forbush", default="FD006", help="Which forbush event to run")
@click.option("--launch-docker", default="launch_docker.py", help="Docker launcher script")
@click.option("--outfile", help="Output file")
def start_script(forbush, launch_docker,outfile):

    min_max_interval = [(5e-5, 6e-4),(0.001,1)]
    number_of_entries = 6
    generation_entry_method = "log"
    initial_candidates = []
    for i in range(6):
        init_candidate = generate_initial_entry(number_of_entries=number_of_entries,
                                                        generation_entry_method=generation_entry_method,
                                                        min_max_interval=min_max_interval)

        initial_candidates.append(init_candidate)

    for i in range(3):
        run_all_files(procedure=optimize,
                          sampling=1,
                          min_max_interval=min_max_interval,
                          number_of_entries=number_of_entries,
                          random_seed=53 + i,
                          max_iter=5,
                          forbush_name = forbush,
                          launch_docker=launch_docker,
                          parquet_name = outfile,
                          methods = ["Powell"],
                          initial_candidates = initial_candidates[i])


if __name__ == "__main__":
    start_script()



