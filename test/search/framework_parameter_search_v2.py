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

from test.lib.files_utils import (
    LisLoader, SimulationPredictionItem, SimulationInput, HeliosphericParameters,
    SimulationExperimentItem, ExperimentalData, SimulationOutput, ModulationResult, estimate_k0
)
from test.lib.isotopes import IONS
from test.lib import metrics

import subprocess
import numpy as np
import yaml
import click

yaml.Dumper.ignore_aliases = lambda self, data: True


########################################################################
# Version 1.0:
# Version 2.0: Enhanced fitness evaluation: if the result lies within the experimental sensitivity bounds, it is not penalized
#
#
#######################################################################

from gradient_based_optimization_docker import base_input

def create_opt_pipeline(**kwargs):


    parquet_file_name = kwargs.get('parquet_name','fpso_optim')
    print(f'parquet_name {parquet_file_name}')
    sim = kwargs.get('sim')
    data_dir = kwargs.get('data_dir')
    cosmica_path = kwargs.get('cosmica_path')

    p_exp = kwargs.get('p_exp')
    p_out = kwargs.get('p_out')
    parquet_dir = kwargs.get('parquet_dir')
    file_input = kwargs.get('file_input')
    lis_loader = kwargs.get('lis_loader')
    random_seed = kwargs.get('random_seed')

    iter_folder = p_out / f'FPSO_iter'

    iter_folder.mkdir(parents=True, exist_ok=True)
    if not iter_folder.exists():
        print(f"Warning: folder not created: {iter_folder}")

    n_part = kwargs.get('n_part')

    template = base_input(data_dir, sim, rnd=random_seed, n_part=n_part)

    print(str(p_exp))

    experimental_data = ExperimentalData.from_data(p_exp, (0, 1, 2, 3))
    print(experimental_data)
    template = base_input(data_dir, sim, rnd=random_seed, n_part=n_part)

    rigidities = experimental_data.rig_flux.rigidity

    from gradient_based_optimization_docker import fluxes_mean

    def aux_get_mean_fitness(k0s,random_seed):

        res = fluxes_mean(template = template,
                                init_pop = [k0[0] for k0 in k0s],
                                k_validation= 1,
                                file_input_ = file_input,
                                cosmica_path = cosmica_path,
                                iter_folder = iter_folder,
                                lis_loader = lis_loader,
                                initial_k0 = k0s,
                                rigidities = rigidities,
                                random_seed = random_seed)
        return res[0]

    parquet_file = parquet_dir / parquet_file_name
    space_dim = 1
    LOWER_BOUND =5e-5
    UPPER_BOUND =6e-4
    MAX_ITERATIONS = 100
    MAX_ITER_WO_NEW_GLOBAL_BEST = 10
    POPULATION_SIZE = 10

    optimizer_fuzzy_pso = FuzzyPSO()
    optimizer_fuzzy_pso.disable_fuzzyrule_maxvelocity()
    optimizer_fuzzy_pso.set_search_space([[LOWER_BOUND,UPPER_BOUND]]*space_dim)
    optimizer_fuzzy_pso.InitCreateParticles(POPULATION_SIZE, space_dim,creation_method={"name":'logarithmic'})
    optimizer_fuzzy_pso._prepare_for_optimization(max_iter=MAX_ITERATIONS,max_iter_without_new_global_best=MAX_ITER_WO_NEW_GLOBAL_BEST,  max_FEs = None)


    optimizer_FUZZYPSO = Optimizer(optimizer=optimizer_fuzzy_pso,name="Fuzzy-PSO",eval_metric=metrics.rmse,
                                   save_results_ended=True,external_stopping_criteria=optimizer_fuzzy_pso.TerminationCriterion,
                                   parquet_dir=parquet_file,max_iterations=MAX_ITERATIONS,budget=POPULATION_SIZE)


    init_solutions = [solution.X for solution in optimizer_fuzzy_pso.Solutions]
    optim_queue = OptimizerQueue(optimizers_list=[optimizer_FUZZYPSO])

    optim_queue.start(procedure=aux_get_mean_fitness,
                      initial_value_parameter=init_solutions,
                      generate_random_seed_each_epoch=True,
                      real_data=experimental_data,
                      experimental_data_str = str(p_exp),
                      n_part=16384)


@click.command()
@click.option("--forbush", default="FD006", help="Which forbush event to run")
@click.option("--launch-docker", default="launch_docker.py", help="Docker launcher script")
@click.option("--outfile", help="Output file")
def test(forbush, launch_docker,outfile):
    from gradient_based_optimization_docker import run_all_files

    for i in range(0,3):
        run_all_files(procedure=create_opt_pipeline,
                      n_part = 16384,
                      forbush_name = forbush,
                      launch_docker=launch_docker,
                      parquet_name = outfile)

if __name__ == "__main__":
    test()
