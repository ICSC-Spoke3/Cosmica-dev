import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

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


'''
    This function has been created to evaluate parameters found out by using rigidity analysis function.
    We need to extract top k0 parameters from parquet named: FDXXX-rigidity_analysis_data
    
'''
@click.command()
@click.option("--forbush", default="FD006", help="Which forbush event to run")
@click.option("--launch-docker", default="launch_docker.py", help="Docker launcher script")
@click.option("--input",default="FD006-rigidity_analysis_data", help="Input file")
@click.option("--output",default="output_file", help="Output file")
def test(forbush, launch_docker,input,output):

    from gradient_based_optimization_docker import launch_simulation

    parquet_path = Path.cwd() / 'parquet' / input
    df = pd.read_parquet(parquet_path)

    df["g_inv_optimized_k0"] = np.exp(df["optimized_k0"])
    df = df[df["success"] == True]
    best_df = (
        df.loc[df.groupby(["data_dir", "bitmask"])["final_fitness"].idxmin()]
        .reset_index(drop=True)
    )
    to_parquet_path = Path.cwd() / 'parquet' / ( 'best_k0_' + input )
    best_df.to_parquet(str(to_parquet_path), engine="pyarrow",index=False)

    print("Dati utilizzati in:"  + str(to_parquet_path))

    for j in range(0,4):
        for exp_datapath, group in best_df.groupby("data_dir"):
            k0_array = group["g_inv_optimized_k0"].to_numpy()
            bitmask = group["bitmask"].to_numpy()

            print(f"Running simulation for: {exp_datapath}")
            print(f"  -> {len(k0_array)} k0 values")

            # Lancia la simulazione (decommenta quando pronto)
            launch_simulation(
                 k0_array=k0_array,
                 bitmask = bitmask,
                 output_file=output,
                 launch_docker=launch_docker,
                 exp_datapath=exp_datapath,
                 forbush=forbush,
                 random_seed=42,
                 n_part=65536,
                 turn=0,
                particle_to_test = ['helium']#['helium','proton']
             )

if __name__ == "__main__":
    test()
