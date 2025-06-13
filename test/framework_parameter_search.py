import subprocess

import numpy as np
import yaml
from pathlib import Path
from glob import glob
import sys

from matplotlib import pyplot as plt
import yaml
from .make_input import InlineList
from .lib.modulation import evaluate_spectra_multiple
from .lib.files_utils import load_experimental_data
from .plot import match_file
from .lib.files_utils import load_simulation_outputs_yaml, load_lis
from .lib.modulation import evaluate_modulations

yaml.Dumper.ignore_aliases = lambda self, data: True
yaml.add_representer(InlineList, InlineList.representer)


def generate_input_file(template_path, k0_list: list, output_path=None):
    """
    Generate a YAML input file for Cosmica using a template and a list of k0 values.
    Args:
        template_path (str or Path): Path to the template YAML file.
        k0_list (list): List of k0 values to be included in the input file.
        output_path (str or Path, optional): Path where the generated input file will be saved.
            If None, the function returns the YAML string instead of saving it to a file.
    Returns:
        str: The generated YAML string if output_path is None, otherwise None.
    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    template = yaml.safe_load(template_path.read_text())

    template['dynamic']['heliosphere']['k0'] = [InlineList([k0]*len(15)) for k0 in k0_list]

    if output_path is not None:
        with open(output_path, 'w') as output_file:
            yaml.dump(template, output_file, sort_keys=False, width=float('inf'))
        return None
    else:
        # If no output path is provided, return the YAML string
        template = yaml.dump(template, sort_keys=False, width=float('inf'))
        return template



def run_cosmica(input_string, cosmica_executable, log_file, output_dir):
    try:
        command = [
            str(cosmica_executable),
            "-v",
            "trace",
            "--stdin",
            "--stdout",
            "--log_file",
            str(log_file),
            "-o",
            str(output_dir) + '/',
        ]
        print(f"Executing command: {' '.join(command)}")

        process = subprocess.run(command, input=input_string, capture_output=True, text=True)

        return process.stdout, process.stderr, process.returncode

    except FileNotFoundError:
        print(f"Error: Cosmica executable not found at {cosmica_executable}")
        return None, None, -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, -1
    

def mistery_function_next_k0list(fitness_score, k0_list):
    """
    Generate the next k0 list based on the fitness score and current k0 list.
    
    Args:
        fitness_score (float): The fitness score from the previous evaluation.
        k0_list (list): The current list of k0 values.
        
    Returns:
        list: A new list of k0 values for the next iteration.
    """
    # Placeholder
    print(f"=== Placeholder for mistery_function_next_k0list ===")
    next_k0_list = [k + 0.1 for k in k0_list] 
    print(f"Next k0 list based on fitness score {fitness_score}: {next_k0_list}")
    return next_k0_list


def evaluate_output(outputs, experimental_data, lis, metric_fn=None):
    """
    Evaluate the output of a simulation and compare it with experimental data.
    Args:
        outputs (list): List of simulation output files.
        experimental_data (np.ndarray): Experimental data with columns (energy, j_mod, inf, sup).
        raw_data (np.ndarray): Raw data with columns (energy, j_mod).
        lis: LIS data for evaluation.
    Returns:
        list: List of RMSE values for each simulation output compared to the experimental data, or a list of metric values if metric_fn is provided.
    Asserts:
        np.allclose: Checks if the simulated rigidity matches the experimental rigidity within a tolerance.
    """

    results = evaluate_modulations(lis, *outputs)
    rig, lis_flux = results.T[:2]
    fluxes = results.T[2:]

    rig = rig[:len(experimental_data)]
    lis_flux = lis_flux[:len(experimental_data)]
    fluxes = fluxes[:, :len(experimental_data)]

    exp_en_rig, exp_j_mod, exp_inf, exp_sup = experimental_data.T

    assert np.allclose(rig, exp_en_rig, rtol=0.02 * rig)

    if metric_fn is not None:
        return [metric_fn(rig, flux, exp_j_mod) for flux in fluxes]
    else:
        rmses = []
        for i, flux in enumerate(fluxes):
            rmse = np.sqrt(np.square(np.subtract(flux, exp_j_mod)).mean())
            rmses.append(rmse)

    return rmses



if __name__ == "__main__":
    directory = Path(__file__).parent.parent
    cosmica_executable = directory / "exefiles" / "Cosmica"
    log_file = directory / "outputs" / "cosmica.log"
    output_dir = directory / "outputs"
    template_path = directory / "templates" / "input_template.yaml"

    ROOTDIR = directory / 'data'
    plis = ROOTDIR / 'LIS_Default2020_Proton'
    pinputs = ROOTDIR / 'inputs'
    prigi = ROOTDIR / 'rigidity_groups.xlsx'

    pexp = ROOTDIR / 'experimental' / '*.dat'
    praw = ROOTDIR / 'helmod' / '*.txt'
    psims = ROOTDIR / 'Simulations.list'
    pplots = directory / 'plots'

    experimental = sorted(glob(pexp), reverse=True)
    helmod = sorted(glob(praw), reverse=True)

    lis = load_lis(plis)

    # exp_data from command line, so retrieve it
    exp_data = sys.argv[1] if len(sys.argv) > 1 else None
    exp_data = load_experimental_data(exp_data, cols=(2, 3, 4, 5), rig_range=(0, 10))

    # Initial k0 list
    k0_list = [1.0, 1.25, 1.5, 1.75, 2.0]

    for iteration in range(3):
        print(f"\nI teration {iteration+1}")
        input_string = generate_input_file(
            template_path=template_path,
            k0_list=k0_list
        )

        # Run Cosmica with the generated input
        print("Running Cosmica with the generated input...")
        stdout, stderr, returncode = run_cosmica(input_string, cosmica_executable, log_file, output_dir)
        yml = yaml.load(stdout, Loader=yaml.SafeLoader)
        res_out = [load_simulation_outputs_yaml(yml, param=i) for i in range(len(k0_list))]
        if res_out is None:
            print("Failed to load simulation outputs.")
            continue

        if stdout is not None:
            with open(directory / f"outputs/tmp_iter{iteration+1}.yaml", "w") as f:
                f.write(stdout)
            print("Cosmica Output (stdout):\n", stdout)
            print("Cosmica Error Output (stderr):\n", stderr)
            print(f"Cosmica Return Code: {returncode}")

            # Evaluate fitness for each k0
            out = evaluate_output(
                    output_execution=res_out,
                    experimental_data=exp_data,
                    lis=lis
                )
            fitnesses = np.array(fitnesses)
            print(f"Fitness scores for k0 values {k0_list}: {fitnesses}")
            average_fitness = np.mean(fitnesses)
            print(f"Average fitness score: {average_fitness}")
        
            # Generate the next k0 list based on the average fitness
            print("Generating next k0 list...")
            k0_list = mistery_function_next_k0list(
                fitness_score=average_fitness,
                k0_list=k0_list
            )
            print(f"Next k0 list: {k0_list}")
        else:
            print("Cosmica execution failed.")
            break

    

