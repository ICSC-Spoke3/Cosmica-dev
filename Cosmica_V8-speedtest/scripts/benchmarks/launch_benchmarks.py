import copy
import subprocess
import time

import numpy as np
import yaml
from pathlib import Path

yaml.Dumper.ignore_aliases = lambda self, data: True


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


def load_baselines():
    names = ['V6', 'V8.0', 'V8.1']

    def to_arrays(s):
        return list(map(lambda x: np.array(x.strip().split(' '), dtype=float),
                        filter(lambda x: x and not x.startswith('#'), s.split('\n'))))

    with open(Path(__file__).parent / "baselines.txt") as f:
        for n, bs in zip(names, f.read().split("---\n")):
            yield to_arrays(bs)[2::2]


def parse_simple(stdout):
    histos = []
    res = yaml.load(stdout, Loader=yaml.FullLoader)
    for rig in res['histograms']:
        isotopes = rig['isotopes']
        proton_0 = isotopes['proton'][0]
        bins = proton_0['bins']
        histos.append(np.array(bins))
    return histos


def run_simulation(seed, particles, nk0, input_yaml, cosmica_executable, log_file, output_dir):
    input_yaml = copy.deepcopy(input_yaml)
    input_yaml['random_seed'] = seed
    input_yaml['n_particles'] = particles
    input_yaml['dynamic']['heliosphere']['k0'] *= nk0
    input_string = yaml.dump(input_yaml, Dumper=yaml.Dumper)

    st = time.time()
    stdout, stderr, returncode = run_cosmica(input_string, cosmica_executable, log_file, output_dir)
    et = time.time()

    assert returncode == 0, f"Cosmica execution failed with return code {returncode}"

    return et - st


if __name__ == "__main__":
    directory = Path(__file__).parent.parent.parent
    cosmica_executable = directory / "exefiles" / "Cosmica"
    log_file = directory / "outputs" / "cosmica.log"
    output_dir = directory / "outputs"
    with open(directory / "runned_tests/AMS-02_PRL2015/Input_Proton_TKO_20110509_20131121_r00100_lat00000.yaml") as f:
        input_yaml = yaml.load(f, Loader=yaml.FullLoader)

    results = []

    for nk0 in range(125, 201, 25):
        nk0 = max(1, nk0)
        t = []
        for i in range(5):
            print(nk0, i)
            t.append(run_simulation(0, 144, nk0, input_yaml, cosmica_executable, log_file, output_dir))
            print(t[-1])
        results.append((nk0, *t))
        with open(Path(__file__).parent / "nk0_times.csv", "a+") as f:
            f.write(f'{nk0}, {", ".join(map(str, t))}\n')
