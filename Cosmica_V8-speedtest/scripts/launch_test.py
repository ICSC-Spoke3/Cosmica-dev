import subprocess

import numpy as np
import yaml
from pathlib import Path

from matplotlib import pyplot as plt


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
        print(bins)
        histos.append(np.array(bins))
        print(histos[-1])
    return histos

if __name__ == "__main__":
    directory = Path(__file__).parent.parent
    cosmica_executable = directory / "exefiles" / "Cosmica"
    log_file = directory / "outputs" / "cosmica.log"
    output_dir = directory / "outputs"
    with open(directory / "runned_tests/AMS-02_PRL2015/Input_Proton_TKO_20110509_20131121_r00100_lat00000.yaml") as f:
        input_data = f.read()

    stdout, stderr, returncode = run_cosmica(input_data, cosmica_executable, log_file, output_dir)

    with open(directory / "outputs/tmp.yaml", "w") as f:
        f.write(stdout)

    if stdout is not None:
        print("Cosmica Output (stdout):\n", stdout)
        print("Cosmica Error Output (stderr):\n", stderr)
        print(f"Cosmica Return Code: {returncode}")
    else:
        print("Cosmica execution failed.")

    bss = list(load_baselines())
    r = parse_simple(stdout)

    for i, (x1, x2, x3, x4) in enumerate(zip(*bss, r)):
        plt.plot(x1, label='V6')
        plt.plot(x2, label='V8.0')
        plt.plot(x3, label='V8.1')
        plt.plot(x4, label='V8 latest')
        plt.legend()
        plt.savefig(Path(__file__).parent / "plots" / f"p{i}.png")
        plt.close()

