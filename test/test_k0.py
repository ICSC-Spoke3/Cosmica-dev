import subprocess
import time
from os.path import join as pjoin, dirname

import yaml

from lib.files_utils import load_simulation_outputs_yaml

yaml.Dumper.ignore_aliases = lambda self, data: True


def run_cosmica_with_k0(cosmica_executable, k0, base_file, output_dir):
    with open(base_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data['dynamic']['heliosphere']['k0'][0][0] = k0
    input_string = yaml.dump(data, Dumper=yaml.Dumper)

    try:
        start = time.time()

        command = [
            str(cosmica_executable),
            "--stdin",
            "--stdout",
            "--log_file", "cosmica.log"
        ]
        print(f"Executing command: {' '.join(command)}")

        process = subprocess.run(command, input=input_string, cwd=output_dir, capture_output=True, text=True)

        print('Time:', time.time() - start)

        if process.returncode != 0:
            print(process.returncode)
            print(process.stderr)
            print(process.stdout)
            return None

        return yaml.load(process.stdout, Loader=yaml.FullLoader)

    except FileNotFoundError as f:
        print(f)
        print(f"Error: Cosmica executable not found at {cosmica_executable}")
        return None, None, -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, -1


if __name__ == "__main__":
    ROOTDIR = pjoin(dirname(__file__), 'data')
    pcosmica = pjoin(dirname(dirname(__file__)), 'Cosmica_V8-speedtest', 'exefiles', 'Cosmica')
    pinput = pjoin(ROOTDIR, 'inputs', 'Proton_Deuteron_20111116_20111116_r00100_lat00000.yaml')
    pwdir = pjoin(ROOTDIR, 'tmp')

    yml_out = run_cosmica_with_k0(pcosmica, 1.0, pinput, pwdir)
    out = load_simulation_outputs_yaml(yml_out)
    print(out)
