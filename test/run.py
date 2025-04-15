import time
from collections import defaultdict
from math import ceil
from os.path import join as pjoin, dirname

import numpy as np
import yaml

from lib.files_utils import load_simulation_list, load_heliospheric_parameters, load_experimental_data
from lib.isotopes import find_ion_or_isotope
from lib.physics_utils import rig_to_en

from glob import glob

import subprocess


def run_cosmica(cosmica_executable, input_file, output_dir):
    try:
        start = time.time()

        command = [
            str(cosmica_executable),
            "-i",
            input_file,
            "--legacy" if 'V8' in cosmica_executable else "",
        ]
        print(f"Executing command: {' '.join(command)}")

        process = subprocess.run(command, cwd=output_dir, capture_output=True, text=True)

        print('Time:', time.time() - start)

        return process.stdout, process.stderr, process.returncode

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
    pinputs = pjoin(ROOTDIR, 'inputs')
    poutputs = pjoin(ROOTDIR, 'outputs', 'v8')

    for inpt in glob(pjoin(pinputs, '*.yaml')):
        print(run_cosmica(pcosmica, inpt, poutputs))
