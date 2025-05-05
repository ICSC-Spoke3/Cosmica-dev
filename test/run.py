import subprocess
import time
from glob import glob
from os.path import join as pjoin, dirname, basename
import pandas as pd


def run_cosmica(cosmica_executable, input_file, output_dir, cuda_devices='0,1'):
    try:

        command = [
            str(cosmica_executable),
            "-i",
            input_file,
            # "--legacy" if 'V8' in cosmica_executable else "",
        ]
        print(f"Executing command: {' '.join(command)}")

        start = time.time()
        process = subprocess.run(command, env={'CUDA_VISIBLE_DEVICES': cuda_devices}, cwd=output_dir,
                                 capture_output=True, text=True)
        bench = time.time() - start

        print('Time:', bench)
        print(process.stdout, process.stderr)

        return process.stdout, process.stderr, process.returncode, bench

    except FileNotFoundError as f:
        print(f)
        print(f"Error: Cosmica executable not found at {cosmica_executable}")
        return None, None, -1, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, -1, None


if __name__ == "__main__":
    ROOTDIR = pjoin(dirname(__file__), 'data')
    pcosmica = pjoin(dirname(dirname(__file__)), 'Cosmica_V8-speedtest', 'exefiles', 'Cosmica')
    pinputs = pjoin(ROOTDIR, 'inputs', '*.yaml')
    poutputs = pjoin(ROOTDIR, 'outputs', 'tmp')

    df = {'file': [], 'time': []}

    for inpt in sorted(glob(pinputs)):
        _, _, _, bench = run_cosmica(pcosmica, inpt, poutputs, cuda_devices='0,1')
        df['file'].append(basename(inpt))
        df['time'].append(bench)

    pd.DataFrame(df).to_csv(pjoin(poutputs, 'time.csv'), index=False)
