import subprocess
import time
import pandas as pd
from pathlib import Path

from pynvml import *


def monitor_gpu_power(power_readings_list, stop_event, device_index='0', interval_seconds=0.5):
    try:
        nvmlInit()
        handles = [nvmlDeviceGetHandleByIndex(int(idx)) for idx in device_index.split(',')]
        while not stop_event.is_set():
            power_readings_list.append(
                sum(nvmlDeviceGetPowerUsage(h) for h in handles)
            )
            time.sleep(interval_seconds)
    finally:
        nvmlShutdown()


def run_cosmica(cosmica_executable, input_file, output_dir, cuda_devices='0,1'):
    try:
        output_dir.mkdir(exist_ok=True, parents=True)

        command = [str(cosmica_executable), "-i", str(input_file)]

        power_readings_mw = []
        stop_monitoring_event = threading.Event()

        monitor_thread = threading.Thread(
            target=monitor_gpu_power,
            args=(power_readings_mw, stop_monitoring_event, cuda_devices, 0.5)
        )
        monitor_thread.start()

        print('Running:', ' '.join(command))
        start = time.time()
        process = subprocess.run(command,
                                 env={'CUDA_VISIBLE_DEVICES': cuda_devices},
                                 cwd=output_dir,
                                 capture_output=True, text=True)
        bench = time.time() - start

        stop_monitoring_event.set()
        monitor_thread.join()

        return process.stdout, process.stderr, process.returncode, bench, power_readings_mw

    except FileNotFoundError as f:
        print(f)
        print(f"Error: Cosmica executable not found at {cosmica_executable}")
        return None, None, -1, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, -1, None


if __name__ == "__main__":
    VERSION = 'V8'

    data_dir = Path(__file__).parent.parent / 'data'
    p_cosmica = Path(__file__).parent.parent.parent / f'Cosmica_{VERSION}-speedtest' / 'exefiles' / 'Cosmica'
    p_inputs = sorted((data_dir / 'benchmark' / 'inputs').rglob('*10_1_42.yaml' if VERSION == 'V8' else '*.txt'))
    p_outputs = data_dir / 'benchmark' / 'outputs'

    for inpt in p_inputs:
        out_dir = p_outputs / inpt.parent.name
        _, _, _, bench, power = run_cosmica(p_cosmica, inpt, out_dir, cuda_devices='0')
        with open(out_dir / 'power.csv', 'w') as f:
            f.writelines([f'{p}\n' for p in power])
        with open(out_dir / 'exetime.txt', 'w') as f:
            f.write(f'{bench}\n')
