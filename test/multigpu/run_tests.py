import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess
import time

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


def run_cosmica(base_command: list[str | Path] | str | Path, input_file: Path, output_dir: Path,
                cuda_devices: str = '0'):
    try:
        output_dir.mkdir(exist_ok=True, parents=True)

        command = list(map(str, base_command)) if isinstance(base_command, list) else [str(base_command)]
        command += ["-i", str(input_file)]

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
        print(f"Error: Cosmica executable not found at {base_command}")
        return None, None, -1, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, -1, None, None


if __name__ == "__main__":
    nvmlInit()
    GPU_COUNT = nvmlDeviceGetCount()
    CUDA_DEVICES = ','.join(map(str, range(GPU_COUNT)))
    nvmlShutdown()

    VERSION = sys.argv[1]
    SETS = list(map(str, sys.argv[2:]))

    data_dir = Path(__file__).parent.parent / 'data'
    if VERSION == 'V0':
        p_cosmica = Path(__file__).parent.parent.parent / f'HelMod_V0' / 'Cosmica'
    else:
        p_cosmica = Path(__file__).parent.parent.parent / f'Cosmica_{VERSION}-speedtest' / 'exefiles' / 'Cosmica'
    # p_cosmica = [sys.executable, Path(__file__).parent.parent.parent / f'Cosmica_{VERSION}-speedtest' / 'launch_docker.py']
    p_inputs = sorted(sum([
        sorted((data_dir / 'benchmark' / 'inputs').rglob(f'{SET}/*.yaml' if VERSION == 'V8' else f'{SET}/*.txt'))
        for SET in SETS
    ], start=[]))
    p_outputs = data_dir / 'multigpu' / f'outputs_{GPU_COUNT}'

    for inpt in p_inputs:
        out_dir = p_outputs / inpt.parent.name
        _, _, _, bench, power = run_cosmica(p_cosmica, inpt, out_dir, cuda_devices=CUDA_DEVICES)
        with open(out_dir / f'power_{VERSION}.csv', 'a') as f:
            f.writelines([f'{p}\n' for p in power])
        with open(out_dir / f'exetime_{VERSION}.csv', 'a') as f:
            f.write(f'{bench}\n')
