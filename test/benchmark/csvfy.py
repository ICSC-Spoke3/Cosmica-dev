import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

if __name__ == '__main__':
    SET = ''

    data_dir = Path(__file__).parent.parent / 'data'
    p_outputs = data_dir / 'benchmark' / 'results'

    data = []
    for exetime_path, power_path in zip(sorted(p_outputs.rglob('exetime_*.csv')),
                                        sorted(p_outputs.rglob('power_*.csv'))):
        with open(exetime_path, 'r') as f:
            exetime = sum(map(float, f.readlines()))
        with open(power_path, 'r') as f:
            power = sum(map(float, f.readlines())) / 1000
        device = ' '.join(exetime_path.parent.parent.name.split('_')[1:])
        spec = exetime_path.parent.name.split('_')
        version = exetime_path.stem.split('_')[-1]
        date, seed, test, val = spec[0], spec[2], spec[3], spec[4]
        data.append((int(date), int(seed), test, int(val), version, device, float(exetime), float(power)))

    df = pd.DataFrame(data, columns=['date', 'seed', 'test', 'val', 'version', 'device', 'exetime', 'power'])
    print(df)
    df.to_csv(f'benchmark.csv', index=False)

