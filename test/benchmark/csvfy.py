import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

if __name__ == '__main__':
    SET = ''

    data_dir = Path(__file__).parent.parent / 'data'
    p_outputs = data_dir / 'benchmark' / f'results'

    data = []
    for exetime_path in p_outputs.rglob('exetime_*.csv'):
        with open(exetime_path, 'r') as f:
            exetime = sum(map(float, f.readlines()))
        device = exetime_path.parent.parent.name.split('_')[-1]
        spec = exetime_path.parent.name.split('_')
        version = exetime_path.stem.split('_')[-1]
        date, test, val = spec[0], spec[2], spec[3]
        data.append((date, test, val, version, device, exetime))

    df = pd.DataFrame(data, columns=['date', 'test', 'val', 'version', 'device', 'exetime'])
    print(df)
    df.to_csv(f'benchmark.csv', index=False)

