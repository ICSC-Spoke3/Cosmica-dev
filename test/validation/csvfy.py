import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import yaml
from test.lib.files_utils import SimulationOutput, LisLoader

if __name__ == '__main__':
    SET = ''

    data_dir = Path(__file__).parent.parent / 'data'
    p_outputs = data_dir / 'benchmark' / 'outputs'
    p_old_outputs = data_dir / 'benchmark' / 'results_old' / 'outputs_A40'

    lis_loader = LisLoader(data_dir / 'LIS_Default2020_Proton')

    data = []
    for v8_path, v6_p_path, v6_d_path in zip(sorted(p_outputs.rglob('*part*/*.yaml')),
                                             sorted(p_outputs.rglob('*part*/proton*.dat')),
                                             sorted(p_outputs.rglob('*part*/deuteron*.dat'))):
        v8 = SimulationOutput.from_yaml(yaml.load(v8_path.read_text(), Loader=yaml.CLoader)).modulate(lis_loader)[0]
        v6 = SimulationOutput.from_txt([{'proton': v6_p_path.read_text(), 'deuteron': v6_d_path.read_text()}]).modulate(
            lis_loader)[0]

        spec = v8_path.parent.name.split('_')
        date, seed, val = spec[0], spec[2], spec[4]

        v8_old_path = next(p_old_outputs.rglob(f'{date}*part_{val}/*.yaml'), None)
        if v8_old_path is None:
            v8_old_path = next(p_old_outputs.rglob(f'{date}*part_{int(val) * 2}/*.yaml'))

        v8_old = SimulationOutput.from_yaml(
            yaml.load(v8_old_path.read_text(), Loader=yaml.CLoader)).modulate(lis_loader)[0]

        for v8r, v8f, v6r, v6f, v8or, v8of in zip(v8.rigidity, v8.flux, v6.rigidity, v6.flux, v8_old.rigidity, v8_old.flux):
            assert (v8r == v6r) and (v8r == v8or)
            data.append((int(date), int(seed), int(val), float(v8r), float(v8f), float(v6f), float(v8of)))

    df = pd.DataFrame(data, columns=['date', 'seed', 'val', 'rigidity', 'flux_v8', 'flux_v6', 'flux_v8o'])
    print(df)
    df.to_csv(f'validation.csv', index=False)
