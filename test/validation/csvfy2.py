import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import yaml
from test.lib.files_utils import SimulationOutput, LisLoader

if __name__ == '__main__':
    SET = ''

    data_dir = Path(__file__).parent.parent / 'data'
    p_outputs = data_dir / 'benchmark' / 'results' / 'outputs_A100'

    lis_loader = LisLoader(data_dir / 'LIS_Default2020_Proton')

    data = []
    for v8_path, v0_p_path, v0_d_path in zip(sorted(p_outputs.rglob('*part*/*.yaml')),
                                             sorted(p_outputs.rglob('*part*/proton*.dat')),
                                             sorted(p_outputs.rglob('*part*/deuteron*.dat'))):
        v8_s = SimulationOutput.from_yaml(yaml.load(v8_path.read_text(), Loader=yaml.CLoader))
        v0_s = SimulationOutput.from_txt([{'proton': v0_p_path.read_text(), 'deuteron': v0_d_path.read_text()}])

        v8 = v8_s.modulate(lis_loader)[0]
        v0 = v0_s.modulate(lis_loader, is_energy=True)[0]

        spec = v8_path.parent.name.split('_')
        date, seed, val = spec[0], spec[2], spec[4]

        for v8r, v8f, v0r, v0f, vl8f, vl0f in zip(v8.rigidity, v8.flux, v0.rigidity, v0.flux, v8.lis_flux.flux, v0.lis_flux.flux):
            assert np.allclose(v8r, v0r, rtol=0.02), (v8r, v0r)
            data.append((int(date), int(seed), int(val), float(v8r), float(v8f), float(v0f), float(vl8f), float(vl0f)))

    df = pd.DataFrame(data, columns=['date', 'seed', 'val', 'rigidity', 'flux_v8', 'flux_v0', 'flux_lis_v8', 'flux_lis_v0'])
    print(df)
    df.to_csv(f'validation2.csv', index=False)
