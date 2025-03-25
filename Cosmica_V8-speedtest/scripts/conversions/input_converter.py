"""
Converts an old TXT-format file into a YAML file with inline (flow) lists for specific keys.

This script reads an input file in the old format and produces a YAML file in the new format.
Numeric values provided as comma-separated strings are converted into Python lists wrapped
in a custom InlineList class, which forces them to be dumped in inline (flow) style.

Usage:
    python input_converter.py input.txt output.yml
"""

import sys
from math import ceil

import numpy as np
import yaml

yaml.Dumper.ignore_aliases = lambda *args: True


class InlineList(list):
    @staticmethod
    def inline_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(InlineList, InlineList.inline_list_representer)


def convert_txt_to_yaml(txt, partname):
    d = {}
    for k, v in map(lambda x: x.split(': '), filter(lambda x: not x.startswith('#'), txt.splitlines())):
        if k in ('HeliosphericParameters', 'HeliosheatParameters'):
            if d.get(k) is None:
                d[k] = [v]
            else:
                d[k].append(v)
        else:
            d[k] = v

    hp_params = np.array(list(map(lambda x: list(map(float, x.split(','))), d['HeliosphericParameters'])))
    hs_params = np.array(list(map(lambda x: list(map(float, x.split(','))), d['HeliosheatParameters'])))

    yml = {
        'random_seed': int(d['RandomSeed']),
        'output_path': d['OutputFilename'],
        'rigidities': InlineList(map(float, d['Tcentr'].split(','))),
        'isotopes': {partname: {
            'nucleon_rest_mass': float(d['Particle_NucleonRestMass']),
            'mass_number': float(d['Particle_MassNumber']),
            'charge': float(d['Particle_Charge']),
        }},
        'sources': {
            'r': InlineList(map(float, d['SourcePos_r'].split(','))),
            'th': InlineList(map(float, d['SourcePos_theta'].split(','))),
            'phi': InlineList(map(float, d['SourcePos_phi'].split(','))),
        },
        'relative_bin_amplitude': float(d.get('RelativeBinAmplitude', 0.00855)),
        'n_particles': int(d['Npart']),
        'n_regions': int(d['Nregions']),
        'dynamic': {'heliosphere': {'k0': [InlineList(hp_params[:, 0].tolist())]}},
        'static': {
            'heliosphere': {
                k: InlineList(hp_params[:, i + 1].tolist())
                for i, k in enumerate(('ssn', 'v0', 'tilt_angle', 'smooth_tilt', 'b_field', 'polarity', 'solar_phase',
                                       'nmcr', 'ts_nose', 'ts_tail', 'hp_nose', 'hp_tail'))
            },
            'heliosheat': {
                k: InlineList(hs_params[:, i].tolist())
                for i, k in enumerate(('k0', 'v0'))
            }
        }
    }

    yml['n_particles'] = int(ceil(yml['n_particles'] / len(yml['sources']['r'])))
    return yaml.dump(yml, sort_keys=False, width=float("inf"))


def convert_yaml_to_txt(yml, partname):
    yml = yaml.load(yml, Loader=yaml.SafeLoader)
    d = {
        'RandomSeed': yml['random_seed'], 'OutputFilename': yml['output_path'],
        'Particle_NucleonRestMass': yml['isotopes'][partname]['nucleon_rest_mass'],
        'Particle_MassNumber': yml['isotopes'][partname]['mass_number'],
        'Particle_Charge': yml['isotopes'][partname]['charge'], 'Tcentr': ', '.join(map(str, yml['rigidities'])),
        'SourcePos_r': ', '.join(map(str, yml['sources']['r'])),
        'SourcePos_theta': ', '.join(map(str, yml['sources']['th'])),
        'SourcePos_phi': ', '.join(map(str, yml['sources']['phi'])),
        'Npart': yml['n_particles'] * len(yml['sources']['r']), 'Nregions': yml['n_regions'],
        'HeliosphericParameters': [
            [yml['dynamic']['heliosphere']['k0'][0][i]] +
            [v[i] for k, v in yml['static']['heliosphere'].items()]
            for i, _ in enumerate(yml['static']['heliosphere']['v0'])
        ], 'HeliosheatParameters': [
            [v[i] for k, v in yml['static']['heliosheat'].items()]
            for i, _ in enumerate(yml['static']['heliosheat']['v0'])
        ]}

    ds = ''
    for k, v in d.items():
        if k in ('HeliosphericParameters', 'HeliosheatParameters'):
            for e in v:
                ds += f'{k}: {", ".join(map(str, e))}\n'
        else:
            ds += f'{k}: {v}\n'
    return ds


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python converter.py input.txt output.yml")
        exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r") as fin:
        fc = fin.read()

    if input_file.endswith('.txt'):
        data = convert_txt_to_yaml(fc, 'proton')
    else:
        data = convert_yaml_to_txt(fc, 'proton')

    with open(output_file, "w") as fout:
        fout.write(data)

    print(f"Converted {input_file} to {output_file}")
