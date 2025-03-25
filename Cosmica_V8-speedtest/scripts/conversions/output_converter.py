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


def convert_dat_to_yaml(txt, partname):
    txt = list(filter(lambda x: not x.startswith('#'), txt.splitlines()))
    configs, bins = [x.split() for x in txt[1::2]], txt[2::2]
    yml = {'histograms': [{partname: [
        {
            'amplitude': float(c[5]),
            'bins': list(map(float, b.split())),
            'lower_bound': float(c[4]),
            'n_part': int(c[1]),
            'n_reg': int(c[2]),
            'rigidity': float(c[0]),
        } for c, b in zip(configs, bins)
    ]}]}
    return yaml.dump(yml, sort_keys=False, width=float("inf"))


def convert_yaml_to_dat(yml, partname):
    yml = yaml.load(yml, Loader=yaml.SafeLoader)
    yml = yml['histograms'][0][partname]
    d = [
        (
            f'{y['rigidity']} {y['n_part']} {y['n_reg']} {len(y['bins'])} {y['lower_bound']} {y['amplitude']}',
            ' '.join(map(str, y['bins']))
        ) for y in yml
    ]
    ds = f'{len(yml)}\n' + ''.join(f'{k}\n{v}\n' for k, v in d)
    return ds


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python converter.py input.txt output.yml")
        exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r") as fin:
        fc = fin.read()

    if input_file.endswith('.dat'):
        data = convert_dat_to_yaml(fc, 'proton')
    else:
        data = convert_yaml_to_dat(fc, 'proton')

    with open(output_file, "w") as fout:
        fout.write(data)

    print(f"Converted {input_file} to {output_file}")
