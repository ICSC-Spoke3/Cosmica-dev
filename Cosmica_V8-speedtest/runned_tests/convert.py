"""
Converts an old TXT-format file into a YAML file with inline (flow) lists for specific keys.

This script reads an input file in the old format and produces a YAML file in the new format.
Numeric values provided as comma-separated strings are converted into Python lists wrapped
in a custom InlineList class, which forces them to be dumped in inline (flow) style.

Usage:
    python converter.py input.txt output.yml
"""

import sys
import yaml


class InlineList(list):
    """
    Custom list type used to force inline (flow) YAML output.
    When an InlineList is dumped via PyYAML, its values are represented in flow style.
    """
    pass


def inline_list_representer(dumper, data):
    """
    Custom YAML representer for InlineList objects.

    This function ensures that the list is represented in inline (flow) style.

    Args:
        dumper: The YAML dumper instance.
        data: An InlineList instance.

    Returns:
        A YAML sequence node with flow_style set to True.
    """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


# Register the custom InlineList representer with PyYAML.
yaml.add_representer(InlineList, inline_list_representer)


def convert_txt_to_yml(txt):
    """
    Convert text from the old TXT format to a dictionary matching the desired YAML structure.

    The conversion performs the following:
      - Reads and retains a header comment (e.g., "#File generated on ...").
      - Maps keys from the old format to new keys.
      - Converts comma-separated numeric values into lists (wrapped in InlineList for inline output).
      - Collects HeliosphericParameters and HeliosheatParameters into 'dynamic' and 'static' sections.

    Args:
        txt (str): Contents of the input TXT file.

    Returns:
        tuple: A tuple containing:
            - header_comment (str): The header comment (if any) from the input.
            - new_data (dict): The converted data ready for YAML dumping.
    """
    # Initialize dictionaries for various sections.
    new_data = {}
    particles = {}
    sources = {}
    dynamic = {"heliosphere": {}}
    static = {"heliosphere": {}, "heliosheat": {}}

    # Lists to collect multi-value parameters.
    heliospheric_params = []  # Each entry is expected to have 13 numeric values.
    heliosheat_params = []  # Each entry is expected to have 2 numeric values.

    # Split the input text into lines.
    lines = txt.splitlines()
    header_comment = ""
    # if lines and lines[0].startswith("#"):
    #     header_comment = lines[0]

    # Process each line.
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Capture header comment if it starts with "#File".
        # if line.startswith("#File"):
        #     header_comment = line
        #     continue
        # Skip other comment lines.
        if line.startswith("#"):
            continue
        # Process only lines that contain a colon.
        if ":" not in line:
            continue

        # Split the line into a key and a value.
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # Map keys to the new format.
        if key == "RandomSeed":
            new_data["random_seed"] = int(value)
        elif key == "OutputFilename":
            new_data["output_path"] = value
        elif key == "Particle_NucleonRestMass":
            particles["nucleon_rest_mass"] = float(value)
        elif key == "Particle_MassNumber":
            particles["mass_number"] = float(value)
        elif key == "Particle_Charge":
            particles["charge"] = float(value)
        elif key == "Tcentr":
            # Convert comma-separated values into an InlineList.
            new_data["rigidities"] = InlineList(
                [float(x.strip()) for x in value.split(",") if x.strip()]
            )
        elif key == "SourcePos_theta":
            sources["th"] = InlineList(
                [float(x.strip()) for x in value.split(",") if x.strip()]
            )
        elif key == "SourcePos_phi":
            sources["phi"] = InlineList(
                [float(x.strip()) for x in value.split(",") if x.strip()]
            )
        elif key == "SourcePos_r":
            sources["r"] = InlineList(
                [float(x.strip()) for x in value.split(",") if x.strip()]
            )
        elif key == "Npart":
            new_data["n_particles"] = int(value)
        elif key == "Nregions":
            new_data["n_regions"] = int(value)
        elif key == "HeliosphericParameters":
            # Each line should have 13 comma-separated numbers.
            params = [float(x.strip()) for x in value.split(",") if x.strip()]
            heliospheric_params.append(params)
        elif key == "HeliosheatParameters":
            # Each line should have 2 comma-separated numbers.
            params = [float(x.strip()) for x in value.split(",") if x.strip()]
            heliosheat_params.append(params)
        else:
            # Ignore unrecognized keys.
            continue

    # Assemble particle and source information.
    new_data["isotopes"] = [particles]
    new_data["sources"] = sources
    new_data["relative_bin_amplitude"] = 0.00855

    # Process HeliosphericParameters.
    if heliospheric_params:
        # For the dynamic section, extract the first value (k0) from each line.
        # If there are exactly 49 lines, split into two groups (first 35 and last 14).
        k0_all = InlineList([row[0] for row in heliospheric_params])
        dynamic["heliosphere"]["k0"] = [k0_all]
        # For the static section, collect the second value (ssn) from each line.
        for v, i in (
                ("ssn", 1), ("v0", 2), ("tilt_angle", 3), ("smooth_tilt", 4), ("b_field", 5), ("polarity", 6),
                ("solar_phase", 7), ("nmcr", 8), ("ts_nose", 9), ("ts_tail", 10), ("hp_nose", 11), ("hp_tail", 12)):
            static["heliosphere"][v] = InlineList([row[i] for row in heliospheric_params])
        # static["heliosphere"]["ssn"] = InlineList([row[1] for row in heliospheric_params])
        # static["heliosphere"]["V0"] = InlineList([row[2] for row in heliospheric_params])

    # Process HeliosheatParameters.
    if heliosheat_params:
        for v, i in (("k0", 0), ("v0", 1)):
            static["heliosheat"][v] = InlineList([row[i] for row in heliosheat_params])

    # Add the dynamic and static sections to the new data.
    new_data["dynamic"] = dynamic
    new_data["static"] = static

    return header_comment, new_data


if __name__ == "__main__":
    # Ensure correct usage.
    if len(sys.argv) < 3:
        print("Usage: python converter.py input.txt output.yml")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read the entire input file.
    with open(input_file, "r") as fin:
        txt = fin.read()

    # Convert the input text to the YAML data structure.
    header_comment, data = convert_txt_to_yml(txt)

    # Write the header comment (if present) and YAML output to the output file.
    with open(output_file, "w") as fout:
        if header_comment:
            fout.write(header_comment + "\n")
        yaml.dump(data, fout, sort_keys=False, width=float("inf"))

    print(f"Converted {input_file} to {output_file}")
