import yaml
from collections import OrderedDict

# Helper function to represent OrderedDict in YAML output correctly
def represent_ordereddict(dumper, data):
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))
    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

yaml.add_representer(OrderedDict, represent_ordereddict)

# --- Component Classes ---

class Isotope:
    """Represents a single isotope type."""
    def __init__(self, name: str, nucleon_rest_mass: float, mass_number: float, charge: float):
        self.name = name
        self.nucleon_rest_mass = nucleon_rest_mass
        self.mass_number = mass_number
        self.charge = charge

    def to_dict(self):
        """Converts isotope data to a dictionary."""
        return {
            'nucleon_rest_mass': self.nucleon_rest_mass,
            'mass_number': self.mass_number,
            'charge': self.charge
        }

    def __repr__(self):
        return (f"Isotope(name='{self.name}', "
                f"nucleon_rest_mass={self.nucleon_rest_mass}, "
                f"mass_number={self.mass_number}, charge={self.charge})")

class DynamicHeliosphere:
    """Represents dynamic parameters specifically for the heliosphere."""
    def __init__(self, k0: list[list[float]]):
        # k0 is expected as a list of lists (e.g., [[...], [...]])
        # based on the input YAML structure. If it should only be one list,
        # adjust the type hint and logic accordingly.
        # The example YAML shows k0 as a list containing *one* list of floats.
        self.k0 = k0

    def to_dict(self):
        """Converts dynamic heliosphere data to a dictionary."""
        return {'k0': self.k0}

    def __repr__(self):
        return f"DynamicHeliosphere(k0={self.k0})"

class DynamicParameters:
    """Represents the 'dynamic' section of the input."""
    def __init__(self, heliosphere: DynamicHeliosphere):
        self.heliosphere = heliosphere

    def to_dict(self):
        """Converts dynamic parameters to a dictionary."""
        return {'heliosphere': self.heliosphere.to_dict()}

    def __repr__(self):
        return f"DynamicParameters(heliosphere={self.heliosphere})"

class StaticHeliosphereRegion:
    """Represents static parameters for one region/timestep of the heliosphere."""
    def __init__(self, ssn: float, v0: float, tilt_angle: float, smooth_tilt: float,
                 b_field: float, polarity: float, solar_phase: float, nmcr: float,
                 ts_nose: float, ts_tail: float, hp_nose: float, hp_tail: float):
        self.ssn = ssn
        self.v0 = v0
        self.tilt_angle = tilt_angle
        self.smooth_tilt = smooth_tilt
        self.b_field = b_field
        self.polarity = polarity
        self.solar_phase = solar_phase
        self.nmcr = nmcr
        self.ts_nose = ts_nose
        self.ts_tail = ts_tail
        self.hp_nose = hp_nose
        self.hp_tail = hp_tail

    def __repr__(self):
        # Keep repr concise for lists
        return (f"StaticHeliosphereRegion(ssn={self.ssn}, v0={self.v0}, "
                f"tilt_angle={self.tilt_angle}, ...)")

class StaticHeliosheathRegion:
    """Represents static parameters for one region/timestep of the heliosheath."""
    def __init__(self, k0: float, v0: float):
        self.k0 = k0
        self.v0 = v0

    def __repr__(self):
        return f"StaticHeliosheathRegion(k0={self.k0}, v0={self.v0})"


class StaticParameters:
    """
    Represents the 'static' section, holding lists of region-specific objects.
    """
    def __init__(self, heliosphere_regions: list[StaticHeliosphereRegion],
                 heliosheath_regions: list[StaticHeliosheathRegion]):
        self.heliosphere = heliosphere_regions # List of StaticHeliosphereRegion objects
        self.heliosheath = heliosheath_regions # List of StaticHeliosheathRegion objects

    def to_dict(self):
        """
        Converts static parameters back to the YAML dictionary-of-lists format.
        """
        static_dict = OrderedDict() # Use OrderedDict to maintain structure

        # Reconstruct heliosphere dictionary of lists
        hs_dict = OrderedDict()
        if self.heliosphere:
            # Get attribute names from the first object (assuming all have the same)
            attrs = vars(self.heliosphere[0]).keys()
            for attr in attrs:
                hs_dict[attr] = [getattr(region, attr) for region in self.heliosphere]
        static_dict['heliosphere'] = hs_dict

        # Reconstruct heliosheath dictionary of lists
        hsh_dict = OrderedDict()
        if self.heliosheath:
            attrs = vars(self.heliosheath[0]).keys()
            for attr in attrs:
                hsh_dict[attr] = [getattr(region, attr) for region in self.heliosheath]
        static_dict['heliosheat'] = hsh_dict # Note the key name from YAML

        return static_dict

    def __repr__(self):
        return (f"StaticParameters(heliosphere=[...{len(self.heliosphere)} regions...], "
                f"heliosheath=[...{len(self.heliosheath)} regions...])")


# --- Main Simulation Class ---

class Simulation:
    """Represents the entire simulation input configuration."""

    def __init__(self, random_seed: int, output_path: str, rigidities: list[float],
                 isotopes: dict[str, Isotope], sources: dict[str, list[float]],
                 relative_bin_amplitude: float, n_particles: int, n_regions: int,
                 dynamic: DynamicParameters, static: StaticParameters):
        self.random_seed = random_seed
        self.output_path = output_path
        self.rigidities = rigidities
        self.isotopes = isotopes # Dict {name: Isotope_object}
        self.sources = sources
        self.relative_bin_amplitude = relative_bin_amplitude
        self.n_particles = n_particles
        self.n_regions = n_regions
        self.dynamic = dynamic
        self.static = static

        # Optional: Validate n_regions consistency
        self._validate_regions()

    def _validate_regions(self):
        """Checks if n_regions matches the length of static parameter lists."""
        hs_len = len(self.static.heliosphere) if self.static and self.static.heliosphere else 0
        hsh_len = len(self.static.heliosheath) if self.static and self.static.heliosheath else 0

        if hs_len > 0 and hs_len != self.n_regions:
             print(f"Warning: n_regions ({self.n_regions}) does not match "
                   f"static heliosphere data length ({hs_len})")
        # You might have different lengths for heliosheath, depending on model
        # if hsh_len > 0 and hsh_len != self.n_regions:
        #      print(f"Warning: n_regions ({self.n_regions}) does not match "
        #            f"static heliosheath data length ({hsh_len})")


    def to_dict(self):
        """Converts the entire simulation configuration to a dictionary for YAML."""
        sim_dict = OrderedDict()
        sim_dict['random_seed'] = self.random_seed
        sim_dict['output_path'] = self.output_path
        sim_dict['rigidities'] = self.rigidities
        sim_dict['isotopes'] = OrderedDict((name, iso.to_dict()) for name, iso in self.isotopes.items())
        sim_dict['sources'] = OrderedDict(self.sources) # Use OrderedDict for consistency
        sim_dict['relative_bin_amplitude'] = self.relative_bin_amplitude
        sim_dict['n_particles'] = self.n_particles
        sim_dict['n_regions'] = self.n_regions
        sim_dict['dynamic'] = self.dynamic.to_dict()
        sim_dict['static'] = self.static.to_dict()
        return sim_dict

    def __str__(self):
        """Generates the YAML string representation of the simulation."""
        # Use sort_keys=False to respect OrderedDict insertion order
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=None, width=1000)

    def save_yaml(self, filepath: str):
        """Saves the simulation configuration to a YAML file."""
        with open(filepath, 'w') as f:
            f.write(str(self))

    @classmethod
    def from_yaml(cls, filepath_or_string):
        """Loads a Simulation object from a YAML file or string."""
        if isinstance(filepath_or_string, str) and ('\n' in filepath_or_string or '\r' in filepath_or_string):
             data = yaml.safe_load(filepath_or_string)
        else:
            try:
                with open(filepath_or_string, 'r') as f:
                    data = yaml.safe_load(f)
            except FileNotFoundError:
                 # Assume it's a string if file not found fails
                 data = yaml.safe_load(filepath_or_string)
            except Exception as e: # Catch other potential file errors
                 print(f"Error loading YAML from {filepath_or_string}: {e}")
                 # Try loading as string as a fallback
                 try:
                     data = yaml.safe_load(filepath_or_string)
                 except yaml.YAMLError as ye:
                     raise ValueError(f"Input is not a valid file path or YAML string: {ye}")


        # --- Create Isotopes ---
        isotopes_data = data.get('isotopes', {})
        isotopes = { name: Isotope(name=name, **params)
                     for name, params in isotopes_data.items() }

        # --- Create Dynamic Parameters ---
        dynamic_data = data.get('dynamic', {})
        dyn_hs_data = dynamic_data.get('heliosphere', {})
        dynamic_heliosphere = DynamicHeliosphere(k0=dyn_hs_data.get('k0', []))
        dynamic_params = DynamicParameters(heliosphere=dynamic_heliosphere)

        # --- Create Static Parameters ---
        static_data = data.get('static', {})
        static_hs_data = static_data.get('heliosphere', {})
        static_hsh_data = static_data.get('heliosheat', {}) # Note key name
        n_regions = data.get('n_regions', 0) # Get n_regions for list construction

        # Build list of StaticHeliosphereRegion objects
        heliosphere_regions = []
        if static_hs_data and n_regions > 0:
            # Get keys from the first list to ensure order/presence
            hs_keys = list(static_hs_data.keys())
            # Check if all lists have the expected length
            valid_lengths = all(len(static_hs_data.get(k, [])) == n_regions for k in hs_keys)
            if not valid_lengths:
                 raise ValueError("Inconsistent list lengths in static heliosphere data "
                                  f"compared to n_regions ({n_regions})")

            for i in range(n_regions):
                region_params = {key: static_hs_data[key][i] for key in hs_keys}
                heliosphere_regions.append(StaticHeliosphereRegion(**region_params))

        # Build list of StaticHeliosheathRegion objects
        heliosheath_regions = []
        # Heliosheath might have a different number of entries (e.g., only 1)
        hsh_keys = list(static_hsh_data.keys())
        if static_hsh_data and hsh_keys:
            # Find the length of the lists (assume they are consistent within heliosheath)
            hsh_length = len(static_hsh_data.get(hsh_keys[0], []))
            if hsh_length > 0:
                 valid_lengths = all(len(static_hsh_data.get(k, [])) == hsh_length for k in hsh_keys)
                 if not valid_lengths:
                     raise ValueError("Inconsistent list lengths in static heliosheath data")

                 for i in range(hsh_length):
                     region_params = {key: static_hsh_data[key][i] for key in hsh_keys}
                     heliosheath_regions.append(StaticHeliosheathRegion(**region_params))


        static_params = StaticParameters(
            heliosphere_regions=heliosphere_regions,
            heliosheath_regions=heliosheath_regions
        )

        # --- Create Simulation Object ---
        return cls(
            random_seed=data.get('random_seed'),
            output_path=data.get('output_path'),
            rigidities=data.get('rigidities', []),
            isotopes=isotopes,
            sources=data.get('sources', {}),
            relative_bin_amplitude=data.get('relative_bin_amplitude'),
            n_particles=data.get('n_particles'),
            n_regions=n_regions, # Use the value read from YAML
            dynamic=dynamic_params,
            static=static_params
        )

# --- Example Usage ---

# 1. Load from the YAML string provided
yaml_input = """
random_seed: 42
output_path: Proton_Deuteron_20111116_20111116_r00100_lat00000
rigidities: [1.08, 1.245, 1.42, 1.61, 1.815, 2.035, 2.275, 2.535, 2.82, 3.13, 3.465, 3.83, 4.225, 4.655, 5.125, 5.635, 6.185, 6.78, 7.425, 8.12, 8.87, 9.68, 10.55, 12.0, 14.8, 19.7, 28.15, 41.0, 59.1, 84.85]
isotopes:
  proton:
    nucleon_rest_mass: 0.938272
    mass_number: 1.0
    charge: 1.0
  deuteron:
    nucleon_rest_mass: 0.938272
    mass_number: 2.0
    charge: 1.0
sources:
  r: [1.0]
  th: [1.5707963267948966]
  phi: [0.0]
relative_bin_amplitude: 0.00855
n_particles: 1200
n_regions: 15
dynamic:
  heliosphere:
    k0:
    - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
static:
  heliosphere:
    ssn: [92.387, 89.414, 88.052, 88.193, 88.628, 85.615, 78.704, 69.687, 61.436, 54.558, 49.942, 47.34, 45.357, 42.635, 38.213]
    v0: [384.0, 363.3, 410.04, 426.3, 455.11, 464.81, 451.89, 435.44, 441.78, 416.11, 435.26, 381.93, 427.93, 381.75, 402.96]
    tilt_angle: [66.9, 69.6, 71.1, 66.7, 67.1, 64.5, 63.5, 63.7, 69.9, 65.2, 56.1, 53.6, 48.2, 48.8, 40.4]
    smooth_tilt: [66.142, 66.617, 67.408, 67.008, 66.092, 64.825, 63.267, 61.533, 58.975, 57.042, 55.583, 53.642, 52.15, 49.167, 46.442]
    b_field: [5.059, 5.741, 5.725, 5.096, 5.007, 5.281, 5.575, 5.356, 4.863, 5.986, 5.656, 4.541, 4.659, 4.246, 4.937]
    polarity: [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    solar_phase: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nmcr: [274.307, 270.473, 271.815, 274.32, 274.113, 272.754, 272.504, 273.648, 276.23, 268.111, 272.785, 281.366, 279.909, 276.43, 279.955]
    ts_nose: [72.55, 72.19, 71.8, 71.4, 70.95, 70.57, 70.23, 69.84, 69.42, 69.05, 68.69, 68.37, 68.13, 67.91, 67.72]
    ts_tail: [77.97, 77.52, 77.05, 76.58, 76.08, 75.63, 75.23, 74.74, 74.27, 73.86, 73.48, 73.25, 73.06, 72.89, 72.78]
    hp_nose: [125.31, 126.24, 127.1, 127.99, 128.82, 129.66, 130.48, 131.2, 131.92, 132.57, 133.19, 133.83, 134.43, 135.04, 135.67]
    hp_tail: [140.29, 141.22, 142.04, 142.78, 143.42, 143.99, 144.52, 145.03, 145.53, 146.09, 146.62, 147.19, 147.78, 148.38, 148.96]
  heliosheat:
    k0: [3.0e-05]
    v0: [402.96]
"""

simulation = Simulation.from_yaml(yaml_input)

# --- Access and Modify data ---
print(f"Loaded Simulation. Random Seed: {simulation.random_seed}")
print(f"Number of static heliosphere regions: {len(simulation.static.heliosphere)}")
print(f"SSN of the first static region: {simulation.static.heliosphere[0].ssn}")
print(f"Number of static heliosheath regions: {len(simulation.static.heliosheath)}")
print(f"k0 of the first heliosheath region: {simulation.static.heliosheath[0].k0}")
print(f"Proton charge: {simulation.isotopes['proton'].charge}")
print("-" * 20)

# Modify some data
simulation.random_seed = 101
simulation.isotopes['proton'].charge = 1.1 # Example modification
simulation.static.heliosphere[0].ssn = 95.0
simulation.static.heliosheath.append(StaticHeliosheathRegion(k0=5.0e-5, v0=410.0)) # Add another heliosheath region

print("Modified simulation object.")
print(f"New Random Seed: {simulation.random_seed}")
print(f"New SSN of the first static region: {simulation.static.heliosphere[0].ssn}")
print(f"New number of static heliosheath regions: {len(simulation.static.heliosheath)}")
print("-" * 20)

# --- Generate YAML Output ---
print("Generated YAML output:")
print(str(simulation))

# --- Save to a file ---
output_filename = "simulation_output.yaml"
simulation.save_yaml(output_filename)
print(f"\nSimulation configuration saved to {output_filename}")

# --- Verify loading from the saved file ---
loaded_simulation = Simulation.from_yaml(output_filename)
print(f"\nSuccessfully loaded simulation from {output_filename}.")
print(f"Loaded Random Seed: {loaded_simulation.random_seed}")
assert loaded_simulation.random_seed == simulation.random_seed # Basic check
assert len(loaded_simulation.static.heliosheath) == len(simulation.static.heliosheath)