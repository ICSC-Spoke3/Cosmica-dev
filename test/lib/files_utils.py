from pathlib import Path
from typing import NamedTuple, Optional, Callable, Union

import astropy.io.fits as pyfits
import numpy as np

from . import func
from .isotopes import Isotope, ISOTOPES, Ion, IONS
from .physics_utils import RigidityVec, FluxVec, RigidityFlux, EnergyFlux, NDArrayBase, EnergyVec

import yaml

yaml.Dumper.ignore_aliases = lambda *args: True


class InlineList(list):
    @staticmethod
    def inline_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(InlineList, InlineList.inline_list_representer)


class LisLoader:
    def __init__(self, path):
        hdulist = pyfits.open(path)
        data = hdulist[0].data
        r_sun = 8.33
        r = np.arange(int(hdulist[0].header["NAXIS1"])) * hdulist[0].header["CDELT1"] + hdulist[0].header["CRVAL1"]
        if r[0] > r_sun:
            indexes = [0]
            weights = [1]
        elif r[-1] <= r_sun:
            indexes = [-1]
            weights = [1]
        else:
            i = np.where((r[:-1] <= r_sun) & (r_sun < r[1:]))[0][0]
            indexes = [i, i + 1]
            weights = [(r[i + 1] - r_sun) / (r[i + 1] - r[i]), (r_sun - r[i]) / (r[i + 1] - r[i])]

        energy = 10 ** (
                float(hdulist[0].header["CRVAL3"]) +
                np.arange(int(hdulist[0].header["NAXIS3"])) *
                float(hdulist[0].header["CDELT3"])
        )

        particle_flux = {}

        n_nuclei = hdulist[0].header["NAXIS4"]
        for i in range(1, n_nuclei + 1):
            id_ = "%03d" % i
            z = int(hdulist[0].header["NUCZ" + id_])
            a = int(hdulist[0].header["NUCA" + id_])
            k = int(hdulist[0].header["NUCK" + id_])

            if z not in particle_flux:
                particle_flux[z] = {}
            if a not in particle_flux[z]:
                particle_flux[z][a] = {}
            if k not in particle_flux[z][a]:
                particle_flux[z][a][k] = []

            d = ((data[i - 1, :, 0, indexes].swapaxes(0, 1)) * np.array(weights)).sum(axis=1)
            particle_flux[z][a][k].append(1e7 * d / energy ** 2)

        energy = energy / 1e3
        hdulist.close()
        self.energy = energy
        self.flux = particle_flux

    def __getitem__(self, isotope: Isotope) -> EnergyFlux:
        z, a, k = isotope.Z, isotope.A, 0
        tk_lis_spectra = self.flux[z][a][k][-1]
        for sec_ind in range(len(self.flux[z][a][k]) - 1):
            tk_lis_spectra = tk_lis_spectra + self.flux[z][a][k][sec_ind]
        return EnergyFlux(EnergyVec(self.energy), FluxVec(tk_lis_spectra))


class ModulationResult(NamedTuple):
    rigidity: RigidityVec
    flux: FluxVec
    lis: FluxVec

    @property
    def rig_flux(self):
        return RigidityFlux(self.rigidity, self.flux)

    @property
    def lis_flux(self):
        return RigidityFlux(self.rigidity, self.lis)

    def trim(self, low, high):
        indexes = (self.rigidity > low) & (self.rigidity < high)
        return ModulationResult(
            RigidityVec(self.rigidity[indexes]),
            FluxVec(self.flux[indexes]),
            FluxVec(self.lis[indexes]))


class IsotopeOutput(NamedTuple):
    input_rig: RigidityVec
    output_rig: list[RigidityVec]
    output_dist: list[np.ndarray]
    n_particles: np.ndarray

    @classmethod
    def from_yaml(cls, histograms: list[dict]) -> 'IsotopeOutput':
        input_rig_ = []
        output_rig_ = []
        output_dist_ = []
        n_particles_ = []
        for his in histograms:
            rig, amp, lb, n_reg, dist = his['rigidity'], his['amplitude'], his['lower_bound'], his['n_reg'], his['bins']
            input_rig_.append(rig)
            n_particles_.append(n_reg)
            bins = lb + np.arange(len(dist)) * amp
            output_rig_.append(RigidityVec((10 ** bins + 10 ** (bins + amp)) / 2))
            output_dist_.append(np.array(dist))
        return cls(RigidityVec(input_rig_), output_rig_, output_dist_, np.array(n_particles_))

    @classmethod
    def from_txt(cls, histograms: str) -> 'IsotopeOutput':
        input_rig_ = []
        output_rig_ = []
        output_dist_ = []
        n_particles_ = []

        lines = list(map(str.split, filter(lambda l: l and not l.startswith("#"), histograms.split('\n'))))
        n_bins = int(lines[0][0])

        for spec, dist in zip(lines[1::2], lines[2::2]):
            e_gen, n_part_gen, n_part_reg, n_bin_out, bin_low, bin_amp = spec[:6]
            input_rig_.append(float(e_gen))
            n_particles_.append(int(n_part_reg))
            bin_low, bin_amp = map(float, (bin_low, bin_amp))
            bins = bin_low + np.arange(int(n_bin_out)) * bin_amp
            output_rig_.append(RigidityVec((10 ** bins + 10 ** (bins + bin_amp)) / 2))
            output_dist_.append(np.array(list(map(float, dist))))

        assert 2 * n_bins == len(lines) - 1 and n_bins == len(input_rig_)

        return cls(RigidityVec(input_rig_), output_rig_, output_dist_, np.array(n_particles_))

    def modulate(self, lis: EnergyFlux, isotope: Isotope) -> tuple[RigidityFlux, RigidityFlux]:
        lis_rig = lis.energy.to_rigidity(isotope)
        lis_flux_en = lis.flux
        lis_flux_en_in = func.lin_log_interpolation(lis_rig, lis_flux_en, self.input_rig)

        un_norm_flux = np.zeros(len(self.input_rig))
        for index_rig in range(len(self.input_rig)):
            lis_flux_en_out = func.lin_log_interpolation(lis_rig, lis_flux_en, self.output_rig[index_rig])

            for outer_rig_bin, boundary_bin, lis_flux_bin in zip(self.output_rig[index_rig],
                                                                 self.output_dist[index_rig],
                                                                 lis_flux_en_out):
                un_norm_flux[index_rig] += boundary_bin * lis_flux_bin / outer_rig_bin ** 2

        conv_coeff = func.d_rig_to_en(
            self.input_rig.to_energy(isotope),
            self.input_rig,
            isotope.Z, isotope.A
        )
        j_mod = conv_coeff * [UnFlux / Npart * R ** 2 for R, UnFlux, Npart in
                              zip(self.input_rig, un_norm_flux, self.n_particles)]
        lis_flux_rig_in = conv_coeff * lis_flux_en_in

        return (RigidityFlux(self.input_rig.copy(), j_mod),
                RigidityFlux(self.input_rig.copy(), lis_flux_rig_in))


class SingleOutput(dict[Isotope, IsotopeOutput]):
    @classmethod
    def from_yaml(cls, parametrization: dict[str, list[dict]]) -> 'SingleOutput':
        return cls({
            ISOTOPES.get(iso): IsotopeOutput.from_yaml(histograms)
            for iso, histograms in parametrization.items()
        })

    @classmethod
    def from_txts(cls, txts: dict[str, str]) -> 'SingleOutput':
        return cls({
            ISOTOPES.get(iso): IsotopeOutput.from_txt(txt)
            for iso, txt in txts.items()
        })

    def modulate(self, lis_loader: LisLoader) -> ModulationResult:
        rig: Optional[RigidityVec] = None
        flux: Optional[FluxVec] = None
        lis_rig_flux: Optional[FluxVec] = None

        for isotope, output in self.items():
            lis_spectrum = lis_loader[isotope]
            j_rig_flux, j_lis_rig_flux = output.modulate(lis_spectrum, isotope)

            if rig is None:
                rig = j_rig_flux.rigidity
                flux = FluxVec(np.zeros_like(rig))
                lis_rig_flux = FluxVec(np.zeros_like(rig))

            flux += j_rig_flux.flux
            lis_rig_flux += j_lis_rig_flux.flux

        return ModulationResult(rig, flux, lis_rig_flux)


class SimulationOutput(list[SingleOutput]):
    @classmethod
    def from_yaml(cls, yml: dict) -> 'SimulationOutput':
        return SimulationOutput([
            SingleOutput.from_yaml(parametrization)
            for parametrization in yml['histograms']
        ])

    @classmethod
    def from_txt(cls, txts: list[dict[str, str]]) -> 'SimulationOutput':
        return SimulationOutput([
            SingleOutput.from_txts(txt)
            for txt in txts
        ])

    @classmethod
    def from_outputs(cls, *outputs: Union[SingleOutput, 'SimulationOutput']) -> 'SimulationOutput':
        return SimulationOutput([o if isinstance(o, SingleOutput) else o[0] for o in outputs])

    def modulate(self, lis_loader: LisLoader) -> list[ModulationResult]:
        return [single_output.modulate(lis_loader) for single_output in self]


class ExperimentalData(NamedTuple):
    rigidity: RigidityVec
    flux: FluxVec
    limits: Optional[tuple[FluxVec, FluxVec]] = None

    @property
    def rig_flux(self) -> RigidityFlux:
        return RigidityFlux(self.rigidity, self.flux)

    @classmethod
    def from_data(cls, path: Path, cols: tuple[int, int] | tuple[int, int, int, int], rig_range=(0, 11),
                  to_rig: Optional[Isotope] = None) -> 'ExperimentalData':
        assert rig_range[0] < rig_range[1]
        assert len(cols) in (2, 4)

        rig_col = cols[0]
        rig_low, rig_high = rig_range

        exp_data = np.loadtxt(str(path))

        if to_rig is not None:
            tkin = exp_data[:, rig_col]
            rigi = func.en_to_rig(tkin, to_rig.Z, to_rig.A)
            factors = func.rig_to_en_flux_factor(tkin, rigi, to_rig.Z, to_rig.A)
            for c in cols[1:]:
                exp_data[:, c] *= factors
            exp_data[:, rig_col] = rigi

        filtered = exp_data[(rig_low <= exp_data[:, rig_col]) & (exp_data[:, rig_col] <= rig_high)][:, cols]
        rigidity, flux = RigidityVec(filtered[:, 0]), FluxVec(filtered[:, 1])

        if len(cols) == 2:
            return cls(rigidity, flux)

        limits = (FluxVec(filtered[:, 2]), FluxVec(filtered[:, 3]))
        return cls(rigidity, flux, limits)


class SimulationExperimentItem(NamedTuple):
    name: str
    ions: list[Ion]
    period: tuple[int, int]
    sources: tuple[np.ndarray, np.ndarray, np.ndarray]
    experimental_data_path: str


class SimulationPredictionItem(NamedTuple):
    name: str
    ions: list[Ion]
    period: tuple[int, int]
    sources: tuple[np.ndarray, np.ndarray, np.ndarray]
    rigidities: RigidityVec


class SimulationList(list[SimulationExperimentItem | SimulationPredictionItem]):
    @classmethod
    def from_listfile(cls, file: Path) -> 'SimulationList':
        def parse(line):
            parsed = [x.strip() for x in line.replace("\t", "").split("|")[:8]]

            def prs(x, rad: float | None = None):
                arr = np.array(list(map(float, x.split(','))))
                return arr if rad is None else (
                    np.radians(arr) if rad == 0 else np.radians(rad - arr)
                )

            return SimulationExperimentItem(
                name=parsed[0],
                ions=[IONS.get(i.lower().strip()) for i in parsed[1].split(",")],
                period=(int(parsed[3]), int(parsed[4])),
                sources=(prs(parsed[5]), prs(parsed[6], 90), prs(parsed[7], 0)),
                experimental_data_path=parsed[2]
            )

        with open(file) as f:
            return cls([
                parse(line)
                for line in filter(lambda x: not x.startswith('#'), f.read().splitlines())
            ])


class HeliosphericParameters(NDArrayBase):
    @classmethod
    def from_files(cls, *paths: Path | str) -> 'HeliosphericParameters':
        arrays = [np.loadtxt(path) for path in paths]
        arrays = np.concatenate(arrays, axis=0)
        sorted_indexes = np.lexsort((arrays[:, 1], arrays[:, 0]))
        return cls(arrays[sorted_indexes][::-1].copy())

    def in_period(self, period: tuple[int, int], n_regions: int = 15) -> tuple[np.ndarray, np.ndarray]:
        cr_ini, cr_end = self[:, 0], self[:, 1]
        cr_ord = np.arange(len(cr_ini))
        # if between start and end
        mask1 = (cr_ini <= period[1]) & (cr_end > period[0])
        # if between (start - num regions) and end
        mask2 = (cr_ini <= period[1]) & (cr_end[cr_ord - (n_regions - 1)] > period[0])
        # if before (start - num regions)
        mask3 = np.append((cr_ord - (n_regions - 1) >= 0) & (cr_ini[cr_ord - (n_regions - 1)] < period[0]), 1)
        # remove all after first True
        mask3 = cr_ord <= np.argwhere(mask3)[0, 0]

        return np.array(self[mask2 & mask3]), np.array(self[np.bool(np.roll(mask1 & mask3, n_regions - 1))])


class SimulationInput(NamedTuple):
    class DynamicParameters(NamedTuple):
        class DynamicHeliosphere(NamedTuple):
            k0: list[np.ndarray]

        heliosphere: DynamicHeliosphere

        def to_dict(self):
            return {
                'heliosphere': {k: [InlineList(v.tolist()) for v in vv] for k, vv in self.heliosphere._asdict().items()}
            }

    class StaticParameters(NamedTuple):
        class StaticHeliosphere(NamedTuple):
            ssn: np.ndarray
            v0: np.ndarray
            tilt_angle: np.ndarray
            smooth_tilt: np.ndarray
            b_field: np.ndarray
            polarity: np.ndarray
            solar_phase: np.ndarray
            nmcr: np.ndarray
            ts_nose: np.ndarray
            ts_tail: np.ndarray
            hp_nose: np.ndarray
            hp_tail: np.ndarray

        class StaticHeliosheat(NamedTuple):
            k0: np.ndarray
            v0: np.ndarray

        heliosphere: StaticHeliosphere
        heliosheat: StaticHeliosheat

        def to_dict(self):
            return {
                'heliosphere': {k: InlineList(v.tolist()) for k, v in self.heliosphere._asdict().items()},
                'heliosheat': {k: InlineList(v.tolist()) for k, v in self.heliosheat._asdict().items()}
            }

    random_seed: int
    output_path: str
    rigidities: RigidityVec
    isotopes: list[Isotope]
    sources: tuple[np.ndarray, np.ndarray, np.ndarray]
    n_particles: int
    n_regions: int
    dynamic: DynamicParameters
    static: StaticParameters
    relative_bin_amplitude: float = 0.00855

    def to_txt(self, output_path_map: Optional[Callable[[int, Isotope, str], str]] = None) -> list[dict[Isotope, str]]:
        return [
            {
                isotope: '\n'.join(
                    [
                        f'RandomSeed: {self.random_seed}',
                        f'OutputFilename: {self.output_path if output_path_map is None else output_path_map(i, isotope, self.output_path)}',
                        f'Particle_Charge: {isotope.Z}',
                        f'Particle_MassNumber: {isotope.A}',
                        f'Particle_NucleonRestMass: {isotope.T0}',
                        f'Tcentr: {", ".join(f"{x:.3e}" for x in self.rigidities)}',
                        f'SourcePos_r: {", ".join(f"{x:.5f}" for x in self.sources[0])}',
                        f'SourcePos_theta: {", ".join(f"{x:.5f}" for x in self.sources[1])}',
                        f'SourcePos_phi: {", ".join(f"{x:.5f}" for x in self.sources[2])}',
                        f'Npart: {self.n_particles * len(self.sources[0])}',
                        f'Nregions: {self.n_regions}'
                    ] + [
                        'HeliosphericParameters: {:.6e}, {:.3f}, {:.2f}, {:.2f}, {:.3f}, {:.3f}, {:.0f}, {:.0f}, {:.3f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
                            *hp
                        ) for hp in zip(*dyn, *self.static.heliosphere._asdict().values())
                    ] + [
                        'HeliosheatParameters: {:.5e}, {:.2f}'.format(*hs)
                        for hs in zip(*self.static.heliosheat._asdict().values())
                    ]
                ) for isotope in self.isotopes
            } for i, dyn in enumerate(zip(*self.dynamic.heliosphere._asdict().values()))
        ]

    def to_dict(self) -> dict:
        return {
            'random_seed': self.random_seed,
            'output_path': str(self.output_path),
            'rigidities': InlineList(self.rigidities.tolist()),
            'isotopes': {iso.name: {
                'nucleon_rest_mass': iso.T0,
                'mass_number': iso.A,
                'charge': iso.Z,
            } for iso in self.isotopes},
            'sources': {
                'r': InlineList(self.sources[0].tolist()),
                'th': InlineList(self.sources[1].tolist()),
                'phi': InlineList(self.sources[2].tolist()),
            },
            'relative_bin_amplitude': self.relative_bin_amplitude,
            'n_particles': self.n_particles,
            'n_regions': self.n_regions,
            'dynamic': self.dynamic.to_dict(),
            'static': self.static.to_dict(),
        }


def estimate_k0(inpt: SimulationInput) -> list[tuple[float, float]]:
    hs = inpt.static.heliosphere
    return [
        func.eval_k0(float(hs.tilt_angle[period:period + inpt.n_regions].mean()) >= 50,
                     hs.polarity[period], inpt.isotopes[0].Z, hs.solar_phase[period],
                     hs.smooth_tilt[period], hs.nmcr[period], hs.ssn[period])
        for period in range(len(hs.tilt_angle) - inpt.n_regions + 1)
    ]
