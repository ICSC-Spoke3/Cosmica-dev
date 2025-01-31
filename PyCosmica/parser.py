from collections import defaultdict
from typing import Any

from PyCosmica.structures import *
from PyCosmica.utils import eval_k0, g_low_comp, r_const_comp, a_sum_comp, eval_p0_drift_suppression_factor, \
    eval_high_rigidity_drift_suppression_plateau


def format_to_dict(
        itr,
        dup_keys: tuple = (),
        float_arr_keys: tuple = (),
        float_list_keys: tuple = (),
        str_keys: tuple = (),
        int_keys: tuple = (),
) -> dict[Any]:
    out = defaultdict(list)
    for k, v in itr:
        if k in str_keys:
            v = str(v).strip()
        elif k in int_keys:
            v = int(v)
        elif k in float_arr_keys:
            v = jnp.array(list(map(float, v.split(','))))
        elif k in float_list_keys:
            v = list(map(float, v.split(',')))
        else:
            v = float(v)

        if k in dup_keys:
            out[k].append(v)
        else:
            out[k] = v
    return dict(out)


def compute_simulated_heliosphere(
        heliospheric_parameters: list[HeliosphericParameters], N_regions: int
) -> SimulatedHeliosphere:
    bounds = []
    activity = []
    for i, hp in enumerate(heliospheric_parameters):
        aver_tilt = sum(x.tilt_angle for x in heliospheric_parameters[i:i + N_regions]) / N_regions
        activity.append(aver_tilt >= tilt_L_max_activity_threshold)
        bounds.append(hp.heliosphere_bound_radius)
    return SimulatedHeliosphere(
        N_regions=N_regions,
        R_boundary_effe=[],
        R_boundary_real=bounds,
        is_high_activity_period=activity,
    )


def heliospheric_parameters_to_properties(params: list[HeliosphericParameters], Z: int) -> list[HeliosphereProperties]:
    props = []
    for hp in params:
        if hp.K0 > 0:
            K0_paral = K0_perp = (hp.K0, hp.K0)
            gauss_var = (0, 0)
        else:
            K0_high = eval_k0(True, hp.polarity, Z, hp.solar_phase, hp.smooth_tilt, hp.NMCR, hp.ssn)
            K0_low = eval_k0(True, hp.polarity, Z, hp.solar_phase, hp.smooth_tilt, hp.NMCR, hp.ssn)
            K0_paral, K0_perp, gauss_var = [(K0_high[i], K0_low[i]) for i in range(3)]
        props.append(
            HeliosphereProperties(
                V0=hp.V0 / AU_KM,
                K0_paral=K0_paral,
                K0_perp=K0_perp,
                gauss_var=gauss_var,
                g_low=g_low_comp(hp.solar_phase, hp.polarity, hp.smooth_tilt),
                r_const=r_const_comp(hp.solar_phase, hp.polarity, hp.smooth_tilt),
                tilt_angle=hp.tilt_angle * PI / 180.,
                A_sun=a_sum_comp(hp.V0, hp.B_earth, hp.polarity),
                P0_d=eval_p0_drift_suppression_factor(True, hp.solar_phase, hp.tilt_angle, 0),
                P0_dNS=eval_p0_drift_suppression_factor(False, hp.solar_phase, hp.tilt_angle, hp.ssn),
                plateau=eval_high_rigidity_drift_suppression_plateau(hp.solar_phase, hp.tilt_angle),
                polarity=0,  # TODO: check
            )
        )
    return props


def heliosheat_parameters_to_properties(params: list[HeliosheatParameters]) -> list[HeliosheatProperties]:
    return [HeliosheatProperties(V0=hp.V0 / AU_KM, K0=hp.K0) for hp in params]


def parse(filename):
    with open(filename) as f:
        lines = f.readlines()
    json = format_to_dict(
        map(lambda s: s.replace('\n', '').split(': '), filter(lambda s: not s.startswith('#'), lines)),
        str_keys=('OutputFilename',),
        int_keys=('Npart', 'Nregions'),
        float_list_keys=(
            'Tcentr', 'SourcePos_theta', 'SourcePos_phi', 'SourcePos_r',
            'HeliosphericParameters', 'HeliosheatParameters',
        ),
        dup_keys=('HeliosphericParameters', 'HeliosheatParameters'),
    )

    json['InitialPosition'] = list(
        map(AdvectiveDrift.from_list, zip(json['SourcePos_r'], json['SourcePos_theta'], json['SourcePos_phi']))
    )
    json['HeliosphericParameters'] = list(map(HeliosphericParameters.from_list, json['HeliosphericParameters']))
    json['HeliosheatParameters'] = list(map(HeliosheatParameters.from_list, json['HeliosheatParameters']))
    json['HeliosphereToBeSimulated'] = compute_simulated_heliosphere(json['HeliosphericParameters'], json['Nregions'])
    json['HeliosphereProperties'] = heliospheric_parameters_to_properties(json['HeliosphericParameters'],
                                                                              json['Particle_Charge'])
    json['HeliosheatProperties'] = heliosheat_parameters_to_properties(json['HeliosheatParameters'])

    sim_params = SimParameters(
        output_file_name=json['OutputFilename'],
        N_part=json['Npart'],
        N_T=len(json['Tcentr']),
        N_initial_positions=len(json['SourcePos_r']),
        T_centr=json['Tcentr'],
        initial_position=json['InitialPosition'],
        ion_to_be_simulated=PartDescription(
            T0=json['Particle_NucleonRestMass'],
            Z=json['Particle_Charge'],
            A=json['Particle_MassNumber'],
        ),
        results=None,
        relative_bin_amplitude=None,
        heliosphere_to_be_simulated=json['HeliosphereToBeSimulated'],
        prop_medium=json['HeliosphereProperties'],
        prop_heliosheat=json['HeliosheatProperties'],
    )
    print(sim_params)
    return sim_params
