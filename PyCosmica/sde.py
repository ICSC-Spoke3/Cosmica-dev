from jax import Array, lax, numpy as jnp

from PyCosmica.structures import PropagationState, DiffusionTensor, PropagationConstantsItem, cos_polar_zone, PI
from PyCosmica.utils import beta_R
from PyCosmica.utils.diffusion_model import diffusion_tensor_hmf_frame, diffusion_coeff_heliosheat
from PyCosmica.utils.solar_wind import solar_wind_speeed


def diffusion_tensor_symmetric(state: PropagationState, const: PropagationConstantsItem, w: Array) -> DiffusionTensor:
    def in_heliosphere():
        hmf = diffusion_tensor_hmf_frame(state, const, beta_R(state.R, const.particle), w)
        is_polar_region = jnp.fabs(jnp.cos(state.th)) > cos_polar_zone

        pol_sign = lax.select((state.th - PI / 2.) > 0, -1, 1)
        sign_A_sun = lax.select(const.LIM.A_sun > 0, 1, -1)

        V_sw = solar_wind_speeed(state, const)

        # TODO: tbc
        return DiffusionTensor(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)

    def in_heliosheat():
        rr, dKrr_dr = diffusion_coeff_heliosheat(state, const, beta_R(state.R, const.particle))
        return DiffusionTensor(rr, 0., 0., 0., 0., 0., dKrr_dr, 0., 0., 0., 0., 0.)

    return lax.cond(state.rad_zone < const.N_regions, in_heliosphere, in_heliosheat)
