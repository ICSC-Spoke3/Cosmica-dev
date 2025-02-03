from jax import Array, lax

from PyCosmica.structures import PropagationState, DiffusionTensor, PropagationConstantsItem
from PyCosmica.utils import beta_R
from PyCosmica.utils.diffusion_model import diffusion_tensor_hmf_frame, diffusion_coeff_heliosheat


def diffusion_tensor_symmetric(state: PropagationState, const: PropagationConstantsItem, w: Array) -> DiffusionTensor:
    def in_heliosphere():
        hmf = diffusion_tensor_hmf_frame(state, const, beta_R(state.R, const.particle), w)
        #TODO: tbc
        return DiffusionTensor(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)

    def in_heliosheat():
        rr, dKrr_dr = diffusion_coeff_heliosheat(state, const, beta_R(state.R, const.particle))
        return DiffusionTensor(rr, 0., 0., 0., 0., 0., dKrr_dr, 0., 0., 0., 0., 0.)

    return lax.cond(state.rad_zone < const.N_regions, in_heliosphere, in_heliosheat)
