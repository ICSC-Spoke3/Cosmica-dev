from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from PyCosmica.structures import ParticleDescription


def beta_R(R: ArrayLike, p: ParticleDescription) -> Array:
    return R / jnp.sqrt(R ** 2 + (p.T0 * p.A / p.Z) ** 2)


def en_to_rig(t, mass_number=1., z=1.):
    """
    Convert energy to rigidity
    :param t: energy
    :param mass_number: mass number
    :param z: charge
    :return: rigidity
    """

    t0 = 0.931494061
    if jnp.fabs(z) == 1:
        t0 = 0.938272046
    if mass_number == 0:
        t0 = 5.11e-4
        mass_number = 1
    return mass_number / jnp.fabs(z) * jnp.sqrt(t * (t + 2. * t0))


def rig_to_en(r, mass_number=1., z=1.):
    """
    Convert rigidity to energy
    :param r: rigidity
    :param mass_number: mass number
    :param z: charge
    :return: energy
    """

    t0 = 0.931494061
    if jnp.fabs(z) == 1:
        t0 = 0.938272046
    if mass_number == 0:
        t0 = 5.11e-4
        mass_number = 1
    return jnp.sqrt((z * z) / (mass_number * mass_number) * (r * r) + (t0 * t0)) - t0


def smooth_transition(initial_val, final_val, center_of_transition, smoothness, x):
    """
    Smooth transition between InitialVal to FinalVal centered at CenterOfTransition as function of x
    If smoothness == 0 use a sharp transition (tanh), otherwise use a smooth transition
    :param initial_val: initial value
    :param final_val: final value
    :param center_of_transition: center of transition
    :param smoothness: smoothness
    :param x: x value
    :return: the transition value
    """

    if smoothness == 0:
        return final_val if x >= center_of_transition else initial_val
    else:
        return (initial_val + final_val) / 2. - (initial_val - final_val) / 2. * jnp.tanh(
            (x - center_of_transition) / smoothness)
