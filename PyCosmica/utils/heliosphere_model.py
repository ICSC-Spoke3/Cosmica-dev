import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from PyCosmica.structures import SimulatedHeliosphere, HeliosphereBoundRadius, Position3D


def boundary(th: ArrayLike, phi: ArrayLike, a: ArrayLike, b: ArrayLike) -> Array:
    x = jnp.cos(phi) * jnp.sin(th)
    y = jnp.sin(phi) * jnp.sin(th)
    z = jnp.cos(th)
    cos_alpha = x * -0.996 + y * 0.03 + z * 0.088

    return lax.select(cos_alpha > 0, b - (b - a) * cos_alpha ** 2, b)

def radial_zone(bound: HeliosphereBoundRadius, N_regions: ArrayLike, pos: Position3D) -> Array:
    r, th, phi = pos.r, pos.th, pos.phi
    R_hp = boundary(th, phi, bound.R_hp_nose, bound.R_hp_tail)
    R_ts = boundary(th, phi, bound.R_ts_nose, bound.R_ts_tail)

    zone = lax.select(
        r < R_hp,
        lax.select(
            r >= R_ts,
            N_regions,  # Inside heliosheath
            lax.select(
                r < bound.R_ts_nose,
                jnp.floor(r / bound.R_ts_nose * N_regions).astype(int),
                N_regions - 1
            )
        ),
        -1  # Outside heliosphere
    )

    return zone

def select_np(cond, tr, fs):
    return tr if cond else fs

def boundary_scalar(th: float, phi: float, a: float, b: float) -> float:
    x = np.cos(phi) * np.sin(th)
    y = np.sin(phi) * np.sin(th)
    z = np.cos(th)
    cos_alpha = x * -0.996 + y * 0.03 + z * 0.088

    return select_np(cos_alpha > 0, b - (b - a) * cos_alpha ** 2, b)

def radial_zone_scalar(bound: HeliosphereBoundRadius, N_regions: float, pos: Position3D) -> int:
    r, th, phi = pos.r, pos.th, pos.phi
    R_hp = boundary_scalar(th, phi, bound.R_hp_nose, bound.R_hp_tail)
    R_ts = boundary_scalar(th, phi, bound.R_ts_nose, bound.R_ts_tail)

    zone = select_np(
        r < R_hp,
        select_np(
            r >= R_ts,
            N_regions,  # Inside heliosheath
            select_np(
                r < bound.R_ts_nose,
                np.floor(r / bound.R_ts_nose * N_regions).astype(int),
                N_regions - 1
            )
        ),
        -1  # Outside heliosphere
    )

    return int(zone)