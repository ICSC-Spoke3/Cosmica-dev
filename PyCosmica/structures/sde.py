from typing import NamedTuple

from jax.typing import ArrayLike


class ConvectionDiffusionTensor(NamedTuple):
    rr: ArrayLike
    tr: ArrayLike
    tt: ArrayLike
    pr: ArrayLike
    pt: ArrayLike
    pp: ArrayLike
    DKrr_dr: ArrayLike
    DKtr_dt: ArrayLike
    DKrt_dr: ArrayLike
    DKtt_dt: ArrayLike
    DKrp_dr: ArrayLike
    DKtp_dt: ArrayLike


class DiffusionTensor(NamedTuple):
    rr: ArrayLike
    tr: ArrayLike
    tt: ArrayLike
    pr: ArrayLike
    pt: ArrayLike
    pp: ArrayLike


class vect3D(NamedTuple):
    r: ArrayLike
    th: ArrayLike
    phi: ArrayLike
