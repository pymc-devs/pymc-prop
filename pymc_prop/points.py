"""Utilities for mapping between PyMC points and flat particle arrays."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytensor.tensor as pt

from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.model import modelcontext


PointType = Dict[str, np.ndarray]


@dataclass
class PointMapper:
    """Bidirectional map between PyMC ``value_vars`` points and flat particles.

    Particles live in unconstrained ``value_vars`` space (state
    :math:`\\vartheta^{(j)} \\in \\Theta`, McLatchie et al. 2025, Sec. 5).
    ``start_point`` comes from ``model.initial_point()``: prior-centred values
    in ``value_vars`` order after PyMC transforms.
    """

    start_point: PointType
    point_map_info: tuple

    def ravel(self, point: PointType) -> np.ndarray:
        return DictToArrayBijection.map(point).data

    def unravel(self, array: np.ndarray) -> PointType:
        raveled = RaveledVars(np.asarray(array, dtype=float), self.point_map_info)
        return DictToArrayBijection.rmap(raveled, start_point=self.start_point)


def flat_to_value_vars(
    flat_particles: pt.TensorVariable,
    point_map_info: tuple[tuple[str, tuple[int, ...], int, np.dtype], ...],
) -> list[pt.TensorVariable]:
    """Map a flat particle vector to PyMC ``value_vars`` tensors."""
    out: list[pt.TensorVariable] = []
    start = 0
    # flat_particles may be a 1-D vector (single particle) or a 2-D matrix
    # (batch, dim). Construct target shape explicitly to avoid ambiguous
    # concatenation of Python-level shape tuples with tensor shapes that
    # can produce symbolic Squeeze/Subtensor nodes during graph rewrites.
    for _name, shape, size, _dtype in point_map_info:
        stop = start + size
        part = flat_particles[..., start:stop]
        shape_tail = pt.as_tensor(np.asarray(shape, dtype="int64"))
        # Single particle (1-D) or one batch axis (2-D matrix) only; ndim is
        # fixed at graph build time (scan rows vs batched compile).
        if flat_particles.ndim == 1:
            target_shape = shape_tail
        else:
            batch_size = pt.shape(flat_particles)[0]
            batch_size_vec = pt.reshape(batch_size, (1,))
            target_shape = pt.concatenate([batch_size_vec, shape_tail], axis=0)
        ndim = flat_particles.ndim - 1 + len(shape)
        part = pt.reshape(part, target_shape, ndim=ndim)
        out.append(part)
        start = stop
    return out


def require_unconstrained_free_rvs(model=None) -> None:
    """Require free RVs without PyMC's default constrained-space transforms.

    TODO: ``model`` has free RVs with implicit transforms are
    rejected until the change-of-variables story for PrO WGF on transformed
    parameters is documented.
    """
    model = modelcontext(model)
    transformed = [
        rv.name
        for rv in model.free_RVs
        if model.rvs_to_transforms[rv] is not None
    ]
    if transformed:
        names = ", ".join(repr(n) for n in transformed)
        raise ValueError(
            "PrO sampling requires native unconstrained free RVs; "
            f"found transformed: [{names}]. "
        )


def make_point_mapper(model=None) -> PointMapper:
    """Build a :class:`PointMapper` from ``model.initial_point()`` and ``value_vars``.
    """
    model = modelcontext(model)
    require_unconstrained_free_rvs(model)
    start_point = model.initial_point()
    ordered_point = {var.name: np.asarray(start_point[var.name]) for var in model.value_vars}
    raveled = DictToArrayBijection.map(ordered_point)
    return PointMapper(start_point=ordered_point, point_map_info=raveled.point_map_info)
