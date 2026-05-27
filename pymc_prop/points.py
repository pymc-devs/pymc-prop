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
    :math:`\\vartheta^{(j)} \\in \\Theta` in Sec. 5 of McLatchie et al. 2025).
    ``start_point`` is typically ``model.initial_point()`` in ``value_vars`` order.
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
    batch_shape = flat_particles.shape[:-1]  # () for one particle, (p,) for a batch
    for _name, shape, size, _dtype in point_map_info:
        stop = start + size
        part = flat_particles[..., start:stop]
        shape_tail = pt.as_tensor(np.asarray(shape, dtype="int64"))
        target_shape = pt.concatenate([batch_shape, shape_tail], axis=0)
        ndim = flat_particles.ndim - 1 + len(shape)
        part = pt.reshape(part, target_shape, ndim=ndim)
        out.append(part)
        start = stop
    return out


def make_point_mapper(model=None) -> PointMapper:
    """Build a :class:`PointMapper` from ``model.initial_point()`` and ``value_vars``.

    Ravel/unravel uses PyMC's :class:`~pymc.blocking.DictToArrayBijection` so particle
    coordinates match the unconstrained parameterisation used in the WGF (Sec. 5,
    McLatchie et al. 2025, https://arxiv.org/abs/2510.01915).
    """
    model = modelcontext(model)
    start_point = model.initial_point()
    ordered_point = {var.name: np.asarray(start_point[var.name]) for var in model.value_vars}
    raveled = DictToArrayBijection.map(ordered_point)
    return PointMapper(start_point=ordered_point, point_map_info=raveled.point_map_info)
