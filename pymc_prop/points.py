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
    """Bidirectional map between PyMC value-space points and flat particles."""

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
    """Build a mapper using PyMC's unconstrained ``value_vars`` ordering."""
    model = modelcontext(model)
    start_point = model.initial_point()
    ordered_point = {var.name: np.asarray(start_point[var.name]) for var in model.value_vars}
    raveled = DictToArrayBijection.map(ordered_point)
    return PointMapper(start_point=ordered_point, point_map_info=raveled.point_map_info)
