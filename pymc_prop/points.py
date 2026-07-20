"""Utilities for mapping between PyMC points and flat particle arrays."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pytensor.tensor as pt

from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.logprob.transforms import IntervalTransform, LogOddsTransform, LogTransform
from pymc.model import modelcontext


PointType = Dict[str, np.ndarray]

_MIRROR_COMPATIBLE = (LogTransform, LogOddsTransform, IntervalTransform)


@dataclass(frozen=True)
class TransformSlice:
    """One flat slab of unconstrained ``value_vars`` and its free-RV / transform metadata."""

    value_name: str
    free_name: str
    offset: int
    size: int
    shape: tuple[int, ...]
    transform: Any | None
    free_rv: Any | None = None
    backward_fn: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass
class PointMapper:
    """Bidirectional map between PyMC ``value_vars`` points and flat particles.

    Particles live in unconstrained ``value_vars``. With elementwise transforms,
    mirror-mapped Wasserstein gradient flow keeps this storage and scales
    diffusion by
    :math:`\\sigma=\\exp(-\\tfrac12\\texttt{log\\_jac\\_det})`
    evaluated on unconstrained coordinates (Gu & Kim 2025, §2.1).
    Identity coordinates use ``σ ≡ 1``.
    """

    start_point: PointType
    point_map_info: tuple
    slices: tuple[TransformSlice, ...] = ()
    has_transforms: bool = False
    _noise_scale_fn: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )
    _primal_scale_fn: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )

    def ravel(self, point: PointType) -> np.ndarray:
        return DictToArrayBijection.map(point).data

    def unravel(self, array: np.ndarray) -> PointType:
        raveled = RaveledVars(np.asarray(array, dtype=float), self.point_map_info)
        return DictToArrayBijection.rmap(raveled, start_point=self.start_point)

    def _apply_scale_fn(
        self,
        particles: np.ndarray,
        scale_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        particles = np.asarray(particles, dtype=float)
        if not self.has_transforms or scale_fn is None:
            return np.ones_like(particles, dtype=float)
        if particles.ndim == 1:
            return np.asarray(scale_fn(particles[None, :])[0], dtype=float)
        return np.asarray(scale_fn(particles), dtype=float)

    def noise_scale(self, particles: np.ndarray) -> np.ndarray:
        """Mirror noise factor ``σ`` on unconstrained ``value_vars``, matching ``particles``.

        Identity (no transform) coordinates are ``1`` so the Euler–Maruyama
        update matches the unconstrained isotropic step.
        """
        return self._apply_scale_fn(particles, self._noise_scale_fn)

    def primal_scale(self, particles: np.ndarray) -> np.ndarray:
        """Unconstrained→constrained chain-rule factor ``exp(-log_jac_det)``.

        Multiply unconstrained ``value_vars`` gradients by this to recover
        :math:`\\nabla_\\theta` for mirror-mapped Euler–Maruyama. Identity
        slabs are ``1``.
        """
        return self._apply_scale_fn(particles, self._primal_scale_fn)

    def backward_slab(self, slab: np.ndarray, sl: TransformSlice) -> np.ndarray:
        """Map an unconstrained flat slab ``(..., size)`` to constrained free-RV values."""
        if sl.transform is None or sl.backward_fn is None:
            return np.asarray(slab, dtype=float)
        slab = np.asarray(slab, dtype=float)
        leading = slab.shape[:-1]
        flat = slab.reshape(-1, sl.size)
        out = np.empty_like(flat, dtype=float)
        for i in range(flat.shape[0]):
            out[i] = np.asarray(sl.backward_fn(flat[i]), dtype=float).reshape(sl.size)
        return out.reshape(*leading, sl.size)


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


def require_mirror_compatible_transforms(model=None) -> None:
    """Require continuous free RVs with elementwise (or no) transforms.

    MVP allowlist: ``LogTransform``, ``LogOddsTransform``, ``Interval`` /
    ``IntervalTransform``. Coupled maps (e.g. ``SimplexTransform``) raise
    ``ValueError`` — the scalar ``σ = exp(-½ log_jac_det)`` formula does not
    apply.
    """
    model = modelcontext(model)
    if model.discrete_value_vars:
        raise ValueError("Log-score sampling requires continuous value variables.")
    unsupported: list[str] = []
    for rv in model.free_RVs:
        transform = model.rvs_to_transforms.get(rv)
        if transform is None:
            continue
        if not isinstance(transform, _MIRROR_COMPATIBLE):
            unsupported.append(f"{rv.name!r} ({type(transform).__name__})")
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            "Only elementwise transforms are supported "
            "(LogTransform, LogOddsTransform, Interval); "
            f"found unsupported: [{names}]. "
            "Simplex / Dirichlet / ordered maps are not supported yet."
        )


def _compile_backward_fn(model, rv, transform, value_var):
    """Compile NumPy ``backward`` for one unconstrained value var."""
    if isinstance(transform, IntervalTransform):
        back = transform.backward(value_var, *rv.owner.inputs)
    else:
        back = transform.backward(value_var)

    back_fn = model.compile_fn(
        inputs=[value_var], outs=back, on_unused_input="ignore"
    )
    name = value_var.name

    def backward_1d(y_flat: np.ndarray) -> np.ndarray:
        shaped = np.asarray(y_flat, dtype=float).reshape(value_var.type.shape)
        out = back_fn({name: shaped})
        return np.asarray(out, dtype=float).reshape(-1)

    return backward_1d


def _build_slices(model, point_map_info) -> tuple[TransformSlice, ...]:
    """Attach free-RV / transform metadata to each ``point_map_info`` slab."""
    value_to_rv = {model.rvs_to_values[rv].name: rv for rv in model.free_RVs}
    slices: list[TransformSlice] = []
    offset = 0
    for name, shape, size, _dtype in point_map_info:
        rv = value_to_rv.get(name)
        if rv is None:
            raise ValueError(
                f"value var {name!r} in point_map_info has no matching free RV."
            )
        transform = model.rvs_to_transforms.get(rv)
        backward_fn = None
        if transform is not None:
            value_var = model.rvs_to_values[rv]
            backward_fn = _compile_backward_fn(model, rv, transform, value_var)
        slices.append(
            TransformSlice(
                value_name=name,
                free_name=rv.name,
                offset=offset,
                size=size,
                shape=tuple(int(s) for s in shape),
                transform=transform,
                free_rv=rv,
                backward_fn=backward_fn,
            )
        )
        offset += size
    return tuple(slices)


def _slab_log_jac_det(y_shaped: pt.TensorVariable, sl: TransformSlice) -> pt.TensorVariable:
    """Elementwise ``log_jac_det`` for one unconstrained slab (Interval needs RV inputs)."""
    if isinstance(sl.transform, IntervalTransform):
        return sl.transform.log_jac_det(y_shaped, *sl.free_rv.owner.inputs)
    return sl.transform.log_jac_det(y_shaped)


def _compile_mirror_scale_fn(
    model,
    slices: tuple[TransformSlice, ...],
    *,
    ljd_coef: float,
):
    """Compile per-coordinate scale ``exp(ljd_coef * log_jac_det)``, shape ``(p, d)``.

    ``ljd_coef=-0.5`` → mirror noise ``σ``; ``ljd_coef=-1`` → constrained-space
    chain-rule factor (``primal_scale``).
    """
    if not any(sl.transform is not None for sl in slices):
        return None

    particles = pt.matrix("particles")
    cols: list[pt.TensorVariable] = []
    for sl in slices:
        y = particles[:, sl.offset : sl.offset + sl.size]
        if sl.transform is None:
            cols.append(pt.ones_like(y))
            continue
        batch = pt.shape(particles)[0]
        y_shaped = pt.reshape(y, (batch, *sl.shape))
        ljd = _slab_log_jac_det(y_shaped, sl)
        cols.append(pt.reshape(pt.exp(ljd_coef * ljd), (batch, sl.size)))
    scale = pt.concatenate(cols, axis=1)
    return model.compile_fn(
        inputs=[particles],
        outs=scale,
        point_fn=False,
        on_unused_input="ignore",
    )


def primal_scale_graph(
    particles: pt.TensorVariable,
    mapper: PointMapper,
) -> pt.TensorVariable:
    """Unconstrained→constrained chain-rule factor ``exp(-log_jac_det)``, shape ``(p, d)``.

    Multiplies unconstrained scores / prior grads to recover
    :math:`\\nabla_\\theta` (Gu & Kim 2025). Identity slabs
    contribute ``1``.
    """
    if not mapper.has_transforms:
        return pt.ones_like(particles)

    cols: list[pt.TensorVariable] = []
    for sl in mapper.slices:
        y = particles[:, sl.offset : sl.offset + sl.size]
        if sl.transform is None:
            cols.append(pt.ones_like(y))
            continue
        batch = pt.shape(particles)[0]
        y_shaped = pt.reshape(y, (batch, *sl.shape))
        ljd = _slab_log_jac_det(y_shaped, sl)
        cols.append(pt.reshape(pt.exp(-ljd), (batch, sl.size)))
    return pt.concatenate(cols, axis=1)


def make_point_mapper(model=None) -> PointMapper:
    """Build a :class:`PointMapper` from ``model.initial_point()`` and ``value_vars``."""
    model = modelcontext(model)
    require_mirror_compatible_transforms(model)
    start_point = model.initial_point()
    ordered_point = {var.name: np.asarray(start_point[var.name]) for var in model.value_vars}
    raveled = DictToArrayBijection.map(ordered_point)
    slices = _build_slices(model, raveled.point_map_info)
    has_transforms = any(sl.transform is not None for sl in slices)
    noise_fn = None
    primal_fn = None
    if has_transforms:
        noise_fn = _compile_mirror_scale_fn(model, slices, ljd_coef=-0.5)
        primal_fn = _compile_mirror_scale_fn(model, slices, ljd_coef=-1.0)
    return PointMapper(
        start_point=ordered_point,
        point_map_info=raveled.point_map_info,
        slices=slices,
        has_transforms=has_transforms,
        _noise_scale_fn=noise_fn,
        _primal_scale_fn=primal_fn,
    )
