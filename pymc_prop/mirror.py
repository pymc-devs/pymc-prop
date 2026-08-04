"""Elementwise mirror-transform helpers for Wasserstein gradient flow.

Mirror-mapped Langevin dynamics (Gu & Kim 2025, §2.1) keep particles in
unconstrained ``value_vars`` and scale drift / noise by functions of
``log_jac_det``. This module owns the allowlist, the PyMC transform call
convention, and compiled NumPy callables for ``backward`` and
``log_jac_det``.

How to add a transform
----------------------
1. Confirm the Jacobian is diagonal / elementwise (coupled maps such as
   simplex / Dirichlet are out of scope for the current allowlist).
2. Add the transform class to :data:`_ELEMENTWISE_TRANSFORMS`.
3. Call methods only through :func:`call_transform`, which always passes
   ``*rv.owner.inputs`` — do not special-case individual transform types.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytensor.tensor as pt

from pymc.logprob.transforms import IntervalTransform, LogOddsTransform, LogTransform
from pymc.model import modelcontext


_ELEMENTWISE_TRANSFORMS = (LogTransform, LogOddsTransform, IntervalTransform)


def require_mirror_compatible_transforms(model=None) -> None:
    """Require continuous free RVs with elementwise (or no) transforms.

    Supported: ``LogTransform``, ``LogOddsTransform``, ``Interval`` /
    ``IntervalTransform``. Coupled maps (e.g. ``SimplexTransform``) raise
    ``ValueError``.
    """
    model = modelcontext(model)
    if model.discrete_value_vars:
        raise ValueError("Log-score sampling requires continuous value variables.")
    unsupported: list[str] = []
    for rv in model.free_RVs:
        transform = model.rvs_to_transforms.get(rv)
        if transform is None:
            continue
        if not isinstance(transform, _ELEMENTWISE_TRANSFORMS):
            unsupported.append(f"{rv.name!r} ({type(transform).__name__})")
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            "Only elementwise transforms are supported "
            "(LogTransform, LogOddsTransform, Interval); "
            f"found unsupported: [{names}]. "
            "Simplex / Dirichlet / ordered maps are not supported yet."
        )


def call_transform(transform: Any, method: str, value: Any, rv: Any) -> Any:
    """Invoke ``transform.<method>(value, *rv.owner.inputs)``.

    All PyMC transforms accept ``*inputs``; always pass them so Interval and
    other input-dependent maps share one call path.
    """
    return getattr(transform, method)(value, *rv.owner.inputs)


def compile_backward_fn(
    model, rv, transform, value_var
) -> Callable[[np.ndarray], np.ndarray]:
    """Compile NumPy ``backward`` for one unconstrained value var."""
    back = call_transform(transform, "backward", value_var, rv)
    back_fn = model.compile_fn(
        inputs=[value_var], outs=back, on_unused_input="ignore"
    )
    name = value_var.name

    def backward_1d(y_flat: np.ndarray) -> np.ndarray:
        shaped = np.asarray(y_flat, dtype=float).reshape(value_var.type.shape)
        out = back_fn({name: shaped})
        return np.asarray(out, dtype=float).reshape(-1)

    return backward_1d


def slab_log_jac_det(y_shaped: pt.TensorVariable, sl: Any) -> pt.TensorVariable:
    """Elementwise ``log_jac_det`` for one unconstrained slab.

    ``sl`` must expose ``.transform`` and ``.free_rv`` (e.g. a
    :class:`~pymc_prop.points.TransformSlice`).
    """
    return call_transform(sl.transform, "log_jac_det", y_shaped, sl.free_rv)


def compile_log_jac_det_fn(
    model, slices: Sequence[Any]
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Compile per-flat-dimension ``log_jac_det(y)``; return a NumPy callable.

    Returns ``None`` when every slab is identity. The callable has signature
    ``ljd_fn(particles) -> ndarray`` with shape ``(p, d)``.
    """
    if not any(sl.transform is not None for sl in slices):
        return None

    particles = pt.matrix("particles")
    cols: list[pt.TensorVariable] = []
    for sl in slices:
        y = particles[:, sl.offset : sl.offset + sl.size]
        if sl.transform is None:
            cols.append(pt.zeros_like(y))
            continue
        batch = pt.shape(particles)[0]
        y_shaped = pt.reshape(y, (batch, *sl.shape))
        ljd = slab_log_jac_det(y_shaped, sl)
        cols.append(pt.reshape(ljd, (batch, sl.size)))
    ljd_graph = pt.concatenate(cols, axis=1)
    return model.compile_fn(
        inputs=[particles], outs=ljd_graph, point_fn=False, on_unused_input="ignore"
    )
