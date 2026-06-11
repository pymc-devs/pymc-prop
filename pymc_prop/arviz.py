"""Convert PrO particle traces to ArviZ DataTree output."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc
import pytensor.tensor as pt
from arviz_base import from_dict
from arviz_base.base import make_attrs
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.model import modelcontext
from pytensor.graph.replace import graph_replace
from pytensor.scan import scan
from xarray import DataTree

from pymc_prop.compile import compile_batched_observed_logp_score
from pymc_prop.points import PointMapper, flat_to_value_vars

# draw = retained index; chain = ensemble particles; step = simulation step
_SAMPLE_DIMS = ["draw", "chain"]
_PROTECTED_DATATREE_KEYS = frozenset({"sample_dims"})


def _retained_step_indices(burn_in: int, thinning: int, n_retained: int) -> np.ndarray:
    """Simulation step index for each retained slice (maps to the ``step`` coord)."""
    return np.asarray([burn_in + i * thinning for i in range(n_retained)], dtype=int)


def _particles_to_posterior(
    particles: np.ndarray,
    model,
    mapper: PointMapper,
) -> dict[str, np.ndarray]:
    """Split flat particles into named free-RV arrays with shape (draw, chain, *event)."""
    if particles.ndim != 3:
        raise ValueError(
            f"particles must be 3-D (n_retained, n_particles, flat); got shape {particles.shape}."
        )

    free_names = {rv.name for rv in model.free_RVs}
    n_retained, n_particles, n_flat = particles.shape
    posterior: dict[str, np.ndarray] = {}
    offset = 0

    for name, shape, size, _dtype in mapper.point_map_info:
        if name not in free_names:
            offset += size
            continue
        if offset + size > n_flat:
            raise ValueError(
                f"point_map_info for {name!r} exceeds flat particle width {n_flat}."
            )
        slab = particles[..., offset : offset + size]
        posterior[name] = np.asarray(slab.reshape((n_retained, n_particles, *shape)), dtype=float)
        offset += size

    if offset != n_flat:
        raise ValueError(
            f"point_map_info covers {offset} flat dims but particles have width {n_flat}."
        )
    return posterior


def _merged_coords(model, coords: dict[str, Any] | None) -> dict[str, Any]:
    """Merge PyMC model coordinates with user-supplied coords."""
    model_coords, _ = coords_and_dims_for_inferencedata(model)
    merged = dict(model_coords)
    if coords:
        merged.update(coords)
    return merged


def _merged_dims(model, dims: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Merge PyMC model variable dims with user-supplied dims."""
    _, model_dims = coords_and_dims_for_inferencedata(model)
    merged = {name: list(dvals) for name, dvals in model_dims.items()}
    if dims:
        merged.update(dims)
    return merged


def _extract_observed_data(model) -> dict[str, np.ndarray]:
    """Read constant observed values from the PyMC model."""
    observed: dict[str, np.ndarray] = {}
    for rv in model.observed_RVs:
        value = model.rvs_to_values[rv]
        observed[rv.name] = np.asarray(value.eval(), dtype=float)
    return observed


def _core_observed_logp_single_rv(
    particle_flat: pt.TensorVariable,
    model,
    mapper: PointMapper,
    rv,
) -> pt.TensorVariable:
    """Elementwise observed logp for one RV at one flat particle."""
    value_vars = model.value_vars
    mapped_value_vars = flat_to_value_vars(particle_flat, mapper.point_map_info)
    replace = dict(zip(value_vars, mapped_value_vars, strict=True))
    logp_terms = model.logp(vars=[rv], sum=False)
    logp_vec = pt.flatten(pt.add(*logp_terms))
    return graph_replace(logp_vec, replace=replace, strict=False)


def _batched_logp_single_rv_graph(
    particles: pt.TensorVariable,
    model,
    mapper: PointMapper,
    rv,
    *,
    use_scan: bool,
) -> pt.TensorVariable:
    """Batch elementwise logp for one observed RV over particle rows."""
    if use_scan:
        logp, _ = scan(
            fn=lambda particle: _core_observed_logp_single_rv(particle, model, mapper, rv),
            sequences=[particles],
        )
        return logp

    vec_fn = pt.vectorize(
        lambda particle: _core_observed_logp_single_rv(particle, model, mapper, rv),
        signature="(d)->(n)",
    )
    return vec_fn(particles)


def _compile_batched_logp_for_rv(model, mapper: PointMapper, rv) -> Any:
    """Compile batched elementwise logp for one observed RV; output (n_particles, n_obs_i)."""
    particles = pt.matrix("particles")
    try:
        logp = _batched_logp_single_rv_graph(particles, model, mapper, rv, use_scan=False)
    except Exception:
        logp = _batched_logp_single_rv_graph(particles, model, mapper, rv, use_scan=True)
    return model.compile_fn(
        inputs=[particles],
        outs=logp,
        point_fn=False,
        on_unused_input="ignore",
    )


def _eval_batched_logp_across_steps(
    batched_fn: Any,
    particles: np.ndarray,
) -> np.ndarray:
    """Evaluate batched logp over all retained steps in one compiled call."""
    n_retained, n_particles, n_flat = particles.shape
    flat = particles.reshape(n_retained * n_particles, n_flat)
    out = batched_fn(flat)
    if isinstance(out, (tuple, list)):
        logp = out[0]
    else:
        logp = out
    logp = np.asarray(logp, dtype=float)
    return logp.reshape(n_retained, n_particles, *logp.shape[1:])


def _compute_log_likelihood(
    particles: np.ndarray,
    model,
    mapper: PointMapper,
) -> dict[str, np.ndarray]:
    """Evaluate per-observation log-likelihood at retained particle clouds."""
    if particles.shape[0] == 0:
        return {}

    observed_rvs = list(model.observed_RVs)
    log_likelihood: dict[str, np.ndarray] = {}

    if len(observed_rvs) == 1:
        rv = observed_rvs[0]
        batched_fn = compile_batched_observed_logp_score(model, mapper)
        log_likelihood[rv.name] = _eval_batched_logp_across_steps(batched_fn, particles)
        return log_likelihood

    for rv in observed_rvs:
        batched_fn = _compile_batched_logp_for_rv(model, mapper, rv)
        log_likelihood[rv.name] = _eval_batched_logp_across_steps(batched_fn, particles)
    return log_likelihood


def _broadcast_draw_stat(values: np.ndarray, n_particles: int) -> np.ndarray:
    """Broadcast step-only diagnostics to the standard (draw, chain) layout."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.broadcast_to(values[:, None], (values.shape[0], n_particles))
    return np.broadcast_to(values[:, None, ...], (values.shape[0], n_particles, *values.shape[1:]))


def _compute_sample_stats(
    particles: np.ndarray,
    posterior: dict[str, np.ndarray],
    log_likelihood: dict[str, np.ndarray],
    learning_rate: float,
) -> dict[str, np.ndarray]:
    """PrO-specific diagnostics derived from retained particle clouds."""
    n_retained, n_particles = particles.shape[:2]
    if n_retained == 0:
        return {}

    stats: dict[str, np.ndarray] = {
        "particle_spread": _broadcast_draw_stat(
            np.mean(np.std(particles, axis=1), axis=-1), n_particles
        ),
        "learning_rate": _broadcast_draw_stat(
            np.full(n_retained, learning_rate, dtype=float), n_particles
        ),
    }

    for name, values in posterior.items():
        stats[f"{name}_spread"] = _broadcast_draw_stat(np.std(values, axis=1), n_particles)

    if log_likelihood:
        total = sum(np.asarray(arr, dtype=float) for arr in log_likelihood.values())
        per_particle = np.sum(total, axis=-1)
        mean_score = np.mean(per_particle, axis=1)
        se_score = np.std(per_particle, axis=1, ddof=1) / np.sqrt(n_particles)
        stats["mean_log_score"] = _broadcast_draw_stat(mean_score, n_particles)
        stats["se_log_score"] = _broadcast_draw_stat(se_score, n_particles)

    return stats


def _merge_datatree_build_kwargs(
    kwargs: dict[str, Any],
    datatree_kwargs: dict[str, Any] | None,
) -> None:
    """Merge user ``datatree_kwargs`` into ``from_dict`` kwargs (pre-build)."""
    if not datatree_kwargs:
        return

    for key, value in datatree_kwargs.items():
        if key in _PROTECTED_DATATREE_KEYS:
            continue
        if key == "attrs" and isinstance(value, dict):
            root_attrs = kwargs["attrs"].setdefault("/", {})
            root_attrs.update(value.get("/", {}))
            for group, group_attrs in value.items():
                if group != "/":
                    kwargs["attrs"].setdefault(group, {}).update(group_attrs)
        elif key == "coords" and isinstance(value, dict):
            kwargs["coords"] = {**kwargs["coords"], **value}
        elif key == "dims" and isinstance(value, dict):
            kwargs["dims"] = {**kwargs["dims"], **value}
        else:
            kwargs[key] = value


def _attach_pro_coords(
    dt: DataTree,
    *,
    draw_coord: np.ndarray,
    step_coord: np.ndarray | None,
) -> None:
    """Attach PrO ``draw`` and ``step`` coords to groups with a draw dimension."""
    for name in list(dt.children):
        node = dt[name]
        if "draw" not in node.dims:
            continue
        ds = node.dataset.assign_coords(draw=draw_coord)
        if step_coord is not None:
            ds = ds.assign_coords(step=("draw", step_coord))
        dt[name] = ds


def _pro_to_datatree(
    particles: np.ndarray,
    *,
    model=None,
    mapper: PointMapper,
    burn_in: int,
    thinning: int,
    learning_rate: float,
    coords: dict[str, Any] | None = None,
    dims: dict[str, list[str]] | None = None,
    include_log_likelihood: bool = True,
    include_observed_data: bool = True,
    include_sample_stats: bool = True,
    datatree_kwargs: dict[str, Any] | None = None,
) -> DataTree:
    """Package retained PrO particles as an ArviZ DataTree.

    The sampler ndarray has shape ``(n_retained, n_particles, flat)``.
    ArviZ ``sample_dims``: ``draw`` is a retained index;
    ``step`` stores the simulation step number; ``chain`` indexes particles.
    """
    model = modelcontext(model)
    n_retained, n_particles = particles.shape[:2]
    draw_coord = np.arange(n_retained, dtype=int)
    step_coord = (_retained_step_indices(burn_in, thinning, n_retained) if n_retained > 0 else None)

    posterior = _particles_to_posterior(particles, model, mapper)
    merged_coords = _merged_coords(model, coords)
    merged_coords["draw"] = draw_coord
    if n_particles > 0:
        merged_coords["chain"] = np.arange(n_particles, dtype=int)
    merged_dims = _merged_dims(model, dims)

    data: dict[str, dict[str, np.ndarray]] = {"posterior": posterior}

    if include_observed_data and model.observed_RVs:
        data["observed_data"] = _extract_observed_data(model)

    log_likelihood: dict[str, np.ndarray] = {}
    if include_log_likelihood and model.observed_RVs and n_retained > 0:
        log_likelihood = _compute_log_likelihood(particles, model, mapper)
        if log_likelihood:
            data["log_likelihood"] = log_likelihood

    if include_sample_stats and n_retained > 0:
        data["sample_stats"] = _compute_sample_stats(
            particles, posterior, log_likelihood, learning_rate
        )

    pymc_attrs = make_attrs(inference_library=pymc, sample_dims=_SAMPLE_DIMS)
    group_attrs: dict[str, dict[str, Any]] = {
        "posterior": dict(pymc_attrs),
    }
    if include_log_likelihood and model.observed_RVs and n_retained > 0:
        group_attrs["log_likelihood"] = dict(pymc_attrs)
    if include_sample_stats and n_retained > 0:
        group_attrs["sample_stats"] = dict(pymc_attrs)

    kwargs: dict[str, Any] = {
        "sample_dims": _SAMPLE_DIMS,
        "coords": merged_coords,
        "dims": merged_dims,
        "attrs": group_attrs,
    }
    _merge_datatree_build_kwargs(kwargs, datatree_kwargs)
    kwargs["coords"]["draw"] = draw_coord

    dt = from_dict(data, **kwargs)
    _attach_pro_coords(dt, draw_coord=draw_coord, step_coord=step_coord)
    return dt
