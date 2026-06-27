"""Convert PrO particle traces to ArviZ DataTree output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pymc
import pytensor.tensor as pt
import xarray as xr
from scipy.special import logsumexp
from arviz_base import from_dict
from arviz_base.base import dict_to_dataset, make_attrs
from pymc.backends.arviz import _DefaultTrace, dataset_to_point_list
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.model import modelcontext
from pymc.sampling.forward import (
    _build_constant_data,
    compile_forward_sampling_function,
    get_vars_in_point_list,
    observed_dependent_deterministics,
)
from pymc.util import _get_seeds_per_chain, get_default_varnames, point_wrapper
from pytensor.graph.replace import graph_replace
from pytensor.scan import scan
from xarray import DataTree

from pymc_prop.compile import compile_batched_observed_logp_score
from pymc_prop.points import PointMapper, flat_to_value_vars

_PRO_SAMPLE_DIMS = ["draw", "chain"]
_SAMPLE_DIMS = _PRO_SAMPLE_DIMS
_MIXTURE_SAMPLE_DIMS = ["draw"]
_PROTECTED_DATATREE_KEYS = frozenset({"sample_dims"})
_VECTORIZE_FALLBACK_ERRORS = (TypeError, ValueError, AttributeError, NotImplementedError)


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
    except _VECTORIZE_FALLBACK_ERRORS:
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


def _compute_mixture_log_predictive(
    log_likelihood: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Empirical mixture log predictive log p_{hat Q}(y) over uniform particle weights."""
    mixture: dict[str, np.ndarray] = {}
    for name, ll in log_likelihood.items():
        ll = np.asarray(ll, dtype=float)
        n_particles = ll.shape[1]
        mixture[name] = logsumexp(ll, axis=1) - np.log(n_particles)
    return mixture


def _mixture_log_predictive_to_dataset(
    mixture_log_predictive: dict[str, np.ndarray],
    *,
    coords: dict[str, Any],
    dims: dict[str, list[str]],
    attrs: dict[str, Any],
) -> xr.Dataset:
    """Build mixture log predictive dataset without a particle ``chain`` dimension."""
    mixture_coords = {k: v for k, v in coords.items() if k != "chain"}
    return dict_to_dataset(
        mixture_log_predictive,
        attrs=attrs,
        coords=mixture_coords,
        dims=dims,
        sample_dims=_MIXTURE_SAMPLE_DIMS,
        skip_event_dims=True,
        inference_library=pymc,
    )


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
    *,
    mixture_log_predictive: dict[str, np.ndarray] | None = None,
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

    if mixture_log_predictive:
        total = sum(
            np.sum(np.asarray(arr, dtype=float), axis=tuple(range(1, arr.ndim)))
            for arr in mixture_log_predictive.values()
        )
        stats["mixture_log_predictive_total"] = _broadcast_draw_stat(total, n_particles)

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
    groups: Sequence[str] | None = None,
) -> None:
    """Attach PrO ``draw`` and ``step`` coords to groups with a draw dimension."""
    if groups is None:
        groups = list(dt.children)
    for name in groups:
        if name not in dt.children:
            continue
        node = dt[name]
        if "draw" not in node.dims:
            continue
        ds = node.dataset.assign_coords(draw=draw_coord)
        if step_coord is not None:
            ds = ds.assign_coords(step=("draw", step_coord))
        dt[name] = ds


def _trace_constant_data_from_datatree(dt: DataTree | None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Recover trace-time constant data and coords from a PrO DataTree."""
    trace_constant_data: dict[str, np.ndarray] = {}
    trace_coords: dict[str, np.ndarray] = {}
    if dt is None:
        return trace_constant_data, trace_coords

    constant_data = dt.get("constant_data", None)
    if constant_data is not None:
        trace_coords.update({str(k): np.asarray(v) for k, v in constant_data.coords.items()})
        trace_constant_data.update({str(k): np.asarray(v) for k, v in constant_data.items()})

    if "posterior" in dt.children:
        posterior = dt["posterior"].dataset
        for dim in ("draw", "chain"):
            if dim in posterior.coords:
                trace_coords.setdefault(dim, np.asarray(posterior.coords[dim].values))

    observed_data = dt.get("observed_data", None)
    if observed_data is not None:
        for name in observed_data.data_vars:
            trace_constant_data.setdefault(name, np.asarray(observed_data[name].values))
        trace_coords.update({str(k): np.asarray(v) for k, v in observed_data.coords.items()})

    return trace_constant_data, trace_coords


def _resolve_forward_output_vars(
    model,
    *,
    var_names: Sequence[str] | None,
    observed_data: xr.Dataset | None,
):
    """Choose model outputs for a forward pass (PyMC PPC default)."""
    if var_names is not None:
        return [model[x] for x in var_names]

    extra_observeds: list = []
    output_vars = list(model.observed_RVs)
    if observed_data is not None:
        for name in observed_data.data_vars:
            if name in model and model[name] not in output_vars:
                output_vars.append(model[name])
                extra_observeds.append(model[name])
    output_vars_set = set(output_vars)
    output_vars += [
        d
        for d in observed_dependent_deterministics(model, extra_observeds)
        if d not in output_vars_set
    ]
    return output_vars


def _forward_from_posterior(
    posterior: xr.Dataset,
    model,
    *,
    predictions: bool,
    dt: DataTree | None = None,
    var_names: Sequence[str] | None = None,
    random_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Forward-sample observed outputs for each ``(draw, chain)`` posterior point."""
    model = modelcontext(model)
    if posterior.sizes.get("draw", 0) == 0 or posterior.sizes.get("chain", 0) == 0:
        return {}

    point_list, stacked_dims = dataset_to_point_list(posterior, _SAMPLE_DIMS)
    if not point_list:
        return {}

    trace_constant_data, trace_coords = _trace_constant_data_from_datatree(dt)
    constant_data = _build_constant_data(trace_constant_data, trace_coords, model)
    vars_in_trace = get_vars_in_point_list(point_list, model)

    observed_data = None if predictions else dt.get("observed_data", None) if dt is not None else None
    if observed_data is not None:
        observed_data = observed_data.dataset

    output_vars = _resolve_forward_output_vars(
        model, var_names=var_names, observed_data=observed_data
    )
    if not output_vars:
        return {}

    trace_rvs = [rv for rv in vars_in_trace if rv in set(model.basic_RVs)]
    frozen_vars = set(trace_rvs)
    vars_to_evaluate = list(get_default_varnames(output_vars, include_transformed=False))

    if random_seed is not None:
        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    ppc_trace_t = _DefaultTrace(len(point_list))
    if vars_to_evaluate:
        sampler_fn, _ = compile_forward_sampling_function(
            outputs=vars_to_evaluate,
            vars_in_trace=vars_in_trace,
            basic_rvs=model.basic_RVs,
            random_seed=random_seed,
            constant_data=constant_data,
            volatile_vars=set(),
            freeze_vars=frozen_vars,
        )
        sampler_fn = point_wrapper(sampler_fn)
        evaluated_names = {v.name for v in vars_to_evaluate}
        copy_from_trace_names = [var.name for var in output_vars if var.name not in evaluated_names]

        for idx, param in enumerate(point_list):
            if sampler_fn is not None:
                values = sampler_fn(**param)
                for k, v in zip(vars_to_evaluate, values, strict=True):
                    ppc_trace_t.insert(k.name, v, idx)
            for name in copy_from_trace_names:
                ppc_trace_t.insert(name, param[name], idx)
    else:
        for idx, param in enumerate(point_list):
            for var in output_vars:
                ppc_trace_t.insert(var.name, param[var.name], idx)

    ppc_trace = ppc_trace_t.trace_dict
    for name, ary in ppc_trace.items():
        ppc_trace[name] = ary.reshape(
            (*[len(coord) for coord in stacked_dims.values()], *ary.shape[1:])
        )
    return ppc_trace


def _forward_dict_to_grid_dataset(
    forward: dict[str, np.ndarray],
    model,
    posterior: xr.Dataset,
    *,
    coords: dict[str, Any] | None = None,
    dims: dict[str, list[str]] | None = None,
) -> xr.Dataset:
    """Wrap a per-particle forward grid as an xarray Dataset."""
    model = modelcontext(model)
    merged_coords = _merged_coords(model, coords)
    merged_dims = _merged_dims(model, dims)
    merged_coords["draw"] = np.asarray(posterior.coords["draw"].values, dtype=int)
    merged_coords["chain"] = np.asarray(posterior.coords["chain"].values, dtype=int)
    return dict_to_dataset(
        forward,
        coords=merged_coords,
        dims=merged_dims,
        sample_dims=_PRO_SAMPLE_DIMS,
        skip_event_dims=True,
        inference_library=pymc,
    )


def _mixture_remix_forward_dataset(
    grid: xr.Dataset,
    *,
    random_seed: int | None = None,
) -> xr.Dataset:
    """Remix a per-particle forward grid into draw-aligned mixture PPC draws.

    For each retained draw index *d*, independently resample particle ``chain``
    per observation element from ``grid.isel(draw=d)`` -- a marginal mixture at
    each :math:`Q_t`. Output shape is ``(draw, *obs)`` with ``sample_dims=["draw"]`` 
    and no ``chain`` dimension.
    """
    if grid.sizes.get("draw", 0) == 0 or grid.sizes.get("chain", 0) == 0:
        return grid

    n_draw = grid.sizes["draw"]
    n_chain = grid.sizes["chain"]
    obs_dims = [d for d in grid.dims if d not in _PRO_SAMPLE_DIMS]
    target_shape = (n_draw,) + tuple(grid.sizes[d] for d in obs_dims)

    rng = np.random.default_rng(random_seed)
    random_chains = rng.integers(0, n_chain, size=target_shape)
    indexing_dims = ["draw", *obs_dims]
    chain_da = xr.DataArray(random_chains, dims=indexing_dims)
    draw_da = xr.DataArray(
        np.broadcast_to(np.arange(n_draw, dtype=int).reshape((-1,) + (1,) * len(obs_dims)), target_shape),
        dims=indexing_dims,
    )

    final_dims = ("draw", *obs_dims)
    remixed_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name in grid.data_vars:
        y_mixed = grid[name].isel(chain=chain_da, draw=draw_da)
        remixed_vars[name] = (final_dims, y_mixed.values)

    remixed_coords: dict[str, Any] = {
        "draw": np.asarray(grid.coords["draw"].values, dtype=int),
    }
    for dim in obs_dims:
        remixed_coords[dim] = (
            grid.coords[dim].values
            if dim in grid.coords
            else np.arange(grid.sizes[dim], dtype=int)
        )

    attrs = make_attrs(inference_library=pymc, sample_dims=_MIXTURE_SAMPLE_DIMS)
    return xr.Dataset(remixed_vars, coords=remixed_coords, attrs=attrs)


def _merge_remixed_forward_into_datatree(
    dt: DataTree,
    remixed: xr.Dataset,
    *,
    predictions: bool,
    posterior: xr.Dataset | None = None,
) -> DataTree:
    """Attach mixture-remixed forward draws to a PrO DataTree."""
    if not remixed.data_vars:
        return dt

    if posterior is None and "posterior" in dt.children:
        posterior = dt["posterior"].dataset

    if posterior is not None and "draw" in posterior.coords:
        remixed = remixed.assign_coords(draw=posterior.coords["draw"])
        if "step" in posterior.coords:
            remixed = remixed.assign_coords(step=("draw", posterior.coords["step"].values))

    group_name = "predictions" if predictions else "posterior_predictive"
    dt[group_name] = DataTree(name=group_name, dataset=remixed)
    return dt


def _spawn_forward_seed(random_seed: int | None) -> int | None:
    """Derive an independent seed for forward sampling."""
    if random_seed is None:
        return None
    child = np.random.SeedSequence(random_seed).spawn(1)[0]
    return int(child.generate_state(1, dtype=np.uint64)[0])


def _spawn_forward_and_remix_seeds(
    random_seed: int | None,
) -> tuple[int | None, int | None]:
    """Derive independent seeds for the forward grid and mixture remix."""
    if random_seed is None:
        return None, None
    children = np.random.SeedSequence(random_seed).spawn(2)
    return (
        int(children[0].generate_state(1, dtype=np.uint64)[0]),
        int(children[1].generate_state(1, dtype=np.uint64)[0]),
    )


def _pro_to_datatree(
    particles: np.ndarray,
    *,
    model=None,
    mapper: PointMapper,
    tune: int,
    learning_rate: float,
    coords: dict[str, Any] | None = None,
    dims: dict[str, list[str]] | None = None,
    include_log_likelihood: bool = True,
    include_observed_data: bool = True,
    include_sample_stats: bool = True,
    datatree_kwargs: dict[str, Any] | None = None,
) -> DataTree:
    """Package retained PrO particles as an ArviZ DataTree.

    The sampler ndarray has shape ``(n_steps, n_particles, flat)`` on the
    normal :func:`~pymc_prop.sampler.run_sampler` path. ArviZ ``sample_dims``:
    ``draw`` is a retained index; ``step`` is ``tune + draw``; ``chain``
    indexes particles.
    """
    model = modelcontext(model)
    n_retained, n_particles = particles.shape[:2]
    draw_coord = np.arange(n_retained, dtype=int)
    step_coord = (np.arange(tune, tune + n_retained, dtype=int) if n_retained > 0 else None)

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
    mixture_log_predictive: dict[str, np.ndarray] = {}
    if include_log_likelihood and model.observed_RVs and n_retained > 0:
        log_likelihood = _compute_log_likelihood(particles, model, mapper)
        if log_likelihood:
            data["log_likelihood"] = log_likelihood
            mixture_log_predictive = _compute_mixture_log_predictive(log_likelihood)

    if include_sample_stats and n_retained > 0:
        data["sample_stats"] = _compute_sample_stats(
            particles,
            posterior,
            log_likelihood,
            learning_rate,
            mixture_log_predictive=mixture_log_predictive or None,
        )

    pymc_attrs = make_attrs(inference_library=pymc, sample_dims=_SAMPLE_DIMS)
    mixture_attrs = make_attrs(inference_library=pymc, sample_dims=_MIXTURE_SAMPLE_DIMS)
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
    if mixture_log_predictive:
        mixture_ds = _mixture_log_predictive_to_dataset(
            mixture_log_predictive,
            coords=merged_coords,
            dims=merged_dims,
            attrs=dict(mixture_attrs),
        )
        dt["mixture_log_predictive"] = DataTree(name="mixture_log_predictive", dataset=mixture_ds)
        _attach_pro_coords(
            dt,
            draw_coord=draw_coord,
            step_coord=step_coord,
            groups=["mixture_log_predictive"],
        )
    return dt
