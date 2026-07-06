"""Public sampling entry for PrO posteriors (PyMC-style ``sample`` module)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pymc as pm
from pymc.model import modelcontext
from xarray import DataTree

from pymc_prop.arviz import (
    _PRO_SAMPLE_DIMS,
    _forward_group_datatree,
    _mixture_remix_forward_dataset,
    _pro_to_datatree,
    _spawn_forward_and_remix_seeds,
)
from pymc_prop.points import make_point_mapper
from pymc_prop.sampler import run_sampler
from pymc_prop.scoring import LogScore, ScoringRule


def sample_posterior_predictive_pro(
    dt: DataTree,
    model=None,
    *,
    predictions: bool = False,
    var_names: Sequence[str] | None = None,
    random_seed: int | None = None,
    extend_inferencedata: bool = True,
) -> DataTree:
    """PrO-native posterior predictive / out-of-sample forward sampling.

    Builds a per-particle forward grid via :func:`pymc.sample_posterior_predictive`
    on the full retained ``dt.posterior`` cloud (``sample_dims=["draw", "chain"]``),
    then remixes it into draw-aligned mixture PPC draws. Do not call PyMC forward
    sampling on PrO output with default ``sample_dims`` -- it mis-pairs multi-draw
    traces.

    The exported ``posterior_predictive`` (or ``predictions``) group has shape
    ``(draw, *obs)`` with ``sample_dims=["draw"]``. Each retained index
    independently resamples particle ``chain`` per observation element at that
    snapshot (marginal mixture at :math:`Q_t`, not a joint draw from one
    :math:`\\theta`). ``draw`` and ``step`` align with ``posterior``. The
    analytic log marginal at observed data is in ``mixture_log_predictive``.

    For out-of-sample ``predictions=True``, call :func:`pymc.set_data` on the
    model before this function.

    Parameters
    ----------
    dt
        PrO :class:`~xarray.DataTree` from :func:`sample_pro`.
    predictions
        When ``False`` (default), populate ``posterior_predictive`` for
        in-sample PPC. When ``True``, populate ``predictions`` for OOS.
    extend_inferencedata
        When ``True``, merge the forward group into ``dt`` and return it.

    See Also
    --------
    sample_pro
        Builds the PrO DataTree; ``mixture_log_predictive`` holds the marginal
        log predictive at observed data.
    """
    model = modelcontext(model)
    if "posterior" not in dt.children:
        raise ValueError("DataTree must contain a posterior group from sample_pro.")

    posterior = dt["posterior"].dataset

    with model:
        forward_seed, remix_seed = _spawn_forward_and_remix_seeds(random_seed)
        forward_group = "predictions" if predictions else "posterior_predictive"
        forward_dt = pm.sample_posterior_predictive(
            dt,
            model=model,
            var_names=var_names,
            sample_dims=_PRO_SAMPLE_DIMS,
            extend_inferencedata=False,
            predictions=predictions,
            random_seed=forward_seed,
            progressbar=False,
        )
        grid = forward_dt[forward_group].dataset
        remixed = _mixture_remix_forward_dataset(
            grid,
            random_seed=remix_seed,
        )
        forward_dt = _forward_group_datatree(
            remixed,
            predictions=predictions,
            posterior=posterior,
        )
        if extend_inferencedata:
            for name in forward_dt.children:
                dt[name] = forward_dt[name]
            return dt
        return forward_dt


def sample_pro(
    model=None,
    scoring_rule: Literal["log"] | ScoringRule = "log",
    n_particles: int = 64,
    n_steps: int = 1000,
    tune: int = 200,
    step_size: float | None = None,
    learning_rate: float = 1.0,
    r_eps: float = 1e-5,
    random_seed: int | None = None,
    coords: dict[str, Any] | None = None,
    dims: dict[str, list[str]] | None = None,
    include_log_likelihood: bool = True,
    include_observed_data: bool = True,
    include_sample_stats: bool = True,
    datatree_kwargs: dict[str, Any] | None = None,
) -> DataTree:
    """Run the PrO particle sampler on a PyMC model.

    Free RVs must be native unconstrained; reparameterize manually for now.

    Returns an ArviZ :class:`xarray.DataTree` with ``posterior``,
    ``observed_data``, ``log_likelihood``, ``mixture_log_predictive``, and
    ``sample_stats`` groups. Retained steps map to ``draw``; simulation step
    numbers are in the ``step`` coordinate; particles map to ``chain``.

    The ``mixture_log_predictive`` group holds :math:`\\log p_{\\hat Q}(y_i)` at
    observed data under the empirical particle mixture
    :math:`\\hat Q = \\frac{1}{p}\\sum_j \\delta_{\\theta^{(j)}}` (no ``chain``
    dimension). ``sample_stats.mixture_log_predictive_total`` sums those values
    over observations. This is the log predictive PrO targets; it differs from
    ``mean_log_score``, which averages per-particle log-score sums. Mixture PPC
    draws are added separately via :func:`sample_posterior_predictive_pro`.

    Parameters
    ----------
    n_steps
        Number of retained simulation steps (maps to the ``draw`` dimension).
        The sampler runs ``tune + n_steps`` total simulation steps.
    tune
        Warmup simulation steps discarded before retention. Unlike
        ``pm.sample``'s ``tune``, this only controls which steps are retained —
        it is not step-size adaptation.
    step_size
        Euler-Maruyama step size. Default ``None`` enables the tuning-free FUSE
        adaptive schedule (Sharrock & Nemeth 2025). Pass a positive float
        (e.g. ``1e-3``) for a fixed step size instead.
    r_eps
        FUSE schedule floor ``r_ε`` (default ``1e-5``). Used only when
        ``step_size=None``.
    learning_rate
        Scales the log-score WGF interaction in the Euler-Maruyama drift (the paper's
        :math:`\\lambda_n`; see Sec. ``Computation via Wasserstein Gradient Flows`` in
        McLatchie et al. 2025, https://arxiv.org/abs/2510.01915). Default ``1.0``.
        The compiled interaction sums over observations; this factor scales that sum
        in :func:`~pymc_prop.particles.time_step`.
    random_seed
        Seeds particle initialization (one independent prior draw per particle,
        with finite-logp retry) and the Euler-Maruyama noise.
    coords, dims
        Optional coordinate and dimension names merged with the PyMC model
        definitions before building the DataTree.
    include_log_likelihood, include_observed_data, include_sample_stats
        Control which optional DataTree groups are populated. Set
        ``include_log_likelihood=False`` to skip the post-sampling logp pass
        when only particle trajectories are needed.
    datatree_kwargs
        Extra keyword arguments forwarded to the internal DataTree builder
        (e.g. ``name``). ``coords``, ``dims``, and ``include_*`` flags should
        use the top-level parameters above; ``sample_dims`` cannot be overridden.

    See Also
    --------
    sample_posterior_predictive_pro
        In-sample mixture PPC and OOS forward sampling (contrast with
        ``mixture_log_predictive``).
    compile_drift_for_logscore
        Log-score WGF drift (compiled interaction + prior grad).
    LogScore
        Scoring-rule wrapper used by default.
    """
    model = modelcontext(model)

    if n_particles < 2:
        raise ValueError("n_particles must be at least 2.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if tune < 0:
        raise ValueError("tune must be non-negative.")
    if step_size is None:
        if r_eps <= 0:
            raise ValueError("r_eps must be positive when step_size is None (FUSE).")
        if r_eps > 1e-5:
            warnings.warn(
                f"r_eps={r_eps:g} is larger than the recommended default (1e-5). "
                "FUSE is typically less sensitive to an r_eps that is too small than one that "
                "is too large. Inspect fuse_step_size traces and particle behaviour if the "
                "schedule looks unstable.",
                UserWarning,
                stacklevel=2,
            )
    elif step_size <= 0:
        raise ValueError("step_size must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not model.observed_RVs:
        raise ValueError(
            "sample_pro requires a model with observed variables for log-score sampling."
        )
    if isinstance(scoring_rule, str):
        if scoring_rule != "log":
            raise ValueError("Only log-score is supported in this version.")
        scoring_rule = LogScore()

    mapper = make_point_mapper(model)

    fuse_diagnostics: dict[str, list[float]] | None = (
        {} if step_size is None else None
    )
    particles = run_sampler(
        model=model,
        mapper=mapper,
        scoring_rule=scoring_rule,
        n_particles=n_particles,
        n_steps=n_steps,
        tune=tune,
        step_size=step_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        r_eps=r_eps,
        fuse_diagnostics=fuse_diagnostics,
    )

    fuse_stats = (
        {key: np.asarray(values, dtype=float) for key, values in fuse_diagnostics.items()}
        if fuse_diagnostics is not None
        else None
    )

    dt = _pro_to_datatree(
        particles,
        model=model,
        mapper=mapper,
        tune=tune,
        learning_rate=learning_rate,
        coords=coords,
        dims=dims,
        include_log_likelihood=include_log_likelihood,
        include_observed_data=include_observed_data,
        include_sample_stats=include_sample_stats,
        datatree_kwargs=datatree_kwargs,
        fuse_stats=fuse_stats,
    )

    return dt
