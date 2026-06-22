"""Public sampling entry for PrO posteriors (PyMC-style ``sample`` module)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pymc.model import modelcontext
from xarray import DataTree

from pymc_prop.arviz import _pro_to_datatree
from pymc_prop.points import make_point_mapper
from pymc_prop.sampler import run_sampler
from pymc_prop.scoring import LogScore, ScoringRule


def sample_pro(
    model=None,
    scoring_rule: Literal["log"] | ScoringRule = "log",
    n_particles: int = 64,
    n_steps: int = 1000,
    tune: int = 200,
    step_size: float | None = 1e-3,
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
    ``observed_data``, ``log_likelihood``, and ``sample_stats`` groups.
    Retained steps map to ``draw``; simulation step
    numbers are in the ``step`` coordinate; particles map to ``chain``.

    Parameters
    ----------
    n_steps
        Number of retained simulation steps (maps to the ``draw`` dimension).
        The sampler runs ``tune + n_steps`` total simulation steps.
    tune
        Warmup simulation steps discarded before retention. Unlike
        ``pm.sample``'s ``tune``, this only controls which steps are retained —
        it is not step-size adaptation. Pass ``step_size=None`` to enable the
        tuning-free FUSE adaptive schedule (Sharrock & Nemeth 2025); ``r_eps``
        sets the schedule floor when FUSE is active.
    step_size
        Fixed Euler-Maruyama step size (default ``1e-3``). Pass ``None`` to
        enable FUSE adaptive step sizes instead of a fixed ``η``.
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
    elif step_size <= 0:
        raise ValueError("step_size must be positive.")
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

    return _pro_to_datatree(
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
