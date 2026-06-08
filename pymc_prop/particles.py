"""Particle initialization and Euler-Maruyama updates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn

from pymc_prop.points import PointMapper


def _init_prior(
    model,
    mapper: PointMapper,
    n_particles: int,
    seeds: np.ndarray,
    *,
    max_retries: int = 10,
    logp_fn: Callable[[dict[str, np.ndarray]], Any] | None = None,
) -> np.ndarray:
    """Draw prior initial points per particle with finite-logp retry.

    Modeled on PyMC ``_init_jitter`` without jitter: each particle draws from
    ``make_initial_point_fn(..., default_strategy="prior")``; non-finite logp
    triggers resampling up to ``max_retries``, then ``model.check_start_vals``.
    """
    ipfn = make_initial_point_fn(
        model=model,
        default_strategy="prior",
        jitter_rvs=set(),
        return_transformed=True,
    )

    if logp_fn is None:
        model_logp_fn = model.compile_logp()
    else:
        model_logp_fn = logp_fn

    population = []
    for seed in seeds:
        seed = int(seed)
        rng = np.random.default_rng(seed)
        for i in range(max_retries + 1):
            point = ipfn(seed)
            for name, value in mapper.start_point.items():
                if name not in point:
                    point[name] = np.asarray(value, dtype=float)

            point_logp = model_logp_fn(point)
            if not np.isfinite(point_logp):
                if i == max_retries:
                    model.check_start_vals(point)
                seed = int(rng.integers(2**30, dtype=np.int64))
            else:
                break

        population.append(DictToArrayBijection.map(point).data)

    return np.asarray(population, dtype=float)


def initialize_particles(
    model,
    mapper: PointMapper,
    n_particles: int,
    rng: np.random.Generator,
    *,
    max_retries: int = 10,
) -> np.ndarray:
    """Draw one prior sample per particle.

    Each particle is an independent prior draw in unconstrained
    ``value_vars`` space (Sec. 5, McLatchie et al. 2025). Non-finite joint
    logp triggers resampling with a new seed (up to ``max_retries``).
    """
    seeds = rng.integers(2**30, size=n_particles)
    return _init_prior(
        model=model,
        mapper=mapper,
        n_particles=n_particles,
        seeds=seeds,
        max_retries=max_retries,
    )


def time_step(
    particles: np.ndarray,
    prior_grad: np.ndarray,
    wgf_grad: np.ndarray,
    step_size: float,
    learning_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance particles one discrete-time step along the log-score WGF.

    ``learning_rate`` is :math:`\\lambda_n`, ``step_size`` is :math:`\\varepsilon`;
    drift comes from :func:`~pymc_prop.compile.compile_drift_for_logscore`.
    """
    # drift = λ_n · wgf_grad − prior_grad
    drift = learning_rate * wgf_grad - prior_grad
    noise = np.sqrt(2.0 * step_size) * rng.standard_normal(size=particles.shape)
    return particles - step_size * drift + noise
