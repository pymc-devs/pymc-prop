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


def raw_drift(wgf_grad: np.ndarray, prior_grad: np.ndarray) -> np.ndarray:
    """Raw interaction drift for FUSE gradient-energy accumulation.

    Returns ``wgf_grad - prior_grad`` (no ``learning_rate``). Used only for
    :math:`g_s^2` in the FUSE denominator; particle motion uses
    :func:`scaled_drift`.
    """
    return wgf_grad - prior_grad


def scaled_drift(
    wgf_grad: np.ndarray, prior_grad: np.ndarray, learning_rate: float
) -> np.ndarray:
    """Deterministic drift passed to :func:`time_step` (includes ``learning_rate``)."""
    return learning_rate * wgf_grad - prior_grad


def time_step(
    particles: np.ndarray,
    prior_grad: np.ndarray,
    wgf_grad: np.ndarray,
    step_size: float,
    learning_rate: float,
    rng: np.random.Generator,
    mapper: PointMapper | None = None,
) -> np.ndarray:
    """Advance particles one discrete-time step along the log-score WGF.

    ``learning_rate`` is :math:`\\lambda_n`, ``step_size`` is :math:`\\varepsilon`;
    drift comes from :func:`~pymc_prop.compile.compile_drift_for_logscore`
    (primal :math:`\\nabla_\\theta` laid out in dual flat coordinates).

    Diffusion uses mirror noise
    :math:`\\sigma(y)=\\exp(-\\tfrac12\\texttt{log\\_jac\\_det}(y))` when
    ``mapper`` has transforms; identity coordinates keep ``σ ≡ 1`` (same as
    isotropic Euler-Maruyama).
    """
    # drift = λ_n · wgf_grad − prior_grad  (primal ∇_θ; particles are dual y)
    drift = learning_rate * wgf_grad - prior_grad
    xi = rng.standard_normal(size=particles.shape)
    if mapper is None:
        noise_scale = 1.0
    else:
        noise_scale = mapper.noise_scale(particles)
    noise = np.sqrt(2.0 * step_size) * noise_scale * xi
    return particles - step_size * drift + noise
