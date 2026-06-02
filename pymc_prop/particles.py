"""Particle initialization and Euler-Maruyama updates."""

from __future__ import annotations

import numpy as np


def initialize_particles(
    start: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
    jitter: float = 0.1,
) -> np.ndarray:
    """Place particles around a shared flat start with Gaussian jitter.

    ``start`` is typically the raveled PyMC ``initial_point()`` (prior-centred
    values in unconstrained ``value_vars`` space). Each row is
    ``start + jitter * N(0, I)``, spreading an initial **empirical particle
    measure**
    :math:`\\widehat{Q}_0 = \\frac{1}{p}\\sum_j \\delta_{\\vartheta_0^{(j)}}`
    so leave-one-particle-out measures :math:`Q_t^{(j)}` can interact from
    step one (see :func:`~pymc_prop.compile.compile_drift_for_logscore`).
    """
    base = np.asarray(start, dtype=float)
    noise = jitter * rng.standard_normal(size=(n_particles, base.size))
    return base[None, :] + noise


def em_step(
    particles: np.ndarray,
    prior_grad: np.ndarray,
    wgf_grad: np.ndarray,
    step_size: float,
    learning_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Euler-Maruyama step for log-score PrO particles.

    ``learning_rate`` is :math:`\\lambda_n`, ``step_size`` is :math:`\\varepsilon`;
    drift comes from :func:`~pymc_prop.compile.compile_drift_for_logscore`.
    """
    # drift = λ_n · wgf_grad − prior_grad
    drift = learning_rate * wgf_grad - prior_grad
    noise = np.sqrt(2.0 * step_size) * rng.standard_normal(size=particles.shape)
    return particles - step_size * drift + noise
