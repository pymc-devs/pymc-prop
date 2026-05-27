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
    ``start + jitter * N(0, I)``, giving a finite-:math:`p` cloud for the
    leave-one-out mixture
    :math:`Q_t^{(j)} \\approx \\frac{1}{p-1}\\sum_{\\ell\\neq j}
    \\delta_{\\vartheta^{(\\ell)}}` (McLatchie et al., 2025, Sec. 5).
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

    .. math::

        z \\leftarrow z - \\epsilon\\,(\\texttt{learning\\_rate}\\cdot W
        - \\nabla\\log\\pi) + \\sqrt{2\\epsilon}\\,\\xi

    where :math:`W` is the batched log-score WGF drift and
    :math:`\\nabla\\log\\pi` the prior score gradient (McLatchie et al.,
    2025, appendix log-score particle SDE; :math:`\\lambda_n` on the
    interaction term matches ``learning_rate``).
    """
    drift = learning_rate * wgf_grad - prior_grad
    noise = np.sqrt(2.0 * step_size) * rng.standard_normal(size=particles.shape)
    return particles - step_size * drift + noise
