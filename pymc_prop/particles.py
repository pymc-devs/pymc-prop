"""Particle initialization and Euler-Maruyama updates."""

from __future__ import annotations

from typing import Dict

import numpy as np


def initialize_particles(
    start: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
    jitter: float = 0.1,
) -> np.ndarray:
    """Jittered particles around a flat start vector."""
    base = np.asarray(start, dtype=float)
    noise = jitter * rng.standard_normal(size=(n_particles, base.size))
    return base[None, :] + noise


def em_step(
    particles: np.ndarray,
    prior_grad: np.ndarray,
    wgf_grad: np.ndarray,
    step_size: float,
    lambda_n: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Euler-Maruyama step for PrO particles."""
    drift = lambda_n * wgf_grad - prior_grad

    # simulate Brownian noise
    noise = np.sqrt(2.0 * step_size) * rng.standard_normal(size=particles.shape)

    # Euler-Maruyama update
    return particles - step_size * drift + noise


def particle_spread(particles: np.ndarray) -> Dict[str, float]:
    spread = np.std(particles, axis=0)
    return {
        "spread_mean": float(np.mean(spread)),
        "spread_min": float(np.min(spread)),
        "spread_max": float(np.max(spread)),
    }
