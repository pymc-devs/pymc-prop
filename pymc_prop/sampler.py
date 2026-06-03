"""Simulation loop for PrO particles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from pymc_prop.compile import compile_batched_prior_grad
from pymc_prop.particles import initialize_particles, time_step
from pymc_prop.points import PointMapper
from pymc_prop.scoring import LogScore, ScoringRule


@dataclass
class PrOResult:
    """Interim sampler output; ArviZ ``InferenceData`` planned (D5).

    ``particles`` has shape ``(n_samples, n_particles, n_params)``. Each
    retained slice is an **empirical particle measure**
    :math:`\\widehat{Q}[t_i] = \\frac{1}{p}\\sum_{j=1}^p
    \\delta_{\\vartheta_{t_i}^{(j)}}`, kept after ``burn_in`` every
    ``thinning`` time steps.
    """

    particles: np.ndarray


def run_sampler(
    model,
    mapper: PointMapper,
    scoring_rule: ScoringRule,
    n_particles: int,
    n_steps: int,
    burn_in: int,
    thinning: int,
    step_size: float,
    learning_rate: float,
    random_seed: int | None,
) -> PrOResult:
    """Run the PrO particle simulation loop.

    Each step: compile drift, then :func:`~pymc_prop.particles.time_step`.
    """
    rng = np.random.default_rng(random_seed)

    # Sec. 5: particles in unconstrained value_vars space (mapper + jittered init)
    start = mapper.ravel(mapper.start_point)
    particles = initialize_particles(start, n_particles, rng)

    wgf_fn = scoring_rule.compile_wgf(model, mapper)
    batched_prior_grad_fn = None
    drift_fn = None
    if isinstance(scoring_rule, LogScore):
        # fused log-score path: one compiled call per step
        drift_fn = scoring_rule.compile_drift(model, mapper)
    else:
        batched_prior_grad_fn = compile_batched_prior_grad(mapper, model)

    retained: List[np.ndarray] = []

    for step in range(n_steps):
        # compile_drift_for_logscore -> time_step
        if drift_fn is not None:
            wgf_grad, prior_grad = drift_fn(particles)
        else:
            wgf_grad = wgf_fn(particles)
            assert batched_prior_grad_fn is not None
            prior_grad = np.asarray(batched_prior_grad_fn(particles), dtype=float)

        particles = time_step(
            particles, prior_grad, wgf_grad, step_size, learning_rate, rng
        )

        if step >= burn_in and (step - burn_in) % thinning == 0:
            retained.append(particles.copy())

    retained_arr = np.stack(retained, axis=0) if retained else np.empty((0, n_particles, particles.shape[1]))
    return PrOResult(particles=retained_arr)
