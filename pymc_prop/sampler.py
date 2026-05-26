"""Simulation loop for PrO particles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from pymc_prop.compile import (
    compile_batched_prior_grad,
    compile_observed_logp,
    count_observations,
)
from pymc_prop.particles import em_step, initialize_particles, particle_spread
from pymc_prop.points import PointMapper
from pymc_prop.scoring import LogScore, ScoringRule


@dataclass
class PrOResult:
    particles: np.ndarray
    diagnostics: Dict[str, np.ndarray]


def run_sampler(
    model,
    mapper: PointMapper,
    scoring_rule: ScoringRule,
    n_particles: int,
    n_steps: int,
    burn_in: int,
    thinning: int,
    step_size: float,
    lambda_n: float | None,
    random_seed: int | None,
    diag_stride: int,
) -> PrOResult:
    rng = np.random.default_rng(random_seed)
    diag_stride = max(1, int(diag_stride))

    # initialise particles around the model start point
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

    if lambda_n is None:
        logp_fn = compile_observed_logp(model)
        n_obs = count_observations(logp_fn, mapper)
        lambda_n = np.sqrt(float(n_obs))

    retained: List[np.ndarray] = []
    diag: Dict[str, List[float]] = {
        "step": [],
        "spread_mean": [],
        "spread_min": [],
        "spread_max": [],
        "clip_count": [],
        "nonfinite_logp": [],
    }

    for step in range(n_steps):
        if drift_fn is not None:
            wgf_grad, prior_grad, clip_count, nonfinite_logp = drift_fn(particles)
            wgf_diag = {"clip_count": float(clip_count), "nonfinite_logp": float(nonfinite_logp)}
        else:
            wgf_grad, wgf_diag = wgf_fn(particles, diagnostics=(step % diag_stride == 0))
            assert batched_prior_grad_fn is not None
            prior_grad = np.asarray(batched_prior_grad_fn(particles), dtype=float)

        # Euler-Maruyama update
        particles = em_step(particles, prior_grad, wgf_grad, step_size, float(lambda_n), rng)

        if step % diag_stride == 0:
            spread = particle_spread(particles)
            diag["step"].append(float(step))
            diag["spread_mean"].append(spread["spread_mean"])
            diag["spread_min"].append(spread["spread_min"])
            diag["spread_max"].append(spread["spread_max"])
            diag["clip_count"].append(wgf_diag.get("clip_count", 0.0))
            diag["nonfinite_logp"].append(wgf_diag.get("nonfinite_logp", 0.0))

        # retain post burn-in / thinning snapshots
        if step >= burn_in and (step - burn_in) % thinning == 0:
            retained.append(particles.copy())

    retained_arr = np.stack(retained, axis=0) if retained else np.empty((0, n_particles, particles.shape[1]))
    diag_arr = {k: np.asarray(v) for k, v in diag.items()}
    return PrOResult(particles=retained_arr, diagnostics=diag_arr)
