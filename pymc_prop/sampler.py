"""Simulation loop for PrO particles."""

from __future__ import annotations

from typing import List

import numpy as np

from pymc_prop.compile import compile_batched_prior_grad
from pymc_prop.fuse import (
    DEFAULT_R_EPS,
    FUSE_GRADIENT_ENERGY_STAT,
    FUSE_HALF_STEP_DISTANCE_SQ_STAT,
    FUSE_STEP_SIZE_STAT,
    FuseState,
    fuse_step_size,
)
from pymc_prop.particles import initialize_particles, time_step
from pymc_prop.points import PointMapper
from pymc_prop.scoring import LogScore, ScoringRule


def run_sampler(
    model,
    mapper: PointMapper,
    scoring_rule: ScoringRule,
    n_particles: int,
    n_steps: int,
    tune: int,
    step_size: float | None,
    learning_rate: float,
    random_seed: int | None,
    r_eps: float = DEFAULT_R_EPS,
    fuse_diagnostics: dict[str, list[float]] | None = None,
) -> np.ndarray:
    """Run the PrO particle simulation loop.

    Returns retained particle arrays with shape ``(n_steps, n_particles,
    n_params)``. The loop runs ``tune + n_steps`` Euler-Maruyama steps; each draw after
    warmup is an empirical particle measure at a retained simulation step.

    When ``step_size`` is ``None``, the tuning-free FUSE schedule (Sharrock &
    Nemeth 2025) adapts ``η_t`` from raw and scaled drift fields; ``r_eps`` is
    the schedule floor. FUSE state persists across the full ``tune + n_steps``
    loop (no reset at the tune boundary).
    """
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2.")
    if step_size is None:
        if r_eps <= 0:
            raise ValueError("r_eps must be positive when step_size is None (FUSE).")
    elif step_size <= 0:
        raise ValueError("step_size must be positive.")

    rng = np.random.default_rng(random_seed)

    # Sec. 5: particles in unconstrained value_vars space
    particles = initialize_particles(model, mapper, n_particles, rng)

    wgf_fn = scoring_rule.compile_wgf(model, mapper)
    batched_prior_grad_fn = None
    drift_fn = None
    if isinstance(scoring_rule, LogScore):
        # fused log-score path: one compiled call per step
        drift_fn = scoring_rule.compile_drift(model, mapper)
    else:
        batched_prior_grad_fn = compile_batched_prior_grad(mapper, model)

    use_fuse = step_size is None
    fuse_state = FuseState() if use_fuse else None

    retained: List[np.ndarray] = []

    for step in range(tune + n_steps):
        # compile_drift_for_logscore -> time_step
        if drift_fn is not None:
            wgf_grad, prior_grad = drift_fn(particles)
        else:
            wgf_grad = wgf_fn(particles)
            assert batched_prior_grad_fn is not None
            prior_grad = np.asarray(batched_prior_grad_fn(particles), dtype=float)

        if use_fuse:
            assert fuse_state is not None
            eta, fuse_state, diag = fuse_step_size(
                particles,
                wgf_grad,
                prior_grad,
                learning_rate,
                fuse_state,
                r_eps,
            )
            if fuse_diagnostics is not None and step >= tune:
                fuse_diagnostics.setdefault(FUSE_GRADIENT_ENERGY_STAT, []).append(diag.gradient_energy)
                fuse_diagnostics.setdefault(FUSE_HALF_STEP_DISTANCE_SQ_STAT, []).append(diag.half_step_distance_sq)
                fuse_diagnostics.setdefault(FUSE_STEP_SIZE_STAT, []).append(diag.step_size)
        else:
            eta = step_size

        particles = time_step(
            particles, prior_grad, wgf_grad, eta, learning_rate, rng
        )

        if step >= tune:
            retained.append(particles.copy())

    if retained:
        return np.stack(retained, axis=0)
    return np.empty((0, n_particles, particles.shape[1]))
