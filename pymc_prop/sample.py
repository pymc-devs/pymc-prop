"""Public sampling entry for PrO posteriors (PyMC-style ``sample`` module)."""

from __future__ import annotations

from typing import Literal

from pymc.model import modelcontext

from pymc_prop.points import make_point_mapper
from pymc_prop.sampler import PrOResult, run_sampler
from pymc_prop.scoring import LogScore, ScoringRule


def sample_pro(
    model=None,
    scoring_rule: Literal["log"] | ScoringRule = "log",
    n_particles: int = 64,
    n_steps: int = 1000,
    burn_in: int = 200,
    thinning: int = 1,
    step_size: float = 1e-3,
    learning_rate: float = 1.0,
    random_seed: int | None = None,
) -> PrOResult:
    """Run the PrO particle sampler on a PyMC model.

    Free RVs must be native unconstrained; reparameterize manually for now.

    Parameters
    ----------
    burn_in
        Time steps to discard before retaining snapshots. Not ``pm.sample``
        ``tune``: discard-only along a fixed ``n_steps`` horizon.
    thinning
        After ``burn_in``, keep every ``thinning``-th time step.
    learning_rate
        Scales the log-score WGF interaction in the Euler-Maruyama drift (the paper's
        :math:`\\lambda_n`; see Sec. ``Computation via Wasserstein Gradient Flows`` in
        McLatchie et al. 2025, https://arxiv.org/abs/2510.01915). Default ``1.0``.
        The compiled interaction sums over observations; this factor scales that sum
        in :func:`~pymc_prop.particles.time_step`.
    random_seed
        Seeds particle initialization (one independent prior draw per particle,
        with finite-logp retry) and the Euler-Maruyama noise.

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
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thinning <= 0:
        raise ValueError("thinning must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if isinstance(scoring_rule, str):
        if scoring_rule != "log":
            raise ValueError("Only log-score is supported in this version.")
        scoring_rule = LogScore()

    mapper = make_point_mapper(model)

    return run_sampler(
        model=model,
        mapper=mapper,
        scoring_rule=scoring_rule,
        n_particles=n_particles,
        n_steps=n_steps,
        burn_in=burn_in,
        thinning=thinning,
        step_size=step_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )
