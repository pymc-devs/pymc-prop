"""Public API for sampling PrO posteriors."""

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
    lambda_n: float | None = None,
    random_seed: int | None = None,
    diag_stride: int = 10,
) -> PrOResult:
    model = modelcontext(model)

    if n_particles < 2:
        raise ValueError("n_particles must be at least 2.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thinning <= 0:
        raise ValueError("thinning must be positive.")

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
        lambda_n=lambda_n,
        random_seed=random_seed,
        diag_stride=diag_stride,
    )
