"""Scoring rules and Wasserstein gradient flow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pymc_prop.compile import compile_drift_for_logscore
from pymc_prop.points import PointMapper


DriftReturn = tuple[np.ndarray, np.ndarray]


class ScoringRule:
    """Scoring rule interface for Wasserstein gradient flow (WGF) drift compilation."""

    def compile_wgf(self, model, mapper: PointMapper) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def compile_drift(self, model, mapper: PointMapper) -> Callable[[np.ndarray], DriftReturn]:
        raise NotImplementedError


@dataclass
class LogScore(ScoringRule):
    """Log-score scoring rule for predictively oriented (PrO) particle sampling.

    Uses :func:`~pymc_prop.compile.compile_drift_for_logscore`. Requires
    continuous ``model.value_vars`` and at least two particles. Elementwise
    transforms are supported via mirror-mapped Wasserstein gradient flow (WGF);
    simplex maps are rejected.

    :meth:`compile_drift` returns ``(wgf_grad, prior_grad)`` per time step.
    :meth:`compile_wgf` returns the interaction term only.

    Parameters
    ----------
    log_ratio_clip
        Clip log likelihood ratios before exponentiating (stability only).
    eps
        Floor for leave-one-particle-out normalising sums in the compiled graph.
    """

    log_ratio_clip: float = 10.0
    eps: float = 1e-300

    def compile_drift(self, model, mapper: PointMapper) -> Callable[[np.ndarray], DriftReturn]:
        return compile_drift_for_logscore(
            mapper,
            model,
            log_ratio_clip=self.log_ratio_clip,
            eps=self.eps,
        )

    def compile_wgf(self, model, mapper: PointMapper) -> Callable[[np.ndarray], np.ndarray]:
        drift_fn = self.compile_drift(model, mapper)

        def wgf(particles: np.ndarray) -> np.ndarray:
            if particles.shape[0] < 2:
                raise ValueError("Log-score requires at least two particles.")
            wgf_grad, _prior_grad = drift_fn(particles)
            return wgf_grad

        return wgf
