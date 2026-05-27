"""Scoring rules and Wasserstein gradient flow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pymc_prop.compile import compile_drift_for_logscore
from pymc_prop.points import PointMapper


DriftReturn = tuple[np.ndarray, np.ndarray]


class ScoringRule:
    """Scoring rule interface for WGF drift compilation."""

    def compile_wgf(self, model, mapper: PointMapper) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def compile_drift(self, model, mapper: PointMapper) -> Callable[[np.ndarray], DriftReturn]:
        raise NotImplementedError


@dataclass
class LogScore(ScoringRule):
    """Log-score WGF with leave-one-out mixture weights."""

    log_ratio_clip: float = 10.0
    eps: float = 1e-300

    def compile_drift(self, model, mapper: PointMapper) -> Callable[[np.ndarray], DriftReturn]:
        """Fused batched WGF and prior gradients."""
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
                raise ValueError("Log-score WGF requires at least two particles.")
            wgf_grad, _prior_grad = drift_fn(particles)
            return wgf_grad

        return wgf
