"""Scoring rules and Wasserstein gradient flow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from pymc_prop.compile import compile_drift_for_logscore
from pymc_prop.points import PointMapper


WGFReturn = tuple[np.ndarray, Dict[str, float]]
DriftReturn = tuple[np.ndarray, np.ndarray, float, float]


class ScoringRule:
    """Scoring rule interface for WGF drift compilation."""

    def compile_wgf(self, model, mapper: PointMapper) -> Callable[[np.ndarray, bool], WGFReturn]:
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

    def compile_wgf(self, model, mapper: PointMapper) -> Callable[[np.ndarray, bool], WGFReturn]:
        drift_fn = self.compile_drift(model, mapper)

        def wgf(particles: np.ndarray, diagnostics: bool = False) -> WGFReturn:
            if particles.shape[0] < 2:
                raise ValueError("Log-score WGF requires at least two particles.")
            # vectorised drift computation (WGF term only)
            wgf_grad, _prior_grad, clip_count, nonfinite_logp = drift_fn(particles)

            diag = {}
            if diagnostics:
                diag = {
                    "clip_count": float(clip_count),
                    "nonfinite_logp": float(nonfinite_logp),
                }
            return wgf_grad, diag

        return wgf
