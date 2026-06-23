"""Tuning-free FUSE adaptive step-size schedule (Sharrock & Nemeth 2025)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymc_prop.particles import raw_drift, scaled_drift

_GRAD_EPS = 1e-16

# Keys written to ``sample_stats`` when ``step_size=None``.
FUSE_STEP_SIZE_STAT = "fuse_step_size"
FUSE_GRADIENT_ENERGY_STAT = "fuse_gradient_energy"
FUSE_HALF_STEP_DISTANCE_SQ_STAT = "fuse_half_step_distance_sq"


@dataclass
class FuseStepDiagnostics:
    """Per-step FUSE schedule diagnostics (Sharrock & Nemeth 2025, Sec. 5.1.1).

    Paper symbols: ``step_size`` is :math:`\\eta_t`; ``gradient_energy`` is the
    incremental :math:`g_s^2` at the current cloud (not cumulative :math:`G_t`);
    ``half_step_distance_sq`` is :math:`d_s^2` (``0`` on the bootstrap step).
    """

    step_size: float
    gradient_energy: float
    half_step_distance_sq: float


@dataclass
class FuseState:
    """Mutable FUSE schedule state (Sharrock & Nemeth 2025, Sec. 5.1.1).

    ``half_ref`` stores the bootstrap scaled half-step; ``grad_energy`` accumulates
    mean squared **raw** drift (:math:`g_s^2`); ``r_bar`` tracks the running
    distance floor. ``learning_rate`` is not applied when accumulating
    ``grad_energy``.
    """

    half_ref: np.ndarray | None = None
    r_bar: float = 0.0
    grad_energy: float = 0.0
    bootstrapped: bool = False


def fuse_grad_energy(raw: np.ndarray) -> float:
    """Empirical gradient energy ``(1/n) Σ_i ‖raw[i]‖²`` for FUSE."""
    return float(np.mean(np.sum(raw * raw, axis=1)))


def fuse_distance(half_ref: np.ndarray, half_scaled: np.ndarray) -> float:
    """Empirical half-step distance ``sqrt((1/n) Σ_i ‖half_ref[i] - half[i]‖²)``."""
    diff = half_ref - half_scaled
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def fuse_step_size(
    particles: np.ndarray,
    wgf_grad: np.ndarray,
    prior_grad: np.ndarray,
    learning_rate: float,
    state: FuseState,
    r_eps: float,
) -> tuple[float, FuseState, FuseStepDiagnostics]:
    """Return adaptive step size, updated state, and per-step FUSE diagnostics.

    On the bootstrap step, ``eta_0 = r_\\varepsilon`` and ``half_ref`` is set from
    the scaled deterministic half-step. Thereafter gradient energy uses
    :func:`~pymc_prop.particles.raw_drift` while half-steps and
    :func:`~pymc_prop.particles.time_step` use
    :func:`~pymc_prop.particles.scaled_drift`.
    """
    scaled = scaled_drift(wgf_grad, prior_grad, learning_rate)
    raw = raw_drift(wgf_grad, prior_grad)

    if not state.bootstrapped:
        step_size = r_eps
        state.half_ref = particles - step_size * scaled
        state.r_bar = r_eps
        state.grad_energy = 0.0
        state.bootstrapped = True
        diag = FuseStepDiagnostics(
            step_size=step_size,
            gradient_energy=fuse_grad_energy(raw),
            half_step_distance_sq=0.0,
        )
        return step_size, state, diag

    g_inc = fuse_grad_energy(raw)
    state.grad_energy += g_inc
    step_size = state.r_bar / np.sqrt(state.grad_energy + _GRAD_EPS)
    half = particles - step_size * scaled
    assert state.half_ref is not None
    d_next = fuse_distance(state.half_ref, half)
    state.r_bar = max(state.r_bar, max(r_eps, d_next))
    diag = FuseStepDiagnostics(
        step_size=step_size,
        gradient_energy=g_inc,
        half_step_distance_sq=d_next * d_next,
    )
    return step_size, state, diag
