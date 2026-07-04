"""Tuning-free FUSE adaptive step-size schedule (Sharrock & Nemeth 2025)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymc_prop.particles import raw_drift, scaled_drift

DEFAULT_R_EPS = 1e-5
# Floor on cumulative gradient energy in the FUSE denominator (not r_eps).
_GRAD_EPS = 1e-12

# Keys written to ``sample_stats`` when ``step_size=None``.
FUSE_STEP_SIZE_STAT = "fuse_step_size"
FUSE_GRADIENT_ENERGY_STAT = "fuse_gradient_energy"
FUSE_HALF_STEP_DISTANCE_SQ_STAT = "fuse_half_step_distance_sq"


@dataclass
class FuseStepDiagnostics:
    """Per-step FUSE schedule diagnostics (Sharrock & Nemeth 2025, Sec. 5.1.1).

    Paper symbols: ``step_size`` is :math:`\\eta_t`; ``gradient_energy`` is the
    incremental :math:`g_s^2` at the current cloud (not cumulative :math:`G_t`);
    ``half_step_distance_sq`` is :math:`d_s^2` (``0`` on the initial schedule
    step, before a second half-step exists).
    """

    step_size: float
    gradient_energy: float
    half_step_distance_sq: float


@dataclass
class FuseState:
    """Mutable FUSE schedule state after bootstrap (Sharrock & Nemeth 2025, Sec. 5.1.1).

    Created by :func:`fuse_bootstrap_step`. ``reference_half_step`` is the fixed
    initial half-step cloud (:math:`x_{1/2}`). ``grad_energy`` accumulates mean
    squared **raw** drift (:math:`g_s^2`); ``r_bar`` tracks the running distance
    floor. ``learning_rate`` is not applied when accumulating ``grad_energy``.
    """

    reference_half_step: np.ndarray
    r_bar: float
    grad_energy: float


def fuse_grad_energy(raw: np.ndarray) -> float:
    """Empirical gradient energy ``(1/n) Σ_i ‖raw[i]‖²`` for FUSE."""
    return float(np.mean(np.sum(raw * raw, axis=1)))


def fuse_distance(
    reference_half_step: np.ndarray, current_half_step: np.ndarray
) -> float:
    """Empirical half-step distance between reference and current half-step clouds."""
    diff = reference_half_step - current_half_step
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def fuse_bootstrap_step(
    particles: np.ndarray,
    wgf_grad: np.ndarray,
    prior_grad: np.ndarray,
    learning_rate: float,
    r_eps: float,
) -> tuple[float, FuseState, FuseStepDiagnostics]:
    """Initial FUSE schedule step (:math:`t = 0`).

    Sets :math:`\\eta_0 = r_\\varepsilon`, freezes the reference half-step cloud
    :math:`x_{1/2}`, and seeds :math:`\\bar r_0 = r_\\varepsilon` with
    :math:`G_0 = 0`. Half-steps use :func:`~pymc_prop.particles.scaled_drift`;
    reported ``gradient_energy`` uses :func:`~pymc_prop.particles.raw_drift`
    (no ``learning_rate``) but is not yet accumulated into ``grad_energy``.
    """
    scaled = scaled_drift(wgf_grad, prior_grad, learning_rate)
    raw = raw_drift(wgf_grad, prior_grad)

    # t = 0: η_0 = r_ε; freeze x_{1/2}; seed r̄_0 = r_ε, G_0 = 0
    step_size = r_eps
    state = FuseState(
        reference_half_step=particles - step_size * scaled,  # x_{1/2}
        r_bar=r_eps,
        grad_energy=0.0,
    )
    diag = FuseStepDiagnostics(
        step_size=step_size,
        gradient_energy=fuse_grad_energy(raw),  # g_0² (diagnostic only; not in G_t yet)
        half_step_distance_sq=0.0,  # no d_s until a second half-step exists
    )
    return step_size, state, diag


def fuse_adaptive_step(
    particles: np.ndarray,
    wgf_grad: np.ndarray,
    prior_grad: np.ndarray,
    learning_rate: float,
    state: FuseState,
    r_eps: float,
) -> tuple[float, FuseState, FuseStepDiagnostics]:
    """Adaptive FUSE schedule update (:math:`t \\geq 1`).

    Particle discretization of the forward-flow FUSE schedule (Sharrock & Nemeth
    2025, Sec. 5.1.1)::

        η_t = r̄_t / sqrt(G_t)

        G_t = Σ_{s=1}^t g_s²,   g_s² ≈ (1/n) Σ_i ‖ζ_s(x_s^i)‖²

        d_s² ≈ (1/n) Σ_i ‖x_{1/2}^i - x_{s-1/2}^i‖²

        r̄_t = max(r_ε, max_{1≤s≤t} d_s)

    Half-steps and :func:`~pymc_prop.particles.time_step` use
    :func:`~pymc_prop.particles.scaled_drift`; ``g_s²`` uses
    :func:`~pymc_prop.particles.raw_drift` (no ``learning_rate``).
    """
    scaled = scaled_drift(wgf_grad, prior_grad, learning_rate)
    raw = raw_drift(wgf_grad, prior_grad)

    g_inc = fuse_grad_energy(raw)  # g_s² (raw ζ, no λ_n)
    state.grad_energy += g_inc  # G_t

    # η_t = r̄_t / sqrt(G_t)
    step_size = state.r_bar / np.sqrt(state.grad_energy + _GRAD_EPS)
    current_half_step = particles - step_size * scaled  # x_{s-1/2}
    d_next = fuse_distance(state.reference_half_step, current_half_step)  # d_s
    state.r_bar = max(state.r_bar, max(r_eps, d_next))  # r̄_t

    diag = FuseStepDiagnostics(
        step_size=step_size,
        gradient_energy=g_inc,
        half_step_distance_sq=d_next * d_next,
    )
    return step_size, state, diag


def fuse_step_size(
    particles: np.ndarray,
    wgf_grad: np.ndarray,
    prior_grad: np.ndarray,
    learning_rate: float,
    state: FuseState | None,
    r_eps: float,
) -> tuple[float, FuseState, FuseStepDiagnostics]:
    """Return adaptive step size, updated state, and per-step FUSE diagnostics.

    Dispatches to :func:`fuse_bootstrap_step` when ``state`` is ``None``;
    otherwise :func:`fuse_adaptive_step`. Prefer calling those functions
    directly when the schedule phase should be explicit at the call site.
    """
    if state is None:
        # t = 0 bootstrap (η_0, x_{1/2})
        return fuse_bootstrap_step(
            particles, wgf_grad, prior_grad, learning_rate, r_eps
        )
    # t ≥ 1 adaptive update (η_t, r̄_t, G_t)
    return fuse_adaptive_step(
        particles, wgf_grad, prior_grad, learning_rate, state, r_eps
    )
