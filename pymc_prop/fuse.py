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
    """Mutable FUSE schedule state (Sharrock & Nemeth 2025, Sec. 5.1.1).

    ``reference_half_step`` is the fixed initial half-step cloud
    (:math:`x_{1/2}`). ``grad_energy`` accumulates mean squared
    **raw** drift (:math:`g_s^2`); ``r_bar`` tracks the running distance floor.
    ``learning_rate`` is not applied when accumulating ``grad_energy``.

    ``reference_half_step_set`` starts ``False``. The first :func:`fuse_step_size`
    call freezes ``reference_half_step``, sets ``η_0 = r_ε``, and initialises
    ``r_bar`` / ``grad_energy``; it then sets ``reference_half_step_set`` to
    ``True``. Later calls run the adaptive update and compare each new
    half-step against ``reference_half_step``.
    """

    reference_half_step: np.ndarray | None = None
    r_bar: float = 0.0
    grad_energy: float = 0.0
    reference_half_step_set: bool = False


def fuse_grad_energy(raw: np.ndarray) -> float:
    """Empirical gradient energy ``(1/n) Σ_i ‖raw[i]‖²`` for FUSE."""
    return float(np.mean(np.sum(raw * raw, axis=1)))


def fuse_distance(
    reference_half_step: np.ndarray, current_half_step: np.ndarray
) -> float:
    """Empirical half-step distance between reference and current half-step clouds."""
    diff = reference_half_step - current_half_step
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

    Particle discretization of the forward-flow FUSE schedule (Sharrock & Nemeth
    2025, Sec. 5.1.1)::

        η_t = r̄_t / sqrt(G_t)

        G_t = Σ_{s=1}^t g_s²,   g_s² ≈ (1/n) Σ_i ‖ζ_s(x_s^i)‖²

        d_s² ≈ (1/n) Σ_i ‖x_{1/2}^i - x_{s-1/2}^i‖²

        r̄_t = max(r_ε, max_{1≤s≤t} d_s)

    Half-steps and :func:`~pymc_prop.particles.time_step` use
    :func:`~pymc_prop.particles.scaled_drift`; ``g_s²`` uses
    :func:`~pymc_prop.particles.raw_drift` (no ``learning_rate``).

    When ``reference_half_step_set`` is ``False``, the initial schedule step
    fixes ``reference_half_step`` (:math:`x_{1/2}`), sets
    ``η_0 = r_ε``, and seeds ``r̄_0 = r_ε`` with ``G_0 = 0``. Each later call
    runs the adaptive update.
    """
    scaled = scaled_drift(wgf_grad, prior_grad, learning_rate)
    raw = raw_drift(wgf_grad, prior_grad)

    if not state.reference_half_step_set:
        # Initial schedule step: freeze reference half-step; adaptive loop after this.
        step_size = r_eps
        state.reference_half_step = particles - step_size * scaled
        state.r_bar = r_eps
        state.grad_energy = 0.0
        state.reference_half_step_set = True
        diag = FuseStepDiagnostics(
            step_size=step_size,
            gradient_energy=fuse_grad_energy(raw),
            half_step_distance_sq=0.0,
        )
        return step_size, state, diag

    # Incremental gradient energy at the current cloud (raw drift).
    g_inc = fuse_grad_energy(raw)

    # Add to cumulative gradient energy.
    state.grad_energy += g_inc

    # Step size = running distance floor / sqrt(cumulative gradient energy).
    step_size = state.r_bar / np.sqrt(state.grad_energy + _GRAD_EPS)

    # Deterministic half-step (scaled drift, includes learning_rate).
    current_half_step = particles - step_size * scaled
    assert state.reference_half_step is not None

    # Distance from the fixed reference half-step to the current half-step.
    d_next = fuse_distance(state.reference_half_step, current_half_step)

    # Running max of prior floor, r_eps, and half-step distance.
    state.r_bar = max(state.r_bar, max(r_eps, d_next))

    diag = FuseStepDiagnostics(
        step_size=step_size,
        gradient_energy=g_inc,
        half_step_distance_sq=d_next * d_next,
    )
    return step_size, state, diag
