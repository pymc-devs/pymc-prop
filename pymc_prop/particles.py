"""Particle initialization and Euler-Maruyama updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn

from pymc_prop.points import PointMapper

_FUSE_GRAD_EPS = 1e-16


def _init_prior(
    model,
    mapper: PointMapper,
    n_particles: int,
    seeds: np.ndarray,
    *,
    max_retries: int = 10,
    logp_fn: Callable[[dict[str, np.ndarray]], Any] | None = None,
) -> np.ndarray:
    """Draw prior initial points per particle with finite-logp retry.

    Modeled on PyMC ``_init_jitter`` without jitter: each particle draws from
    ``make_initial_point_fn(..., default_strategy="prior")``; non-finite logp
    triggers resampling up to ``max_retries``, then ``model.check_start_vals``.
    """
    ipfn = make_initial_point_fn(
        model=model,
        default_strategy="prior",
        jitter_rvs=set(),
        return_transformed=True,
    )

    if logp_fn is None:
        model_logp_fn = model.compile_logp()
    else:
        model_logp_fn = logp_fn

    population = []
    for seed in seeds:
        seed = int(seed)
        rng = np.random.default_rng(seed)
        for i in range(max_retries + 1):
            point = ipfn(seed)
            for name, value in mapper.start_point.items():
                if name not in point:
                    point[name] = np.asarray(value, dtype=float)

            point_logp = model_logp_fn(point)
            if not np.isfinite(point_logp):
                if i == max_retries:
                    model.check_start_vals(point)
                seed = int(rng.integers(2**30, dtype=np.int64))
            else:
                break

        population.append(DictToArrayBijection.map(point).data)

    return np.asarray(population, dtype=float)


def initialize_particles(
    model,
    mapper: PointMapper,
    n_particles: int,
    rng: np.random.Generator,
    *,
    max_retries: int = 10,
) -> np.ndarray:
    """Draw one prior sample per particle.

    Each particle is an independent prior draw in unconstrained
    ``value_vars`` space (Sec. 5, McLatchie et al. 2025). Non-finite joint
    logp triggers resampling with a new seed (up to ``max_retries``).
    """
    seeds = rng.integers(2**30, size=n_particles)
    return _init_prior(
        model=model,
        mapper=mapper,
        n_particles=n_particles,
        seeds=seeds,
        max_retries=max_retries,
    )


@dataclass
class FuseStepDiagnostics:
    """Per-step FUSE schedule ingredients (Sharrock & Nemeth 2025)."""

    eta: float
    g_sq: float
    d_sq: float


@dataclass
class FuseState:
    """Mutable FUSE schedule state (Sharrock & Nemeth 2025, Sec. 5.1.1).

    ``half_ref`` stores the bootstrap scaled half-step; ``grad_energy`` accumulates
    mean squared **raw** drift (``g_s^2``); ``r_bar`` tracks the running distance
    floor. ``learning_rate`` is not applied when accumulating ``grad_energy``.
    """

    half_ref: np.ndarray | None = None
    r_bar: float = 0.0
    grad_energy: float = 0.0
    bootstrapped: bool = False


def raw_drift(wgf_grad: np.ndarray, prior_grad: np.ndarray) -> np.ndarray:
    """Raw interaction drift for FUSE gradient-energy accumulation.

    Returns ``wgf_grad - prior_grad`` (no ``learning_rate``). Used only for
    ``g_s^2`` in the FUSE denominator; particle motion uses :func:`scaled_drift`.
    """
    return wgf_grad - prior_grad


def scaled_drift(
    wgf_grad: np.ndarray, prior_grad: np.ndarray, learning_rate: float
) -> np.ndarray:
    """Deterministic drift passed to :func:`time_step` (includes ``learning_rate``)."""
    return learning_rate * wgf_grad - prior_grad


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
    """Return ``eta_t``, updated state, and per-step FUSE diagnostics.

    On the bootstrap step, ``eta_0 = r_\\varepsilon`` and ``half_ref`` is set from
    the scaled deterministic half-step. Thereafter ``g_s^2`` uses :func:`raw_drift`
    while half-steps and :func:`time_step` use :func:`scaled_drift`.

    ``g_sq`` is the incremental gradient energy at the current cloud; ``d_sq`` is
    the squared empirical half-step distance (``0`` on bootstrap).
    """
    scaled = scaled_drift(wgf_grad, prior_grad, learning_rate)
    raw = raw_drift(wgf_grad, prior_grad)

    if not state.bootstrapped:
        eta = r_eps
        state.half_ref = particles - eta * scaled
        state.r_bar = r_eps
        state.grad_energy = 0.0
        state.bootstrapped = True
        diag = FuseStepDiagnostics(eta=eta, g_sq=fuse_grad_energy(raw), d_sq=0.0)
        return eta, state, diag

    g_inc = fuse_grad_energy(raw)
    state.grad_energy += g_inc
    eta = state.r_bar / np.sqrt(state.grad_energy + _FUSE_GRAD_EPS)
    half = particles - eta * scaled
    assert state.half_ref is not None
    d_next = fuse_distance(state.half_ref, half)
    state.r_bar = max(state.r_bar, max(r_eps, d_next))
    diag = FuseStepDiagnostics(eta=eta, g_sq=g_inc, d_sq=d_next * d_next)
    return eta, state, diag


def time_step(
    particles: np.ndarray,
    prior_grad: np.ndarray,
    wgf_grad: np.ndarray,
    step_size: float,
    learning_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance particles one discrete-time step along the log-score WGF.

    ``learning_rate`` is :math:`\\lambda_n`, ``step_size`` is :math:`\\varepsilon`;
    drift comes from :func:`~pymc_prop.compile.compile_drift_for_logscore`.
    """
    # drift = λ_n · wgf_grad − prior_grad
    drift = learning_rate * wgf_grad - prior_grad
    noise = np.sqrt(2.0 * step_size) * rng.standard_normal(size=particles.shape)
    return particles - step_size * drift + noise
