"""FUSE adaptive step-size schedule (Sharrock & Nemeth 2025)."""

from __future__ import annotations

import warnings

import numpy as np
import pymc as pm
import pytest

from pymc_prop.fuse import (
    DEFAULT_R_EPS,
    FUSE_GRADIENT_ENERGY_STAT,
    FUSE_HALF_STEP_DISTANCE_SQ_STAT,
    fuse_adaptive_step,
    fuse_initial_step,
    fuse_distance,
    fuse_grad_energy,
    fuse_step_size,
)
from pymc_prop.particles import raw_drift, scaled_drift, time_step
from pymc_prop.points import make_point_mapper
from pymc_prop.sample import sample_pro
from pymc_prop.sampler import run_sampler
from pymc_prop.scoring import LogScore

MU = np.array([2.0, -1.0])
SIGMA = np.array([[1.0, 0.9], [0.9, 1.0]])
SIGMA_INV = np.linalg.inv(SIGMA)


def _gaussian_target_drift(particles: np.ndarray) -> np.ndarray:
    """Drift ζ(x) = -∇V(x) for the correlated Gaussian target."""
    return -(particles - MU) @ SIGMA_INV.T


def _simulate_fuse_forward_reference(
    rng: np.random.Generator,
    init_particles: np.ndarray,
    num_steps: int,
    r_eps: float,
    lambda_n: float = 1.0,
) -> np.ndarray:
    """Standalone NumPy reference for forward-flow FUSE on the Gaussian target."""
    # t = 0: η_0 = r_ε, freeze x_{1/2}
    eta0 = r_eps
    drifts0 = _gaussian_target_drift(init_particles)
    reference_half_step = init_particles + eta0 * drifts0
    noise0 = np.sqrt(2.0 * lambda_n * eta0) * rng.standard_normal(init_particles.shape)
    particles = reference_half_step + noise0
    r_bar = r_eps
    grad_energy = 0.0
    trajectory = [particles.copy()]

    for _ in range(num_steps - 1):
        # t ≥ 1: accumulate G_t, η_t = r̄_t / sqrt(G_t), update r̄_t
        drifts = _gaussian_target_drift(particles)
        grad_energy += float(np.mean(np.sum(drifts * drifts, axis=1)))
        eta = r_bar / np.sqrt(grad_energy + 1e-16)
        half = particles + eta * drifts
        noise = np.sqrt(2.0 * lambda_n * eta) * rng.standard_normal(particles.shape)
        particles = half + noise
        d_next = fuse_distance(reference_half_step, half)
        r_bar = max(r_bar, max(r_eps, d_next))
        trajectory.append(particles.copy())

    return np.stack(trajectory, axis=0)


def _simulate_fuse_pymc_helpers(
    rng: np.random.Generator,
    init_particles: np.ndarray,
    num_steps: int,
    r_eps: float,
    learning_rate: float = 1.0,
) -> np.ndarray:
    """FUSE loop using pymc_prop helpers (PrO sign convention)."""
    particles = init_particles.copy()
    prior_grad = np.zeros_like(particles)
    trajectory = []

    wgf_grad = -_gaussian_target_drift(particles)
    # t = 0 initial schedule step, then Euler-Maruyama step
    eta, fuse_state, _ = fuse_initial_step(
        particles, wgf_grad, prior_grad, learning_rate, r_eps
    )
    particles = time_step(
        particles, prior_grad, wgf_grad, eta, learning_rate, rng
    )
    trajectory.append(particles.copy())

    for _ in range(num_steps - 1):
        wgf_grad = -_gaussian_target_drift(particles)
        # t ≥ 1 adaptive η_t, then Euler-Maruyama step
        eta, fuse_state, _ = fuse_adaptive_step(
            particles,
            wgf_grad,
            prior_grad,
            learning_rate,
            fuse_state,
            r_eps,
        )
        particles = time_step(
            particles, prior_grad, wgf_grad, eta, learning_rate, rng
        )
        trajectory.append(particles.copy())

    return np.stack(trajectory, axis=0)


def test_fuse_schedule_parity_at_learning_rate_one():
    n_particles, dim = 32, 2
    num_steps = 25
    r_eps = 1e-4
    init = np.random.default_rng(99).standard_normal((n_particles, dim)) * 0.3 - 4.0

    rng_ref = np.random.default_rng(0)
    rng_pymc = np.random.default_rng(0)
    traj_ref = _simulate_fuse_forward_reference(rng_ref, init, num_steps, r_eps)
    traj_pymc = _simulate_fuse_pymc_helpers(rng_pymc, init, num_steps, r_eps)

    np.testing.assert_allclose(traj_pymc, traj_ref, rtol=0.0, atol=1e-12)


def test_fuse_raw_vs_scaled_learning_rate():
    """G accumulates raw drift; half-steps and time_step use scaled drift."""
    rng = np.random.default_rng(1)
    particles = rng.standard_normal((8, 2))
    wgf_grad = rng.standard_normal((8, 2))
    prior_grad = rng.standard_normal((8, 2))
    # lr != 1 so raw and scaled differ; at lr=1 both coincide numerically.
    learning_rate = 0.5
    r_eps = 1e-5

    raw = raw_drift(wgf_grad, prior_grad)
    scaled = scaled_drift(wgf_grad, prior_grad, learning_rate)
    assert not np.allclose(raw, scaled)

    _, fuse_state, _ = fuse_initial_step(
        particles, wgf_grad, prior_grad, learning_rate, r_eps
    )
    eta, fuse_state, _ = fuse_adaptive_step(
        particles, wgf_grad, prior_grad, learning_rate, fuse_state, r_eps
    )
    half = particles - eta * scaled

    expected_g = fuse_grad_energy(raw)
    wrong_g = fuse_grad_energy(scaled)
    assert fuse_state.grad_energy == pytest.approx(expected_g)
    assert fuse_state.grad_energy != pytest.approx(wrong_g)

    half_from_scaled = particles - eta * scaled
    half_from_raw = particles - eta * raw
    assert not np.allclose(half_from_scaled, half_from_raw)
    np.testing.assert_allclose(half, half_from_scaled)


def test_fuse_step_size_dispatches_initial_and_adaptive():
    """Thin dispatcher matches explicit initial then adaptive calls."""
    rng = np.random.default_rng(7)
    particles = rng.standard_normal((4, 2))
    wgf_grad = rng.standard_normal((4, 2))
    prior_grad = rng.standard_normal((4, 2))
    learning_rate = 1.0
    r_eps = 1e-5

    eta_init, state_init, diag_init = fuse_initial_step(
        particles, wgf_grad, prior_grad, learning_rate, r_eps
    )
    eta_disp, state_disp, diag_disp = fuse_step_size(
        particles, wgf_grad, prior_grad, learning_rate, None, r_eps
    )
    assert eta_init == pytest.approx(eta_disp)
    np.testing.assert_allclose(
        state_init.reference_half_step, state_disp.reference_half_step
    )
    assert state_init.r_bar == pytest.approx(state_disp.r_bar)
    assert state_init.grad_energy == pytest.approx(state_disp.grad_energy)
    assert diag_init.step_size == pytest.approx(diag_disp.step_size)
    assert diag_init.gradient_energy == pytest.approx(diag_disp.gradient_energy)
    assert diag_init.half_step_distance_sq == pytest.approx(
        diag_disp.half_step_distance_sq
    )

    eta_adapt, state_adapt, diag_adapt = fuse_adaptive_step(
        particles, wgf_grad, prior_grad, learning_rate, state_init, r_eps
    )
    eta_disp2, state_disp2, diag_disp2 = fuse_step_size(
        particles, wgf_grad, prior_grad, learning_rate, state_disp, r_eps
    )
    assert eta_adapt == pytest.approx(eta_disp2)
    assert state_adapt.r_bar == pytest.approx(state_disp2.r_bar)
    assert state_adapt.grad_energy == pytest.approx(state_disp2.grad_energy)
    assert diag_adapt.step_size == pytest.approx(diag_disp2.step_size)


def test_sample_pro_warns_when_r_eps_exceeds_default():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(3))

    with pytest.warns(UserWarning, match="larger than the recommended default"):
        sample_pro(
            model=model,
            n_particles=4,
            n_steps=2,
            tune=0,
            step_size=None,
            r_eps=DEFAULT_R_EPS * 10,
            random_seed=0,
            include_log_likelihood=False,
        )


def test_sample_pro_no_r_eps_warning_at_default():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(3))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        sample_pro(
            model=model,
            n_particles=4,
            n_steps=2,
            tune=0,
            step_size=None,
            r_eps=DEFAULT_R_EPS,
            random_seed=0,
            include_log_likelihood=False,
        )

    assert not any("r_eps" in str(w.message) for w in caught)


def test_run_sampler_fuse_rejects_nonpositive_r_eps():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(5))

    mapper = make_point_mapper(model)
    with pytest.raises(ValueError, match="r_eps must be positive"):
        run_sampler(
            model=model,
            mapper=mapper,
            scoring_rule=LogScore(),
            n_particles=4,
            n_steps=2,
            tune=0,
            step_size=None,
            learning_rate=1.0,
            random_seed=0,
            r_eps=0.0,
        )


def test_sample_pro_fuse_integration_smoke():
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=20)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)

    n_steps = 20
    dt = sample_pro(
        model=model,
        n_particles=8,
        n_steps=n_steps,
        tune=0,
        step_size=None,
        r_eps=1e-5,
        random_seed=123,
    )

    assert dt.posterior.sizes["draw"] == n_steps
    assert dt.posterior.sizes["chain"] == 8
    assert np.all(np.isfinite(dt.posterior["mu"].values))
    assert FUSE_GRADIENT_ENERGY_STAT in dt.sample_stats
    assert FUSE_HALF_STEP_DISTANCE_SQ_STAT in dt.sample_stats
    assert dt.sample_stats.sizes["draw"] == n_steps
    assert np.all(np.isfinite(dt.sample_stats[FUSE_GRADIENT_ENERGY_STAT].isel(chain=0).values))