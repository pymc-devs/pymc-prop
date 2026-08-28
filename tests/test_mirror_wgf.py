"""Mirror-mapped WGF for constrained parameters."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from pymc_prop.compile import (
    compile_batched_observed_logp_score,
    compile_batched_prior_grad,
    compile_drift_for_logscore,
    compile_flat_prior_grad,
    compile_prior_gradient,
)
from pymc_prop.particles import time_step
from pymc_prop.points import make_point_mapper, require_mirror_compatible_transforms
from pymc_prop.sample import sample_pro


def test_identity_noise_scale_is_one():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.array([0.0, 0.5]))

    mapper = make_point_mapper(model)
    assert not mapper.has_transforms
    particles = np.array([[0.1], [-0.2], [1.5]])
    np.testing.assert_array_equal(mapper.noise_scale(particles), np.ones_like(particles))


def test_identity_time_step_matches_isotropic():
    """With no transforms, mirror step must bit-match σ≡1 Euler–Maruyama."""
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    particles = np.array([[0.3, -1.0], [0.1, 0.5]], dtype=float)
    prior = np.zeros_like(particles)
    wgf = np.ones_like(particles)
    step_size = 0.01
    lr = 1.0

    with pm.Model() as model:
        pm.Normal("mu", 0.0, 1.0)
        pm.Normal("tau", 0.0, 1.0)
        pm.Normal("y", 0.0, 1.0, observed=0.0)
    mapper = make_point_mapper(model)

    out_mirror = time_step(particles.copy(), prior, wgf, step_size, lr, rng_a, mapper)
    # Isotropic reference (mapper=None ⇒ σ=1)
    out_iso = time_step(particles.copy(), prior, wgf, step_size, lr, rng_b, None)
    np.testing.assert_array_equal(out_mirror, out_iso)


def test_log_transform_noise_scale():
    with pm.Model() as model:
        pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", 0.0, sigma=1.0, observed=0.0)

    mapper = make_point_mapper(model)
    assert mapper.has_transforms
    y = np.array([[0.0], [2.0], [-1.0]])
    expected = np.exp(-0.5 * y)
    np.testing.assert_allclose(mapper.noise_scale(y), expected, rtol=1e-10)


def test_dirichlet_simplex_rejected():
    with pm.Model() as model:
        pm.Dirichlet("w", a=np.ones(3))

    with pytest.raises(ValueError, match="Simplex|elementwise"):
        require_mirror_compatible_transforms(model)

    with pytest.raises(ValueError, match="Simplex|elementwise"):
        make_point_mapper(model)


def test_score_primal_scale_broadcast_shape():
    """score (p, n_obs, d) and primal_scale (p, d) broadcast via [:, None, :]."""
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=5)
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

    mapper = make_point_mapper(model)
    assert mapper.has_transforms
    base = mapper.ravel(mapper.start_point)
    particles = base[None, :] + 0.05 * rng.standard_normal((4, base.size))
    n_particles, n_flat = particles.shape
    n_obs = y.shape[0]

    _logp, score = compile_batched_observed_logp_score(model, mapper)(particles)
    scale = mapper.primal_scale(particles)
    assert score.shape == (n_particles, n_obs, n_flat)
    assert scale.shape == (n_particles, n_flat)
    scaled = score * scale[:, None, :]
    assert scaled.shape == score.shape

    wgf_grad, prior_grad = compile_drift_for_logscore(mapper, model)(particles)
    assert wgf_grad.shape == (n_particles, n_flat)
    assert prior_grad.shape == (n_particles, n_flat)


def test_halfnormal_prior_primal_grad_fd():
    """jacobian=False dual grad × exp(-log_jac_det) matches ∇_θ of HalfNormal."""
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        # Couple sigma into the likelihood so the score graph stays connected.
        pm.Normal("y", 0.0, sigma=sigma, observed=np.array([0.0]))

    mapper = make_point_mapper(model)
    drift_fn = compile_drift_for_logscore(mapper, model)
    # Need ≥2 particles for LOO; prior_grad is per-row independent of mix.
    y_dual = np.array([[0.5], [-0.3]])
    _wgf, prior_grad = drift_fn(y_dual)

    theta = np.exp(y_dual[:, 0])
    # HalfNormal(σ=1): ∇_θ log π(θ) = -θ
    expected = -theta
    np.testing.assert_allclose(prior_grad[:, 0], expected, rtol=1e-5, atol=1e-6)


def test_point_prior_grad_is_dual_particle_apis_are_primal():
    """Point helper stays ∇_y; flat/batched/drift apply primal conversion."""
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", 0.0, sigma=sigma, observed=np.array([0.0]))

    mapper = make_point_mapper(model)
    y_dual = np.array([[0.5], [-0.3]])
    theta = np.exp(y_dual[:, 0])
    # ∇_y log π(θ(y)) = (-θ) · exp(y) = -exp(2y); ∇_θ = -θ
    expected_dual = -np.exp(2.0 * y_dual[:, 0])
    expected_primal = -theta

    point_fn = compile_prior_gradient(model)
    dual_rows = np.stack(
        [np.asarray(point_fn(mapper.unravel(row)), dtype=float) for row in y_dual],
        axis=0,
    )
    np.testing.assert_allclose(dual_rows[:, 0], expected_dual, rtol=1e-5, atol=1e-6)

    flat_fn = compile_flat_prior_grad(mapper, model)
    flat_rows = np.stack([flat_fn(row) for row in y_dual], axis=0)
    np.testing.assert_allclose(flat_rows[:, 0], expected_primal, rtol=1e-5, atol=1e-6)

    batched_fn = compile_batched_prior_grad(mapper, model)
    batched = np.asarray(batched_fn(y_dual), dtype=float)
    np.testing.assert_allclose(batched[:, 0], expected_primal, rtol=1e-5, atol=1e-6)

    _wgf, drift_prior = compile_drift_for_logscore(mapper, model)(y_dual)
    np.testing.assert_allclose(batched, drift_prior, rtol=1e-7, atol=1e-8)
    scaled_dual = dual_rows * mapper.primal_scale(y_dual)
    np.testing.assert_allclose(
        scaled_dual,
        expected_primal[:, None],
        rtol=1e-5,
        atol=1e-6,
    )
    # NumPy wrap contract: dual × primal_scale == particle-facing prior grads
    np.testing.assert_allclose(scaled_dual, batched, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(scaled_dual, drift_prior, rtol=1e-7, atol=1e-8)


def test_identity_drift_primal_scale_is_one():
    """Without transforms, primal_scale ≡ 1 so dual ≡ particle-facing prior."""
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.array([0.0, 0.5]))

    mapper = make_point_mapper(model)
    assert not mapper.has_transforms
    particles = np.array([[0.1], [-0.2], [1.5]])
    np.testing.assert_array_equal(
        mapper.primal_scale(particles), np.ones_like(particles)
    )

    point_fn = compile_prior_gradient(model)
    dual_rows = np.stack(
        [np.asarray(point_fn(mapper.unravel(row)), dtype=float) for row in particles],
        axis=0,
    )
    batched = np.asarray(
        compile_batched_prior_grad(mapper, model)(particles), dtype=float
    )
    _wgf, drift_prior = compile_drift_for_logscore(mapper, model)(particles)

    np.testing.assert_allclose(dual_rows, batched, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(batched, drift_prior, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        dual_rows * mapper.primal_scale(particles),
        drift_prior,
        rtol=1e-10,
        atol=1e-12,
    )


def test_uniform_interval_noise_scale_and_sample_pro_feasible():
    """Interval (Uniform) uses *rv.owner.inputs; noise_scale + sample_pro work."""
    rng = np.random.default_rng(9)
    data = rng.normal(0.5, 0.1, size=20)
    with pm.Model() as model:
        mu = pm.Uniform("mu", lower=0.0, upper=1.0)
        pm.Normal("y", mu=mu, sigma=0.2, observed=data)

    mapper = make_point_mapper(model)
    assert mapper.has_transforms
    base = mapper.ravel(mapper.start_point)
    particles = base[None, :] + np.array([[-1.0], [0.0], [1.5]])
    scale = mapper.noise_scale(particles)
    assert scale.shape == particles.shape
    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0.0)

    dt = sample_pro(
        model,
        n_particles=8,
        n_steps=5,
        tune=2,
        step_size=1e-3,
        learning_rate=1.0,
        random_seed=13,
        include_log_likelihood=False,
        include_sample_stats=False,
    )
    assert "mu" in dt.posterior
    assert "mu_interval__" not in dt.posterior
    vals = dt.posterior["mu"].values
    assert np.all((vals > 0.0) & (vals < 1.0))


def test_sample_pro_halfnormal_feasible_and_named():
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 1.5, size=20)
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=2.0)
        pm.Normal("y", mu=0.0, sigma=sigma, observed=data)

    dt = sample_pro(
        model,
        n_particles=8,
        n_steps=5,
        tune=2,
        step_size=1e-3,
        learning_rate=1.0,
        random_seed=11,
        include_log_likelihood=False,
        include_sample_stats=False,
    )
    assert "sigma" in dt.posterior
    assert "sigma_log__" not in dt.posterior
    vals = dt.posterior["sigma"].values
    assert np.all(vals > 0.0)


def test_sample_pro_beta_feasible():
    rng = np.random.default_rng(3)
    data = rng.binomial(1, 0.7, size=30)
    with pm.Model() as model:
        p = pm.Beta("p", 1.0, 1.0)
        pm.Bernoulli("y", p=p, observed=data)

    dt = sample_pro(
        model,
        n_particles=8,
        n_steps=5,
        tune=2,
        step_size=1e-3,
        random_seed=5,
        include_log_likelihood=False,
        include_sample_stats=False,
    )
    vals = dt.posterior["p"].values
    assert np.all((vals > 0.0) & (vals < 1.0))


def test_compile_prior_gradient_accepts_halfnormal():
    with pm.Model() as model:
        pm.HalfNormal("sigma", sigma=1.0)

    grad_fn = compile_prior_gradient(model)
    mapper = make_point_mapper(model)
    grad = np.asarray(grad_fn(mapper.start_point), dtype=float)
    assert grad.shape == (1,)
    assert np.isfinite(grad).all()
