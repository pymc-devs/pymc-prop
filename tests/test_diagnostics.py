import numpy as np
import pymc as pm
import pytest

import pymc_prop.sampler as sampler_module
from pymc_prop.diagnostics import compute_kgd, get_bandwidth, imq_kernel, rbf_kernel
from pymc_prop.sample import sample_pro
from pymc_prop.points import make_point_mapper
from pymc_prop.scoring import LogScore, ScoringRule


def test_imq_kernel_components():
    sq_dists = np.array([[0.0, 2.0], [2.0, 0.0]])
    dimension = 3
    bandwidth = 1.5

    kernel, cross_coeff, trace_term = imq_kernel(
        sq_dists, dimension, bandwidth
    )
    expected_kernel = (1.0 + sq_dists / bandwidth) ** -0.5

    np.testing.assert_allclose(kernel, expected_kernel)
    np.testing.assert_allclose(
        cross_coeff, expected_kernel**3 / bandwidth
    )
    np.testing.assert_allclose(
        trace_term,
        dimension * expected_kernel**3 / bandwidth
        - 3.0 * sq_dists * expected_kernel**5 / bandwidth**2,
    )


def test_get_bandwidth_median_heuristic():
    particles = np.array([[0.0], [1.0], [3.0]])
    diffs = particles[:, None, :] - particles[None, :, :]
    sq_dists = np.sum(diffs**2, axis=-1)

    expected = np.median([1.0, 9.0, 4.0]) / np.log(3)
    assert np.isclose(get_bandwidth(sq_dists), expected)
    assert get_bandwidth(sq_dists, bandwidth=2.5) == 2.5


def test_get_bandwidth_rejects_invalid_explicit_value():
    sq_dists = np.zeros((2, 2))

    for bandwidth in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            get_bandwidth(sq_dists, bandwidth=bandwidth)


def test_compute_kgd_matches_sqrt_direct_v_statistic():
    particles = np.array([[-1.0], [0.5], [2.0]])
    score = np.array([[0.25], [-0.75], [1.5]])
    bandwidth = 1.7

    stein_kernel = np.empty((len(particles), len(particles)))
    for i, (x_i,) in enumerate(particles):
        for j, (x_j,) in enumerate(particles):
            difference = x_i - x_j
            squared_difference = difference**2
            kernel = (1.0 + squared_difference / bandwidth) ** -0.5
            cross_terms = (
                (score[i, 0] - score[j, 0])
                * difference
                * kernel**3
                / bandwidth
            )
            derivative_trace = (
                kernel**3 / bandwidth
                - 3.0 * squared_difference * kernel**5 / bandwidth**2
            )
            stein_kernel[i, j] = (
                score[i, 0] * score[j, 0] * kernel
                + cross_terms
                + derivative_trace
            )

    actual = compute_kgd(particles, score, bandwidth=bandwidth)
    np.testing.assert_allclose(actual, np.sqrt(np.mean(stein_kernel)))


def test_compute_kgd_clips_negative_roundoff_before_sqrt():
    particles = np.array([[0.0], [1.0]])
    score = np.zeros_like(particles)

    def roundoff_kernel(sq_dists, dimension, bandwidth):
        shape = sq_dists.shape
        return np.zeros(shape), np.zeros(shape), np.full(shape, -1e-16)

    assert (
        compute_kgd(
            particles,
            score,
            kernel_fn=roundoff_kernel,
            bandwidth=1.0,
        )
        == 0.0
    )


def test_normal_target_has_lower_kgd_than_wrong_score():
    rng = np.random.default_rng(91)
    particles = rng.normal(size=(500, 1))
    sq_dists = (particles[:, None, 0] - particles[None, :, 0]) ** 2
    bandwidth = get_bandwidth(sq_dists)

    target_score = -particles
    shifted_score = -(particles - 2.0)

    target_kgd = compute_kgd(
        particles, target_score, bandwidth=bandwidth
    )
    shifted_kgd = compute_kgd(
        particles, shifted_score, bandwidth=bandwidth
    )

    assert target_kgd < shifted_kgd


def test_sample_pro_validates_kgd_options():
    with pm.Model() as model:
        location = pm.Normal("location", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=location, sigma=1.0, observed=np.array([0.0]))

    cases = (
        ({"kgd_interval": 0}, "positive integer"),
        ({"kgd_interval": True}, "positive integer"),
        (
            {"kgd_interval": 1, "include_sample_stats": False},
            "require sample_stats",
        ),
        ({"kgd_bandwidth": 1.0}, "requires kgd_interval"),
        (
            {"kgd_interval": 1, "kgd_bandwidth": 0.0},
            "finite and positive",
        ),
        ({"kgd_kernel": rbf_kernel}, "requires kgd_interval"),
        (
            {"kgd_interval": 1, "kgd_kernel": "not_callable"},
            "callable",
        ),
    )

    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            sample_pro(model=model, **kwargs)


def test_sample_pro_records_kgd_at_requested_interval():
    with pm.Model() as model:
        location = pm.Normal("location", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=location, sigma=1.0, observed=np.array([0.0]))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=4,
        tune=1,
        step_size=1e-3,
        random_seed=42,
        include_log_likelihood=False,
        kgd_interval=2,
        kgd_bandwidth=1.0,
    )

    kgd = dt.sample_stats["kgd"]
    first_chain = kgd.isel(chain=0).values

    assert np.all(np.isfinite(first_chain[[0, 2]]))
    assert np.all(np.isnan(first_chain[[1, 3]]))
    np.testing.assert_allclose(
        kgd.values,
        np.broadcast_to(first_chain[:, None], kgd.shape),
        equal_nan=True,
    )
    np.testing.assert_array_equal(kgd.coords["draw"], np.arange(4))
    np.testing.assert_array_equal(kgd.coords["step"], np.arange(1, 5))


def test_kgd_disabled_skips_bandwidth_computation(monkeypatch):
    with pm.Model() as model:
        location = pm.Normal("location", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=location, sigma=1.0, observed=np.array([0.0]))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("get_bandwidth should not run when KGD is disabled")

    monkeypatch.setattr(sampler_module, "get_bandwidth", fail_if_called)

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=4,
        tune=0,
        step_size=1e-3,
        random_seed=42,
        include_log_likelihood=False,
    )

    assert "kgd" not in dt.sample_stats


def test_sample_pro_records_kgd_with_custom_kernel():
    with pm.Model() as model:
        location = pm.Normal("location", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=location, sigma=1.0, observed=np.array([0.0]))

    dt_imq = sample_pro(
        model=model,
        n_particles=4,
        n_steps=4,
        tune=1,
        step_size=1e-3,
        random_seed=42,
        include_log_likelihood=False,
        kgd_interval=1,
        kgd_bandwidth=1.0,
        kgd_kernel=imq_kernel,
    )

    dt_rbf = sample_pro(
        model=model,
        n_particles=4,
        n_steps=4,
        tune=1,
        step_size=1e-3,
        random_seed=42,
        include_log_likelihood=False,
        kgd_interval=1,
        kgd_bandwidth=1.0,
        kgd_kernel=rbf_kernel,
    )

    kgd_imq = dt_imq.sample_stats["kgd"].isel(chain=0).values
    kgd_rbf = dt_rbf.sample_stats["kgd"].isel(chain=0).values

    assert np.all(np.isfinite(kgd_imq))
    assert np.all(np.isfinite(kgd_rbf))
    assert not np.allclose(kgd_imq, kgd_rbf)


def _mixed_transform_model():
    """Identity, log and logodds slabs, with ``loc`` first to exercise offsets."""
    with pm.Model() as model:
        loc = pm.Normal("loc", mu=0.0, sigma=1.0)
        scale = pm.Gamma("scale", alpha=5.0, beta=1.0)
        prob = pm.Beta("prob", alpha=2.0, beta=3.0)
        pm.Normal("y", mu=loc, sigma=1.0, observed=np.array([0.0, 0.5]))
    return model


def _prior_score(posterior_draw):
    """Constrained-space prior score of the mixed model (jacobian-free potential)."""
    loc = posterior_draw["loc"]
    scale = posterior_draw["scale"]
    prob = posterior_draw["prob"]
    return np.stack(
        [-loc, 4.0 / scale - 1.0, 1.0 / prob - 2.0 / (1.0 - prob)], axis=1
    )


def test_kgd_uses_constrained_particles_and_score():
    """Check that KGD is computed using constrained-space particles and prior score."""
    model = _mixed_transform_model()
    bandwidth = 3.0

    dt = sample_pro(
        model=model,
        n_particles=16,
        n_steps=5,
        tune=2,
        step_size=1e-3,
        learning_rate=1e-10,
        random_seed=3,
        include_log_likelihood=False,
        kgd_interval=1,
        kgd_bandwidth=bandwidth,
    )

    posterior = dt.posterior
    assert posterior["scale"].min() > 0
    for draw in range(posterior.sizes["draw"]):
        constrained = {
            name: posterior[name].isel(draw=draw).values
            for name in ("loc", "scale", "prob")
        }
        particles = np.stack(
            [constrained["loc"], constrained["scale"], constrained["prob"]], axis=1
        )
        expected = compute_kgd(
            particles, _prior_score(constrained), bandwidth=bandwidth
        )
        actual = dt.sample_stats["kgd"].isel(chain=0, draw=draw).item()
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


@pytest.mark.parametrize("step_size", [1e-3, None])
def test_kgd_does_not_change_particle_trajectories(step_size):
    model = _mixed_transform_model()
    kwargs = dict(
        model=model,
        n_particles=8,
        n_steps=6,
        tune=3,
        step_size=step_size,
        random_seed=11,
        include_log_likelihood=False,
    )

    baseline = sample_pro(**kwargs)
    with_kgd = sample_pro(kgd_interval=2, **kwargs)

    for name in baseline.posterior.data_vars:
        np.testing.assert_array_equal(
            with_kgd.posterior[name].values, baseline.posterior[name].values
        )
    if step_size is None:
        np.testing.assert_array_equal(
            with_kgd.sample_stats["fuse_step_size"].values,
            baseline.sample_stats["fuse_step_size"].values,
        )


def test_kgd_matches_between_drift_and_wgf_code_paths():
    class WgfOnly(ScoringRule):
        def compile_wgf(self, model, mapper):
            return LogScore().compile_wgf(model, mapper)

    model = _mixed_transform_model()
    kwargs = dict(
        model=model,
        n_particles=8,
        n_steps=5,
        tune=2,
        step_size=1e-3,
        random_seed=13,
        include_log_likelihood=False,
        kgd_interval=2,
    )

    fused = sample_pro(**kwargs)
    split = sample_pro(scoring_rule=WgfOnly(), **kwargs)

    for name in fused.posterior.data_vars:
        np.testing.assert_allclose(
            split.posterior[name].values, fused.posterior[name].values
        )
    np.testing.assert_allclose(
        split.sample_stats["kgd"].values,
        fused.sample_stats["kgd"].values,
        equal_nan=True,
    )


def test_point_mapper_constrain_matches_posterior_layout():
    model = _mixed_transform_model()
    mapper = make_point_mapper(model)
    rng = np.random.default_rng(0)
    flat = rng.normal(size=(5, 3))

    constrained = mapper.constrain(flat)

    np.testing.assert_allclose(constrained[:, 0], flat[:, 0])
    np.testing.assert_allclose(constrained[:, 1], np.exp(flat[:, 1]))
    np.testing.assert_allclose(constrained[:, 2], 1.0 / (1.0 + np.exp(-flat[:, 2])))
