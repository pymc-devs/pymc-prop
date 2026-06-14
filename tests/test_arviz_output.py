import numpy as np
import pymc as pm
import pytest
from arviz_base import extract
from scipy.special import logsumexp
from xarray import DataTree

from pymc_prop.arviz import _pro_to_datatree
from pymc_prop.points import make_point_mapper
from pymc_prop.sample import sample_pro
from pymc_prop.sampler import run_sampler
from pymc_prop.scoring import LogScore


def test_sample_pro_returns_datatree_with_posterior():
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=20)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)

    n_steps = 60
    n_particles = 8

    dt = sample_pro(
        model=model,
        n_particles=n_particles,
        n_steps=n_steps,
        tune=10,
        step_size=5e-3,
        random_seed=123,
    )

    assert isinstance(dt, DataTree)
    assert "posterior" in dt
    assert "mu" in dt.posterior
    assert set(dt.posterior.dims) >= {"chain", "draw"}
    assert dt.posterior.attrs["sample_dims"] == ["draw", "chain"]
    assert dt.posterior.attrs["inference_library"] == "pymc"
    assert dt.posterior.sizes["draw"] == n_steps
    assert dt.posterior.sizes["chain"] == n_particles
    assert np.all(np.isfinite(dt.posterior["mu"].values))


def test_datatree_includes_observed_log_likelihood_and_sample_stats():
    y = np.array([0.1, -0.2, 0.3])

    with pm.Model(coords={"obs": ["a", "b", "c"]}) as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y, dims="obs")

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=20,
        tune=0,
        step_size=5e-3,
        random_seed=0,
    )

    assert "observed_data" in dt
    assert "y" in dt.observed_data
    np.testing.assert_allclose(dt.observed_data["y"].values, y)
    assert "obs" in dt.observed_data["y"].dims

    assert "log_likelihood" in dt
    assert "y" in dt.log_likelihood
    assert set(dt.log_likelihood["y"].dims) >= {"chain", "draw", "obs"}

    assert "mixture_log_predictive" in dt
    assert "y" in dt.mixture_log_predictive
    assert "chain" not in dt.mixture_log_predictive["y"].dims
    assert set(dt.mixture_log_predictive["y"].dims) >= {"draw", "obs"}

    assert "sample_stats" in dt
    for var in (
        "particle_spread",
        "mean_log_score",
        "se_log_score",
        "mixture_log_predictive_total",
        "learning_rate",
        "mu_spread",
    ):
        assert var in dt.sample_stats


def test_posterior_matches_point_mapper_unravel():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(5))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=10,
        tune=0,
        step_size=5e-3,
        random_seed=7,
    )

    mapper = make_point_mapper(model)
    final_cloud = dt.posterior.isel(draw=-1)["mu"].values
    for chain_idx in range(final_cloud.shape[0]):
        point = mapper.unravel(np.asarray([final_cloud[chain_idx]], dtype=float))
        np.testing.assert_allclose(point["mu"], final_cloud[chain_idx], rtol=1e-8, atol=1e-8)


def test_extract_posterior_group():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(3))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=8,
        tune=0,
        random_seed=1,
    )

    extracted = extract(dt, group="posterior", keep_dataset=True)
    assert "mu" in extracted.data_vars


def test_pro_draw_and_step_coords():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(3))

    n_steps = 12
    tune = 2

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=n_steps,
        tune=tune,
        learning_rate=0.5,
        random_seed=4,
    )

    np.testing.assert_array_equal(dt.posterior.coords["draw"].values, np.arange(n_steps))
    np.testing.assert_array_equal(dt.posterior.coords["step"].values, np.arange(tune, tune + n_steps))


def test_n_steps_retained_even_when_tune_is_large():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(3))

    n_steps = 10
    tune = 20

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=n_steps,
        tune=tune,
        random_seed=0,
    )

    assert dt.posterior.sizes["draw"] == n_steps
    np.testing.assert_array_equal(dt.posterior.coords["step"].values, np.arange(tune, tune + n_steps))
    assert "log_likelihood" in dt
    assert "sample_stats" in dt


def test_tune_validation_raises_on_negative():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(3))

    with pytest.raises(ValueError, match="tune must be non-negative"):
        sample_pro(model=model, n_particles=4, n_steps=10, tune=-1)


def test_include_log_likelihood_false_skips_group():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(5))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=8,
        tune=0,
        include_log_likelihood=False,
        random_seed=2,
    )

    assert "posterior" in dt
    assert "log_likelihood" not in dt
    assert "mixture_log_predictive" not in dt
    assert "mean_log_score" not in dt.sample_stats
    assert "se_log_score" not in dt.sample_stats
    assert "mixture_log_predictive_total" not in dt.sample_stats


def test_multi_observed_rv_log_likelihood():
    y1 = np.array([0.1, -0.2])
    y2 = np.array([0.3])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y1", mu=mu, sigma=1.0, observed=y1)
        pm.Normal("y2", mu=mu, sigma=1.0, observed=y2)

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=10,
        tune=0,
        random_seed=3,
    )

    assert "y1" in dt.log_likelihood
    assert "y2" in dt.log_likelihood
    assert dt.log_likelihood["y1"].shape[-1] == 2
    assert dt.log_likelihood["y2"].shape[-1] == 1


def test_pro_to_datatree_direct_from_sampler():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(4))

    mapper = make_point_mapper(model)
    particles = run_sampler(
        model=model,
        mapper=mapper,
        scoring_rule=LogScore(),
        n_particles=4,
        n_steps=8,
        tune=0,
        step_size=5e-3,
        learning_rate=1.0,
        random_seed=5,
    )

    dt = _pro_to_datatree(
        particles,
        model=model,
        mapper=mapper,
        tune=0,
        learning_rate=1.0,
    )

    assert dt.posterior.sizes["draw"] == particles.shape[0]

    per_particle = np.sum(dt.log_likelihood["y"].values, axis=-1)
    expected_se = np.std(per_particle, axis=1, ddof=1) / np.sqrt(4)
    np.testing.assert_allclose(dt.sample_stats["se_log_score"].isel(chain=0).values, expected_se, rtol=1e-8)


def test_datatree_kwargs_merges_coords_without_losing_draw():
    with pm.Model(coords={"obs": ["a", "b"]}) as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=[0.0, 1.0], dims="obs")

    mapper = make_point_mapper(model)
    particles = np.zeros((2, 3, mapper.ravel(model.initial_point()).size))

    dt = _pro_to_datatree(
        particles,
        model=model,
        mapper=mapper,
        tune=0,
        learning_rate=1.0,
        datatree_kwargs={
            "coords": {"obs": ["x", "y"], "draw": [99, 100], "step": [99, 100]},
            "sample_dims": ["draw", "chain"],
        },
    )

    np.testing.assert_array_equal(dt.posterior.coords["draw"].values, [0, 1])
    np.testing.assert_array_equal(dt.posterior.coords["step"].values, [0, 1])
    assert dt.posterior.attrs["sample_dims"] == ["draw", "chain"]
    np.testing.assert_array_equal(dt.observed_data.coords["obs"].values, ["x", "y"])


def test_mixture_log_predictive_gaussian_closed_form():
    """Mixture log predictive matches logsumexp over particles minus log p."""
    y = np.array([0.5, -1.0])
    sigma = 1.0

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    mapper = make_point_mapper(model)
    n_particles = 2
    flat_parts = [mapper.ravel({"mu": np.array([mu_val])}) for mu_val in (-2.0, 3.0)]
    flat = np.stack(flat_parts, axis=0)
    particles = flat[None, :, :]

    dt = _pro_to_datatree(
        particles,
        model=model,
        mapper=mapper,
        tune=0,
        learning_rate=1.0,
    )

    ll = dt.log_likelihood["y"].values
    expected = logsumexp(ll, axis=1) - np.log(n_particles)
    np.testing.assert_allclose(dt.mixture_log_predictive["y"].values, expected, rtol=1e-10)


def test_mixture_log_predictive_total_matches_sum():
    with pm.Model(coords={"obs": ["a", "b", "c"]}) as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=[0.1, -0.2, 0.3], dims="obs")

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=5,
        tune=0,
        step_size=5e-3,
        include_posterior_predictive=False,
        random_seed=11,
    )

    expected_total = dt.mixture_log_predictive["y"].values.sum(axis=-1)
    actual_total = dt.sample_stats["mixture_log_predictive_total"].isel(chain=0).values
    np.testing.assert_allclose(actual_total, expected_total, rtol=1e-8)


def test_mixture_log_predictive_not_equal_mean_log_score():
    """Per-particle mean log score differs from mixture total when particles diverge."""
    y = np.array([0.5, -1.0])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)

    mapper = make_point_mapper(model)
    flat_parts = [mapper.ravel({"mu": np.array([mu_val])}) for mu_val in (-2.0, 3.0)]
    flat = np.stack(flat_parts, axis=0)
    particles = flat[None, :, :]

    dt = _pro_to_datatree(
        particles,
        model=model,
        mapper=mapper,
        tune=0,
        learning_rate=1.0,
    )

    mean_score = dt.sample_stats["mean_log_score"].isel(chain=0, draw=0).values
    mixture_total = dt.sample_stats["mixture_log_predictive_total"].isel(chain=0, draw=0).values
    assert not np.isclose(mean_score, mixture_total)


def test_mixture_log_predictive_dims():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(4))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=6,
        tune=2,
        include_posterior_predictive=False,
        random_seed=6,
    )

    assert "chain" not in dt.mixture_log_predictive["y"].dims
    assert dt.mixture_log_predictive.sizes["draw"] == dt.posterior.sizes["draw"]
    assert dt.mixture_log_predictive.attrs["sample_dims"] == ["draw"]
    np.testing.assert_array_equal(
        dt.mixture_log_predictive.coords["draw"].values,
        dt.posterior.coords["draw"].values,
    )


def test_multi_observed_rv_mixture_log_predictive():
    y1 = np.array([0.1, -0.2])
    y2 = np.array([0.3])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y1", mu=mu, sigma=1.0, observed=y1)
        pm.Normal("y2", mu=mu, sigma=1.0, observed=y2)

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=10,
        tune=0,
        include_posterior_predictive=False,
        random_seed=3,
    )

    assert "y1" in dt.mixture_log_predictive
    assert "y2" in dt.mixture_log_predictive
    expected_total = (
        dt.mixture_log_predictive["y1"].values.sum()
        + dt.mixture_log_predictive["y2"].values.sum()
    )
    actual_total = dt.sample_stats["mixture_log_predictive_total"].isel(chain=0).values.sum()
    np.testing.assert_allclose(actual_total, expected_total, rtol=1e-8)
