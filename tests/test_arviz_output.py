import numpy as np
import pymc as pm
from arviz_base import extract
from xarray import DataTree

from pymc_prop.arviz import pro_to_datatree
from pymc_prop.points import make_point_mapper
from pymc_prop.sample import sample_pro
from pymc_prop.sampler import run_sampler
from pymc_prop.scoring import LogScore


def _retained_count(n_steps: int, burn_in: int, thinning: int) -> int:
    return sum(
        1
        for step in range(n_steps)
        if step >= burn_in and (step - burn_in) % thinning == 0
    )


def test_sample_pro_returns_datatree_with_posterior():
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=20)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)

    dt = sample_pro(
        model=model,
        n_particles=8,
        n_steps=60,
        burn_in=10,
        thinning=5,
        step_size=5e-3,
        random_seed=123,
    )

    assert isinstance(dt, DataTree)
    assert "posterior" in dt
    assert "mu" in dt.posterior
    assert set(dt.posterior.dims) >= {"chain", "draw"}
    assert dt.posterior.attrs["sample_dims"] == ["draw", "chain"]
    assert dt.posterior.attrs["inference_library"] == "pymc"

    n_retained = _retained_count(60, 10, 5)
    assert dt.posterior.sizes["draw"] == n_retained
    assert dt.posterior.sizes["chain"] == 8
    assert dt.posterior["mu"].shape == (n_retained, 8)
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
        burn_in=0,
        thinning=2,
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

    assert "sample_stats" in dt
    for var in ("particle_spread", "mean_log_score", "learning_rate", "mu_spread"):
        assert var in dt.sample_stats


def test_posterior_matches_point_mapper_unravel():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(5))

    mapper = make_point_mapper(model)
    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=10,
        burn_in=0,
        thinning=1,
        step_size=5e-3,
        random_seed=7,
    )

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
        burn_in=0,
        thinning=1,
        random_seed=1,
    )

    extracted = extract(dt, group="posterior", keep_dataset=True)
    assert "mu" in extracted.data_vars


def test_pro_step_coord_and_root_attrs():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(3))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=12,
        burn_in=2,
        thinning=3,
        learning_rate=0.5,
        random_seed=4,
    )

    expected_steps = np.array([2, 5, 8, 11])
    np.testing.assert_array_equal(dt.posterior.coords["draw"].values, expected_steps)
    assert dt.attrs["pro_burn_in"] == 2
    assert dt.attrs["pro_thinning"] == 3
    assert dt.attrs["pro_n_steps"] == 12
    assert dt.attrs["pro_learning_rate"] == 0.5


def test_empty_retention_when_burn_in_exceeds_n_steps():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(3))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=10,
        burn_in=20,
        random_seed=0,
    )

    assert dt.posterior.sizes["draw"] == 0
    assert dt.posterior.sizes["chain"] == 4
    assert dt.posterior["mu"].shape == (0, 4)
    assert "log_likelihood" not in dt
    assert "sample_stats" not in dt
    assert "observed_data" in dt


def test_include_log_likelihood_false_skips_group():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(5))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=8,
        burn_in=0,
        thinning=1,
        include_log_likelihood=False,
        random_seed=2,
    )

    assert "posterior" in dt
    assert "log_likelihood" not in dt
    assert "mean_log_score" not in dt.sample_stats


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
        burn_in=0,
        thinning=1,
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
        burn_in=0,
        thinning=2,
        step_size=5e-3,
        learning_rate=1.0,
        random_seed=5,
    )

    dt = pro_to_datatree(
        particles,
        model=model,
        mapper=mapper,
        burn_in=0,
        thinning=2,
        n_steps=8,
        learning_rate=1.0,
    )

    assert dt.posterior.sizes["draw"] == particles.shape[0]
    assert dt.posterior.sizes["chain"] == 4


def test_datatree_kwargs_merges_coords_without_losing_draw():
    with pm.Model(coords={"obs": ["a", "b"]}) as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=[0.0, 1.0], dims="obs")

    mapper = make_point_mapper(model)
    particles = np.zeros((2, 3, mapper.ravel(model.initial_point()).size))

    dt = pro_to_datatree(
        particles,
        model=model,
        mapper=mapper,
        burn_in=0,
        thinning=1,
        n_steps=2,
        learning_rate=1.0,
        datatree_kwargs={
            "coords": {"obs": ["x", "y"]},
            "sample_dims": ["draw", "chain"],
        },
    )

    np.testing.assert_array_equal(dt.posterior.coords["draw"].values, [0, 1])
    assert dt.posterior.attrs["sample_dims"] == ["draw", "chain"]
    np.testing.assert_array_equal(dt.observed_data.coords["obs"].values, ["x", "y"])
