import numpy as np
import pymc as pm
import pytest
from pymc.exceptions import SamplingError

from pymc_prop.points import make_point_mapper
from pymc_prop.particles import _init_prior, initialize_particles
from pymc_prop.sample import sample_pro
from pymc_prop.sampler import run_sampler
from pymc_prop.scoring import LogScore


def test_sample_pro_runs_gaussian():
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=20)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)

    n_steps = 60
    dt = sample_pro(
        model=model,
        n_particles=8,
        n_steps=n_steps,
        tune=10,
        step_size=5e-3,
        random_seed=123,
    )

    assert "mu" in dt.posterior
    assert dt.posterior.sizes["draw"] == n_steps
    assert dt.posterior.sizes["chain"] == 8
    assert np.all(np.isfinite(dt.posterior["mu"].values))


def test_prior_init_particles_are_non_degenerate():
    rng = np.random.default_rng(0)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(5))

    mapper = make_point_mapper(model)
    particles = initialize_particles(model, mapper, n_particles=8, rng=rng)

    assert particles.shape == (8, 1)
    spread = np.linalg.norm(particles - particles.mean(axis=0), axis=1)
    assert np.all(spread > 0)


def test_init_prior_retries_invalid_start():
    with pm.Model() as model:
        pm.HalfNormal("x", default_transform=None, initval=0)

    mapper = make_point_mapper(model)
    calls = {"n": 0}

    def logp_fn(point):
        calls["n"] += 1
        return -np.inf if calls["n"] < 3 else 0.0

    particles = _init_prior(
        model,
        mapper,
        n_particles=1,
        seeds=np.array([1]),
        max_retries=10,
        logp_fn=logp_fn,
    )
    assert particles.shape == (1, 1)
    assert np.isfinite(particles).all()
    assert calls["n"] == 3


def test_init_prior_check_start_vals_on_exhaustion(monkeypatch):
    with pm.Model() as model:
        pm.HalfNormal("x", default_transform=None, initval=0)

    mapper = make_point_mapper(model)
    calls = []

    def fake_check_start_vals(point):
        calls.append(point)
        raise SamplingError("bad start")

    monkeypatch.setattr(model, "check_start_vals", fake_check_start_vals)

    with pytest.raises(SamplingError, match="bad start"):
        _init_prior(
            model,
            mapper,
            n_particles=1,
            seeds=np.array([1]),
            max_retries=2,
            logp_fn=lambda point: -np.inf,
        )

    assert len(calls) == 1


def test_run_sampler_rejects_single_particle():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(5))

    mapper = make_point_mapper(model)

    with pytest.raises(ValueError, match="n_particles must be at least 2"):
        run_sampler(
            model=model,
            mapper=mapper,
            scoring_rule=LogScore(),
            n_particles=1,
            n_steps=4,
            tune=0,
            step_size=1e-3,
            learning_rate=1.0,
            random_seed=0,
        )
