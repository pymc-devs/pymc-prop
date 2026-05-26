import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc_prop.compile import (
    compile_batched_observed_logp_score,
    compile_batched_prior_grad,
    compile_observed_logp,
    compile_observed_score,
    compile_prior_gradient,
)
from pymc_prop.points import flat_to_value_vars, make_point_mapper


def test_flat_to_value_vars_matches_mapper_unravel():
    with pm.Model() as model:
        pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=0.0, sigma=1.0, observed=np.array([0.1, -0.2]))

    mapper = make_point_mapper(model)
    particles = pt.matrix("particles")
    symbolic_vars = flat_to_value_vars(particles, mapper.point_map_info)
    fn = pytensor.function([particles], symbolic_vars)

    rng = np.random.default_rng(100)
    base = mapper.ravel(mapper.start_point)
    particles_np = base[None, :] + 0.05 * rng.standard_normal((4, base.size))
    outputs = fn(particles_np)

    for row_idx in range(particles_np.shape[0]):
        point = mapper.unravel(particles_np[row_idx])
        for value_var, out in zip(model.value_vars, outputs, strict=True):
            np.testing.assert_allclose(out[row_idx], point[value_var.name], rtol=1e-8, atol=1e-8)


def test_batched_observed_logp_score_matches_loop():
    rng = np.random.default_rng(321)
    y = rng.normal(0.0, 1.0, size=10)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    mapper = make_point_mapper(model)
    logp_fn = compile_observed_logp(model)
    score_fn = compile_observed_score(model)
    batched_fn = compile_batched_observed_logp_score(model, mapper)

    base = mapper.ravel(mapper.start_point)
    particles = base[None, :] + 0.1 * rng.standard_normal((5, base.size))
    logp_batched, score_batched = batched_fn(particles)

    logp_loop = []
    score_loop = []
    for particle in particles:
        point = mapper.unravel(particle)
        logp_loop.append(np.asarray(logp_fn(point), dtype=float).reshape(-1))
        score_loop.append(np.asarray(score_fn(point), dtype=float))

    np.testing.assert_allclose(logp_batched, np.stack(logp_loop, axis=0), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(score_batched, np.stack(score_loop, axis=0), rtol=1e-8, atol=1e-8)


def test_batched_prior_grad_matches_loop():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=2.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=np.array([0.0, 0.4]))

    mapper = make_point_mapper(model)
    single_grad_fn = compile_prior_gradient(model)
    batched_grad_fn = compile_batched_prior_grad(mapper, model)

    rng = np.random.default_rng(456)
    base = mapper.ravel(mapper.start_point)
    particles = base[None, :] + 0.05 * rng.standard_normal((6, base.size))

    batched = np.asarray(batched_grad_fn(particles), dtype=float)
    looped = np.stack([np.asarray(single_grad_fn(mapper.unravel(p)), dtype=float) for p in particles], axis=0)

    np.testing.assert_allclose(batched, looped, rtol=1e-7, atol=1e-8)
