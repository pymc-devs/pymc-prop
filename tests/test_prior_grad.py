import numpy as np
import pymc as pm
import pytest

from pymc_prop.compile import compile_prior_gradient
from pymc_prop.points import make_point_mapper


def _central_diff_prior(logp_fn, mapper, base, index, eps=1e-5):
    up = base.copy()
    dn = base.copy()
    up[index] += eps
    dn[index] -= eps
    pup = logp_fn(mapper.unravel(up))
    pdn = logp_fn(mapper.unravel(dn))
    return (float(pup) - float(pdn)) / (2.0 * eps)


def test_prior_gradient_gaussian_closed_form():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=2.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.array([0.2, -0.1]))

    mapper = make_point_mapper(model)
    grad_fn = compile_prior_gradient(model)
    point = mapper.start_point
    grad = grad_fn(point)

    mu = float(point["mu"])
    expected = np.array([-mu / 4.0])
    np.testing.assert_allclose(grad, expected, rtol=1e-6, atol=1e-8)


def test_prior_gradient_halfnormal_transform():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=np.array([0.0, 0.5]))

    mapper = make_point_mapper(model)
    grad_fn = compile_prior_gradient(model)
    logp_fn = model.compile_logp(vars=model.free_RVs, jacobian=True)

    base = mapper.ravel(mapper.start_point)
    grad = grad_fn(mapper.start_point)

    fd = np.array([_central_diff_prior(logp_fn, mapper, base, i) for i in range(base.size)])
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-5)


def test_prior_gradient_rejects_discrete():
    with pm.Model() as model:
        pm.Categorical("k", p=np.array([0.5, 0.5]))

    with pytest.raises(ValueError, match="continuous"):
        compile_prior_gradient(model)
