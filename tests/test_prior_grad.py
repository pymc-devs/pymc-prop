import numpy as np
import pymc as pm
import pytest

from pymc_prop.compile import compile_prior_gradient
from pymc_prop.points import make_point_mapper


def test_prior_gradient_gaussian_closed_form():
    # ∇_μ log π(μ) for μ ~ N(0, 4): should be -μ/4 (free RVs only, no likelihood)
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


def test_prior_gradient_rejects_halfnormal():
    # implicit HalfNormal transform → require_unconstrained_free_rvs rejects
    with pm.Model() as model:
        pm.HalfNormal("sigma", sigma=1.0)

    with pytest.raises(ValueError, match="native unconstrained"):
        compile_prior_gradient(model)


def test_prior_gradient_rejects_discrete():
    # log-score / prior grad path needs continuous value_vars
    with pm.Model() as model:
        pm.Categorical("k", p=np.array([0.5, 0.5]))

    with pytest.raises(ValueError, match="continuous"):
        compile_prior_gradient(model)
