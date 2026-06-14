import numpy as np
import pymc as pm
import pytest

from pymc_prop import sample_posterior_predictive_pro, sample_pro


def _gaussian_model(y=None):
    if y is None:
        y = np.zeros(5)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)
    return model


def test_sample_pro_includes_posterior_predictive_by_default():
    model = _gaussian_model()
    n_particles = 4

    dt = sample_pro(
        model=model,
        n_particles=n_particles,
        n_steps=8,
        tune=0,
        random_seed=0,
    )

    assert "posterior_predictive" in dt
    assert dt.posterior_predictive.attrs["sample_dims"] == ["draw", "chain"]
    assert dt.posterior_predictive.sizes["draw"] == 1
    assert dt.posterior_predictive.sizes["chain"] == n_particles
    assert np.all(np.isfinite(dt.posterior_predictive["y"].values))


def test_include_posterior_predictive_false():
    model = _gaussian_model()

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=8,
        tune=0,
        include_posterior_predictive=False,
        random_seed=1,
    )

    assert "posterior_predictive" not in dt


def test_final_draw_shape():
    model = _gaussian_model()
    n_particles = 6
    n_obs = 5

    dt = sample_pro(
        model=model,
        n_particles=n_particles,
        n_steps=12,
        tune=0,
        include_posterior_predictive=False,
        random_seed=2,
    )

    sample_posterior_predictive_pro(dt, model=model, draw=-1)
    assert dt.posterior_predictive["y"].shape == (1, n_particles, n_obs)


def test_thinned_draw_shape():
    model = _gaussian_model()
    n_particles = 4
    n_steps = 10

    dt = sample_pro(
        model=model,
        n_particles=n_particles,
        n_steps=n_steps,
        tune=0,
        include_posterior_predictive=False,
        random_seed=3,
    )

    sample_posterior_predictive_pro(dt, model=model, draw=slice(0, None, 2))
    assert dt.posterior_predictive.sizes["draw"] == n_steps // 2
    assert dt.posterior_predictive.sizes["chain"] == n_particles


def test_plot_ppc_dist_smoke():
    import arviz as az

    model = _gaussian_model()
    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=8,
        tune=0,
        random_seed=4,
    )
    fig = az.plot_ppc_dist(dt)
    assert fig is not None


def test_oos_predictions_linear_regression():
    x_train = np.linspace(0.0, 1.0, 5)
    x_test = np.linspace(1.1, 2.0, 3)

    with pm.Model(coords={"trial": np.arange(5)}) as model:
        x = pm.Data("x", x_train, dims="trial")
        alpha = pm.Normal("alpha", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=0.5, observed=np.zeros(5), dims="trial")

        dt = sample_pro(
            model=model,
            n_particles=4,
            n_steps=6,
            tune=0,
            include_posterior_predictive=False,
            random_seed=5,
        )
        sample_posterior_predictive_pro(
            dt,
            model=model,
            predictions=True,
            data={"x": x_test},
            coords={"trial": np.arange(3)},
        )

    assert "predictions" in dt
    assert dt.predictions["y"].shape == (1, 4, 3)
    assert np.all(np.isfinite(dt.predictions["y"].values))


def test_oos_does_not_mutate_posterior():
    x_train = np.linspace(0.0, 1.0, 4)
    x_test = np.array([2.0, 2.5])

    with pm.Model(coords={"trial": np.arange(4)}) as model:
        x = pm.Data("x", x_train, dims="trial")
        alpha = pm.Normal("alpha", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=0.5, observed=np.zeros(4), dims="trial")

        dt = sample_pro(
            model=model,
            n_particles=4,
            n_steps=5,
            tune=0,
            include_posterior_predictive=False,
            random_seed=6,
        )
        posterior_before = {name: dt.posterior[name].values.copy() for name in dt.posterior.data_vars}
        sample_posterior_predictive_pro(
            dt,
            model=model,
            predictions=True,
            data={"x": x_test},
            coords={"trial": np.arange(2)},
        )

    for name, values in posterior_before.items():
        np.testing.assert_array_equal(dt.posterior[name].values, values)


def test_both_groups_separate_calls():
    x_train = np.linspace(0.0, 1.0, 4)
    x_test = np.array([2.0])

    with pm.Model(coords={"trial": np.arange(4)}) as model:
        x = pm.Data("x", x_train, dims="trial")
        alpha = pm.Normal("alpha", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=0.5, observed=np.zeros(4), dims="trial")

        dt = sample_pro(
            model=model,
            n_particles=4,
            n_steps=5,
            tune=0,
            include_posterior_predictive=False,
            random_seed=7,
        )
        sample_posterior_predictive_pro(dt, model=model, draw=-1)
        sample_posterior_predictive_pro(
            dt,
            model=model,
            predictions=True,
            data={"x": x_test},
            coords={"trial": np.arange(1)},
        )

    assert "posterior_predictive" in dt
    assert "predictions" in dt


def test_pymc_default_sample_dims_differs_on_multi_draw():
    """Wrong flatten order swaps draw/chain when attrs are ignored."""
    import copy

    model = _gaussian_model()
    n_particles = 4
    n_steps = 6

    dt = sample_pro(
        model=model,
        n_particles=n_particles,
        n_steps=n_steps,
        tune=0,
        include_posterior_predictive=False,
        random_seed=8,
    )

    dt_bare = copy.deepcopy(dt)
    dt_bare.posterior.attrs.pop("sample_dims", None)

    with model:
        wrong = pm.sample_posterior_predictive(
            dt_bare,
            var_names=["y"],
            extend_inferencedata=True,
            random_seed=9,
        )
        native = sample_posterior_predictive_pro(
            dt,
            model=model,
            draw=slice(None),
            random_seed=9,
        )

    assert wrong.posterior_predictive["y"].shape == (n_particles, n_steps, 5)
    assert native.posterior_predictive["y"].shape == (n_steps, n_particles, 5)
    assert native.posterior_predictive.attrs["sample_dims"] == ["draw", "chain"]


def test_predictions_true_requires_data():
    model = _gaussian_model()
    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=4,
        tune=0,
        include_posterior_predictive=False,
        random_seed=10,
    )

    with pytest.raises(ValueError, match="predictions=True requires data"):
        sample_posterior_predictive_pro(dt, model=model, predictions=True)


def test_extend_datatree_false_returns_forward_group():
    model = _gaussian_model()
    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=6,
        tune=0,
        include_posterior_predictive=False,
        random_seed=11,
    )

    out = sample_posterior_predictive_pro(
        dt,
        model=model,
        draw=-1,
        extend_datatree=False,
    )

    assert "posterior_predictive" in out
    assert np.all(np.isfinite(out.posterior_predictive["y"].values))


def test_oos_restores_pm_data():
    x_train = np.linspace(0.0, 1.0, 4)
    x_test = np.array([2.0, 2.5])

    with pm.Model(coords={"trial": np.arange(4)}) as model:
        x = pm.Data("x", x_train, dims="trial")
        alpha = pm.Normal("alpha", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=0.5, observed=np.zeros(4), dims="trial")

        dt = sample_pro(
            model=model,
            n_particles=4,
            n_steps=5,
            tune=0,
            include_posterior_predictive=False,
            random_seed=12,
        )
        sample_posterior_predictive_pro(
            dt,
            model=model,
            predictions=True,
            data={"x": x_test},
            coords={"trial": np.arange(2)},
        )

        np.testing.assert_allclose(model["x"].eval(), x_train)
        assert len(model.coords["trial"]) == len(x_train)


def test_oos_restores_integer_pm_data_dtype():
    x_train = np.array([0, 1, 0, 1], dtype="int32")
    x_test = np.array([1, 0], dtype="int32")

    with pm.Model(coords={"obs": np.arange(4)}) as model:
        x = pm.Data("x", x_train, dims="obs")
        beta = pm.Normal("beta", 0.0, 1.0, shape=2)
        mu = beta[x]
        pm.Normal("y", mu=mu, sigma=0.5, observed=np.zeros(4), dims="obs")

        dt = sample_pro(
            model=model,
            n_particles=4,
            n_steps=5,
            tune=0,
            include_posterior_predictive=False,
            random_seed=21,
        )
        sample_posterior_predictive_pro(
            dt,
            model=model,
            predictions=True,
            data={"x": x_test},
            coords={"obs": np.arange(2)},
        )

        restored = model["x"].eval()
        assert restored.dtype == x_train.dtype
        np.testing.assert_array_equal(restored, x_train)


def test_sample_pro_requires_observed_rvs():
    with pm.Model() as model:
        pm.Normal("mu", mu=0.0, sigma=1.0)

    with pytest.raises(ValueError, match="observed"):
        sample_pro(model=model, n_particles=4, n_steps=4, tune=0)


def test_sample_pro_rejects_nonpositive_learning_rate():
    model = _gaussian_model()

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        sample_pro(model=model, n_particles=4, n_steps=4, tune=0, learning_rate=0)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        sample_pro(model=model, n_particles=4, n_steps=4, tune=0, learning_rate=-1.0)
