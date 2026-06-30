import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pymc_prop import sample_posterior_predictive_pro, sample_pro
from pymc_prop.arviz import _mixture_remix_forward_dataset


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
    n_steps = 8
    n_obs = 5

    dt = sample_pro(
        model=model,
        n_particles=n_particles,
        n_steps=n_steps,
        tune=0,
        random_seed=0,
    )

    assert "posterior_predictive" in dt
    assert dt.posterior_predictive.attrs["sample_dims"] == ["draw"]
    assert dt.posterior_predictive.sizes["draw"] == n_steps
    assert "chain" not in dt.posterior_predictive.dims
    assert dt.posterior_predictive["y"].shape == (n_steps, n_obs)
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


def test_plot_ppc_dist_draw_only_layout():
    """ArviZ plot_ppc_dist works with draw-only sample_dims (no chain)."""
    import arviz as az

    model = _gaussian_model()
    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=8,
        tune=0,
        random_seed=40,
    )
    fig = az.plot_ppc_dist(dt, num_samples=5)
    assert fig is not None


def test_oos_predictions_linear_regression():
    x_train = np.linspace(0.0, 1.0, 5)
    x_test = np.linspace(1.1, 2.0, 3)
    n_steps = 6

    with pm.Model(coords={"trial": np.arange(5)}) as model:
        x = pm.Data("x", x_train, dims="trial")
        alpha = pm.Normal("alpha", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=0.5, observed=np.zeros(5), dims="trial")

        dt = sample_pro(
            model=model,
            n_particles=4,
            n_steps=n_steps,
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
    assert dt.predictions["y"].shape == (n_steps, 3)
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
    n_steps = 5

    with pm.Model(coords={"trial": np.arange(4)}) as model:
        x = pm.Data("x", x_train, dims="trial")
        alpha = pm.Normal("alpha", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=0.5, observed=np.zeros(4), dims="trial")

        dt = sample_pro(
            model=model,
            n_particles=4,
            n_steps=n_steps,
            tune=0,
            include_posterior_predictive=False,
            random_seed=7,
        )
        sample_posterior_predictive_pro(dt, model=model)
        sample_posterior_predictive_pro(
            dt,
            model=model,
            predictions=True,
            data={"x": x_test},
            coords={"trial": np.arange(1)},
        )

    assert "posterior_predictive" in dt
    assert "predictions" in dt
    assert dt.posterior_predictive["y"].shape == (n_steps, 4)
    assert dt.predictions["y"].shape == (n_steps, 1)


def test_pymc_default_sample_dims_differs_on_multi_draw():
    """Wrong flatten order swaps draw/chain when attrs are ignored."""
    import copy

    model = _gaussian_model()
    n_particles = 4
    n_steps = 6
    n_obs = 5

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
            random_seed=9,
        )

    assert wrong.posterior_predictive["y"].shape == (n_particles, n_steps, n_obs)
    assert native.posterior_predictive["y"].shape == (n_steps, n_obs)
    assert native.posterior_predictive.attrs["sample_dims"] == ["draw"]


def test_forward_grid_shape_before_remix():
    model = _gaussian_model()
    n_particles = 4
    n_steps = 6
    n_obs = 5

    dt = sample_pro(
        model=model,
        n_particles=n_particles,
        n_steps=n_steps,
        tune=0,
        include_posterior_predictive=False,
        random_seed=13,
    )

    with model:
        forward_dt = pm.sample_posterior_predictive(
            dt,
            var_names=["y"],
            sample_dims=["draw", "chain"],
            extend_inferencedata=False,
            random_seed=14,
            progressbar=False,
        )

    assert forward_dt.posterior_predictive["y"].shape == (n_steps, n_particles, n_obs)


def test_mixture_remix_matches_manual_isel():
    n_draw, n_chain, n_obs = 3, 4, 2
    grid_values = np.arange(n_draw * n_chain * n_obs, dtype=float).reshape(
        n_draw, n_chain, n_obs
    )
    grid = xr.Dataset(
        {"y": (("draw", "chain", "y_dim_0"), grid_values)},
        coords={
            "draw": np.arange(n_draw),
            "chain": np.arange(n_chain),
            "y_dim_0": np.arange(n_obs),
        },
    )

    seed = 221
    remixed = _mixture_remix_forward_dataset(grid, random_seed=seed)

    rng = np.random.default_rng(seed)
    target_shape = (n_draw, n_obs)
    random_chains = rng.integers(0, n_chain, size=target_shape)
    chain_da = xr.DataArray(random_chains, dims=["draw", "y_dim_0"])
    draw_da = xr.DataArray(
        np.broadcast_to(np.arange(n_draw, dtype=int).reshape(-1, 1), target_shape),
        dims=["draw", "y_dim_0"],
    )
    expected = grid["y"].isel(chain=chain_da, draw=draw_da).values

    np.testing.assert_array_equal(remixed["y"].values, expected)
    assert remixed.attrs["sample_dims"] == ["draw"]
    assert "chain" not in remixed.dims
    assert remixed.sizes["draw"] == n_draw


def test_remix_seed_reproducibility():
    model = _gaussian_model()

    dt_a = sample_pro(
        model=model,
        n_particles=4,
        n_steps=6,
        tune=0,
        include_posterior_predictive=False,
        random_seed=15,
    )
    dt_b = sample_pro(
        model=model,
        n_particles=4,
        n_steps=6,
        tune=0,
        include_posterior_predictive=False,
        random_seed=15,
    )

    sample_posterior_predictive_pro(dt_a, model=model, random_seed=16)
    sample_posterior_predictive_pro(dt_b, model=model, random_seed=16)
    np.testing.assert_array_equal(
        dt_a.posterior_predictive["y"].values,
        dt_b.posterior_predictive["y"].values,
    )

    sample_posterior_predictive_pro(dt_a, model=model, random_seed=17)
    assert not np.allclose(
        dt_a.posterior_predictive["y"].values,
        dt_b.posterior_predictive["y"].values,
    )


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


def test_extend_inferencedata_false_returns_forward_group():
    model = _gaussian_model()
    n_steps = 6
    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=n_steps,
        tune=0,
        include_posterior_predictive=False,
        random_seed=11,
    )

    out = sample_posterior_predictive_pro(
        dt,
        model=model,
        extend_inferencedata=False,
    )

    assert "posterior_predictive" in out
    assert out.posterior_predictive["y"].shape == (n_steps, 5)
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


def test_posterior_predictive_draw_aligned_with_posterior():
    """PPC draw and step coords match posterior."""
    model = _gaussian_model()
    n_steps = 8
    tune = 3

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=n_steps,
        tune=tune,
        random_seed=30,
    )

    np.testing.assert_array_equal(
        dt.posterior_predictive.coords["draw"].values,
        dt.posterior.coords["draw"].values,
    )
    if "step" in dt.posterior.coords:
        np.testing.assert_array_equal(
            dt.posterior_predictive.coords["step"].values,
            dt.posterior.coords["step"].values,
        )


def test_posterior_predictive_no_chain_dim():
    """PPC group has no chain dimension."""
    model = _gaussian_model()

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=6,
        tune=0,
        random_seed=31,
    )

    assert "chain" not in dt.posterior_predictive.dims


def test_mixture_remix_uses_all_retained_draws():
    """Forward grid n_steps → ppc n_steps (no thinning knob)."""
    model = _gaussian_model()
    n_steps = 10

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=n_steps,
        tune=0,
        random_seed=32,
    )

    assert dt.posterior_predictive.sizes["draw"] == n_steps
    assert dt.posterior_predictive["y"].shape[0] == n_steps
