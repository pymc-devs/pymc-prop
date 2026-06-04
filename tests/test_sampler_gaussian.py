import numpy as np
import pymc as pm

from pymc_prop.sample import sample_pro


def test_sample_pro_runs_gaussian():
    # smoke: full simulation loop returns finite retained particles (burn_in + thinning)
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=20)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)

    result = sample_pro(
        model=model,
        n_particles=8,
        n_steps=60,
        burn_in=10,
        thinning=5,
        step_size=5e-3,
        random_seed=123,
    )

    assert result.particles.ndim == 3
    assert result.particles.shape[1] == 8
    assert np.all(np.isfinite(result.particles))
