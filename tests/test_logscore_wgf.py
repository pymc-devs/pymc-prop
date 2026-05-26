import numpy as np
import pymc as pm

from pymc_prop.points import make_point_mapper
from pymc_prop.scoring import LogScore


def _manual_wgf(particles, y, sigma, log_ratio_clip=50.0, eps=1e-300):
    p, d = particles.shape
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]

    diff = y[None, :] - particles
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma**2)
    logp = log_norm - 0.5 * (diff**2) / (sigma**2)

    logp_max = np.max(logp, axis=0, keepdims=True)
    exp_shifted = np.exp(logp - logp_max)
    sum_all = np.sum(exp_shifted, axis=0, keepdims=True)
    sum_excl = np.maximum(sum_all - exp_shifted, eps)
    log_mix = logp_max + np.log(sum_excl) - np.log(float(p - 1))

    log_ratio_raw = logp - log_mix
    log_ratio = np.clip(log_ratio_raw, -log_ratio_clip, log_ratio_clip)
    ratio = np.exp(log_ratio)

    score = diff / (sigma**2)
    return -np.mean(ratio[:, :, None] * score[:, :, None], axis=1)


def test_logscore_wgf_matches_manual():
    rng = np.random.default_rng(10)
    y = rng.normal(0.0, 1.0, size=8)
    sigma = 1.0

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    particles = np.array([[-1.0], [0.5], [2.0]], dtype=float)

    mapper = make_point_mapper(model)
    logscore = LogScore(log_ratio_clip=50.0)
    wgf_fn = logscore.compile_wgf(model, mapper)
    wgf, _ = wgf_fn(particles, diagnostics=False)

    manual = _manual_wgf(particles, y, sigma, log_ratio_clip=50.0)
    np.testing.assert_allclose(wgf, manual, rtol=1e-5, atol=1e-6)
