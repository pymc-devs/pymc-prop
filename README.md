# pymc-prop

**Predictively Oriented (PrO) posteriors for [PyMC](https://www.pymc.io/).**

Experimental package for fitting predictively oriented posteriors in PyMC. The current implementation focuses on the **log-score** scoring rule and its **Wasserstein gradient flow (WGF)** sampler.

## Predictively oriented posteriors

Predictively oriented (PrO) posteriors combine the most desirable aspects of both parameter inference and density estimation. They express parameter uncertainty as a consequence of predictive ability and learn the predictively optimal mixing distribution over model parameters. Doing so leads to inferences which predictively dominate both classical and generalised Bayes posterior predictive distributions: up to logarithmic factors, PrO posteriors converge to the predictively optimal model average. This means that they concentrate around the true model in the same way as Bayes and Gibbs posteriors if the model can recover the data-generating distribution, but do not concentrate in the presence of non-trivial forms of model misspecification. Instead, they stabilise towards a predictively-optimal posterior whose degree of irreducible uncertainty admits an interpretation as the degree of model misspecification. This package performs computation via a particle-discretised Wasserstein gradient flow via the primary exposed method `sample_pro`.

## Overview

This library compiles PyMC log-probability graphs fit for PrO posterior with three main components:

- **Prior score** — gradient of the prior log-density (free RVs only).
- **Observed score** — per-observation contributions from the likelihood (elementwise logp).
- **Sampler** — particle-discretised Wasserstein gradient flow (`sample_pro`).

Status: early research code; API and numerics may change.

## Requirements

- Python ≥ 3.10
- [PyMC](https://www.pymc.io/) and [PyTensor](https://pytensor.readthedocs.io/)
- NumPy; ArviZ and xarray for result containers

## Installation

From a clone of this repository:

```bash
pip install -e ".[test]"
```

## Quick start

```python
import pymc as pm
import numpy as np
from pymc_prop import sample_pro

y = np.random.default_rng(0).normal(0.0, 1.0, size=50)

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0.0, sigma=1.0)
    pm.Normal("y", mu=mu, sigma=1.0, observed=y)

    result = sample_pro(n_particles=32, n_steps=500, random_seed=0)
```

`result` is a `PrOResult` with particle traces. Tune `n_particles`, `n_steps`, `burn_in`, `thinning`, `step_size`, and `learning_rate` for your model.

## Tutorials

The main walkthrough is [examples/bimodal_gaussian.ipynb](examples/bimodal_gaussian.ipynb): simulate a bimodal Gaussian mixture, fit a deliberately misspecified unimodal location model with `sample_pro`, and inspect the PrO posterior predictive. We find that, since the true data-generating process can be recovered by a mixture over the model class, the PrO posterior recovers it exactly.

## Implementation

Log-score particle drift is implemented in [`compile_drift_for_logscore`](pymc_prop/compile.py). Analytical checks and regression tests live in `tests/test_logscore_wgf.py`.

## API

Import from `pymc_prop`:

| Symbol | Role |
|--------|------|
| `sample_pro` | Main entry: run the PrO particle sampler on a PyMC model |
| `LogScore` | Log-score scoring rule (drift via `compile_drift_for_logscore`) |
| `ScoringRule` | Protocol for scoring rules (extensible) |
| `compile_prior_gradient` | Compile ∇ log π (prior only) in unconstrained space |
| `compile_prior_grad` | Alias for `compile_prior_gradient` |

**Conventions**

- Particles live in PyMC’s unconstrained `value_vars` coordinates (via `PointMapper` / `DictToArrayBijection`).
- For now, free RVs **must** be native unconstrained.
- Observed log-probability is **elementwise** — one term per observation, not a single summed likelihood.
- Log-score sampling requires **continuous** value variables.

Only `scoring_rule="log"` is supported in this version.

## Further reading

- McLatchie, Chérief-Abdellatif, Frazier & Knoblauch (2025). [*Predictively Oriented Posteriors*](https://arxiv.org/abs/2510.01915). [arXiv:2510.01915](https://arxiv.org/abs/2510.01915) — primary reference for theory and notation.
- [Yann McLatchie's blog tutorial](https://yannmclatchie.github.io/blog/posts/pro-tutorial/) — same WGF and particle picture in a misspecified Gaussian example; uses **MMD** and **JAX**. Helpful supplementary exposition.

## Development

```bash
pytest
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
