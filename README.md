# pymc-prop

**Predictively Oriented (PrO) posteriors for [PyMC](https://www.pymc.io/).**

Experimental package implementing particle-based PrO inference in PyMC models. The current implementation focuses on the **log-score** scoring rule and its **Wasserstein gradient flow (WGF)** sampler.

## What is PrO?

Predictively Oriented posteriors optimize over **mixing distributions** \(Q\) on parameters—the distributions you actually use to form predictions—not over a single point estimate obtained by integrating under a Gibbs posterior and then predicting. Under **misspecification**, Bayes and Gibbs inference often collapse to one mode that minimizes KL to the data-generating process, even when no single parameter value predicts well everywhere. PrO instead asks which \(Q\) minimizes expected scoring-rule loss; when the model family cannot match the data with one point, the optimal \(Q\) may spread mass across several modes—the **mixability gap** between point predictives and mixtures over them. This package approximates that \(Q\) with a **particle ensemble** in unconstrained PyMC `value_vars` space, updated by a log-score **Wasserstein gradient flow** via `sample_pro`.

## Overview

This library compiles PyMC log-probability graphs and runs those particles with three main pieces:

- **Prior score** — gradient of the prior log-density (free RVs only).
- **Observed score** — per-observation contributions from the likelihood (elementwise logp).
- **Sampler** — EM-style updates with log-score WGF particle dynamics (`sample_pro`).

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

For the full misspecification demo (the running example from the PrO literature), see the [tutorial notebook](examples/bimodal_gaussian.ipynb) below.

## Tutorial

The main walkthrough is [examples/bimodal_gaussian.ipynb](examples/bimodal_gaussian.ipynb): simulate a bimodal Gaussian mixture, fit a deliberately misspecified unimodal location model, run `sample_pro`, and inspect particle trajectories plus posterior predictive mass on both data modes.

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
- Free RVs must be native unconstrained; reparameterize manually for now.
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
