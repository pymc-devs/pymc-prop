# pymc-pro

**Predictively Oriented (PrO) posteriors for [PyMC](https://www.pymc.io/).**

Experimental package implementing particle-based PrO inference in PyMC models. The current implementation focuses on the **log-score** scoring rule and its **Wasserstein gradient flow (WGF)** sampler.

## Overview

PrO posteriors target predictive performance under a chosen scoring rule, rather than the usual posterior from Bayes’ rule alone. This library compiles PyMC log-probability graphs and runs a particle ensemble in unconstrained `value_vars` space:

- **Prior score** — gradient of the prior log-density (free RVs only).
- **Observed score** — per-observation contributions from the likelihood (elementwise logp).
- **Sampler** — EM-style updates with log-score WGF particle dynamics (`sample_pro`).

Status: early research code; API and numerics may change.

## Reference

McLatchie, Chérief-Abdellatif, Frazier & Knoblauch (2025). [*Predictively Oriented Posteriors*](https://arxiv.org/abs/2510.01915). [arXiv:2510.01915](https://arxiv.org/abs/2510.01915)

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

    result = sample_pro(n_particles=32, n_steps=500, random_seed=0)```

`result` is a `PrOResult` with particle traces. Tune `n_particles`, `n_steps`, `burn_in`, `thinning`, `step_size`, and `learning_rate` for your model.

## API

Import from `pymc_prop`:

| Symbol | Role |
|--------|------|
| `sample_pro` | Main entry: run the PrO particle sampler on a PyMC model |
| `LogScore` | Log-score scoring rule and WGF implementation |
| `ScoringRule` | Protocol for scoring rules (extensible) |
| `compile_prior_gradient` | Compile ∇ log π (prior only) in unconstrained space |
| `compile_prior_grad` | Alias for `compile_prior_gradient` |

**Conventions**

- Particles live in PyMC’s unconstrained `value_vars` coordinates (via `PointMapper` / `DictToArrayBijection`).
- Observed log-probability is **elementwise** — one term per observation, not a single summed likelihood.
- Log-score sampling requires **continuous** value variables.

Only `scoring_rule="log"` is supported in this version.

## Development

```bash
pytest
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
