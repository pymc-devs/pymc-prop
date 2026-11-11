# pymc-prop

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pymc-devs/pymc-prop)

This is an experimental package for fitting predictively oriented posteriors in PyMC and is not ready for general use. API and numerics are likely to change.

## Predictively oriented posteriors

Predictively oriented (PrO) posteriors combine the strengths of parameter inference and density estimation. Instead of asking which parameter values are most plausible, they ask which mix of parameter values predicts new data best, and they express uncertainty accordingly. As a result, their predictions are at least as good as those from standard and generalised Bayesian posteriors.

When the model can reproduce the process that generated the data, PrO posteriors behave like typical Bayesian posteriors and narrow in on the true parameters. When the model is misspecified, they do not collapse onto a single answer. Instead, they settle on the best-predicting mixture of parameter values, and the uncertainty that remains can be read as a measure of how misspecified the model is.

## Overview

The long-term goal of pymc-prop is to fit predictively oriented posteriors for any valid PyMC model. In most cases, you should be able to use `sample_pro` in place of `pm.sample` and get results in ArviZ-compatible containers, ready for analysis and visualisation.


## Requirements

- Python ≥ 3.12
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

    dt = sample_pro(n_particles=32, n_steps=500, random_seed=0)
```

## Examples

The main walkthrough is [examples/bimodal_gaussian.ipynb](examples/bimodal_gaussian.ipynb): simulate a bimodal Gaussian mixture, fit a deliberately misspecified unimodal location model with `sample_pro`.

## Further reading

- McLatchie, Chérief-Abdellatif, Frazier & Knoblauch (2025). [*Predictively Oriented Posteriors*](https://arxiv.org/abs/2510.01915). [arXiv:2510.01915](https://arxiv.org/abs/2510.01915) — primary reference for theory and notation.
- Gu & Kim (2026). [*Mirror Langevin Dynamics*](https://arxiv.org/abs/2505.02621). [arXiv:2505.02621](https://arxiv.org/abs/2505.02621) — dual SDE / mirror noise for constrained parameters.
- Sharrock & Nemeth (2025). [*Tuning-Free Sampling via Optimization on the Space of Probability Measures*](https://arxiv.org/abs/2510.25315). [arXiv:2510.25315](https://arxiv.org/abs/2510.25315) — FUSE adaptive step-size schedule (default in `sample_pro`).
- [Yann McLatchie's blog tutorial](https://yannmclatchie.github.io/blog/posts/pro-tutorial/) — same WGF and particle picture in a misspecified Gaussian example; uses **MMD** and **JAX**. Helpful supplementary exposition.

## License

Apache-2.0 — see [LICENSE](LICENSE).

