"""Particle initialization and Euler-Maruyama updates."""

from __future__ import annotations

import numpy as np

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_expression
from pymc.sampling.forward import draw

from pymc_prop.points import PointMapper


def initialize_particles(
    start: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
    jitter: float = 0.1,
) -> np.ndarray:
    """Place particles around a shared flat start with Gaussian jitter.

    ``start`` is typically the raveled PyMC ``initial_point()`` (prior-centred
    values in unconstrained ``value_vars`` space). Each row is
    ``start + jitter * N(0, I)``, spreading an initial **empirical particle
    measure**
    :math:`\\widehat{Q}_0 = \\frac{1}{p}\\sum_j \\delta_{\\vartheta_0^{(j)}}`
    so leave-one-particle-out measures :math:`Q_t^{(j)}` can interact from
    step one (see :func:`~pymc_prop.compile.compile_drift_for_logscore`).
    """
    base = np.asarray(start, dtype=float)
    noise = jitter * rng.standard_normal(size=(n_particles, base.size))
    return base[None, :] + noise


def initialize_particles_from_prior(
    model,
    mapper: PointMapper,
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one prior sample per particle (SMC-style population init).

    Uses ``make_initial_point_expression`` with ``default_strategy="prior"`` and
    PyMC's :func:`~pymc.sampling.forward.draw`, then ravels each draw through
    ``DictToArrayBijection`` into unconstrained ``value_vars`` space.
    """
    prior_expression = make_initial_point_expression(
        free_rvs=model.free_RVs,
        rvs_to_transforms=model.rvs_to_transforms,
        initval_strategies={**model.rvs_to_initial_values},
        default_strategy="prior",
        return_transformed=True,
    )
    prior_var_names = [model.rvs_to_values[rv].name for rv in model.free_RVs]

    prior_values = draw(prior_expression, draws=n_particles, random_seed=rng)

    dict_prior = dict(zip(prior_var_names, prior_values))
    population = []
    for i in range(n_particles):
        point = {
            var.name: np.asarray(dict_prior[var.name][i], dtype=float)
            for var in model.value_vars
            if var.name in dict_prior
        }
        for name, value in mapper.start_point.items():
            if name not in point:
                point[name] = np.asarray(value, dtype=float)
        population.append(DictToArrayBijection.map(point).data)

    return np.asarray(population, dtype=float)


def time_step(
    particles: np.ndarray,
    prior_grad: np.ndarray,
    wgf_grad: np.ndarray,
    step_size: float,
    learning_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance particles one discrete-time step along the log-score WGF.

    ``learning_rate`` is :math:`\\lambda_n`, ``step_size`` is :math:`\\varepsilon`;
    drift comes from :func:`~pymc_prop.compile.compile_drift_for_logscore`.
    """
    # drift = λ_n · wgf_grad − prior_grad
    drift = learning_rate * wgf_grad - prior_grad
    noise = np.sqrt(2.0 * step_size) * rng.standard_normal(size=particles.shape)
    return particles - step_size * drift + noise
