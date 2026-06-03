"""Compilation helpers for logp and gradients."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.replace import graph_replace
from pytensor.scan import scan

from pymc.model import modelcontext
from pymc.pytensorf import gradient, jacobian

from pymc_prop.points import PointMapper, flat_to_value_vars, require_unconstrained_free_rvs


PointFunc = Callable[[dict[str, np.ndarray]], np.ndarray]
FlatGradFunc = Callable[[np.ndarray], np.ndarray]
BatchedLogpScoreFunc = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
BatchedGradFunc = Callable[[np.ndarray], np.ndarray]
DriftFunc = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def compile_observed_logp(model=None) -> PointFunc:
    """Elementwise observed logp; output shape ``(n_obs,)`` (not summed)."""
    model = modelcontext(model)
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")

    logp_terms = model.logp(vars=model.observed_RVs, sum=False)
    logp_vec = pt.flatten(pt.add(*logp_terms))

    return model.compile_fn(inputs=model.value_vars, outs=logp_vec, on_unused_input="ignore")


def compile_observed_score(model=None) -> PointFunc:
    """Per-observation score rows via ``jacobian``; shape ``(n_obs, n_params)``."""
    model = modelcontext(model)
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")
    if model.discrete_value_vars:
        raise ValueError("Predictive score requires continuous model parameters.")
    require_unconstrained_free_rvs(model)

    value_vars = model.value_vars

    logp_terms = model.logp(vars=model.observed_RVs, sum=False)
    logp_vec = pt.flatten(pt.add(*logp_terms))
    # ensure a 1-D vector of per-observation logp
    logp_vec = pt.flatten(logp_vec)

    scores = jacobian(logp_vec, value_vars)
    return model.compile_fn(inputs=value_vars, outs=scores, on_unused_input="ignore")


def compile_prior_gradient(model=None, *, jacobian: bool = True) -> PointFunc:
    """Compile prior score gradient in unconstrained value-var space."""
    model = modelcontext(model)
    if model.discrete_value_vars:
        raise ValueError("Prior gradient requires continuous value variables.")
    if not model.free_RVs:
        raise ValueError("Model has no free random variables.")
    require_unconstrained_free_rvs(model)

    # prior term: free RVs only (not the joint logp)
    logp_prior = model.logp(vars=model.free_RVs, jacobian=jacobian, sum=True)
    return model.compile_fn(
        inputs=model.value_vars,
        outs=gradient(logp_prior, model.value_vars),
        on_unused_input="ignore",
    )


def compile_prior_grad(model=None, *, jacobian: bool = True) -> PointFunc:
    """Alias for :func:`compile_prior_gradient`."""
    return compile_prior_gradient(model, jacobian=jacobian)


def compile_flat_prior_grad(
    mapper: PointMapper, model=None, *, jacobian: bool = True
) -> FlatGradFunc:
    """Wrap the prior gradient as a function of flat particle vectors."""
    grad_fn = compile_prior_gradient(model, jacobian=jacobian)

    def flat_grad(particle: np.ndarray) -> np.ndarray:
        return np.asarray(grad_fn(mapper.unravel(particle)), dtype=float)

    return flat_grad


def _core_observed_logp_score(
    particle_flat: pt.TensorVariable, model, mapper: PointMapper
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """Elementwise observed logp and score for one flat particle."""
    value_vars = model.value_vars
    mapped_value_vars = flat_to_value_vars(particle_flat, mapper.point_map_info)
    replace = dict(zip(value_vars, mapped_value_vars, strict=True))

    logp_terms = model.logp(vars=model.observed_RVs, sum=False)
    logp_vec = pt.flatten(pt.add(*logp_terms))
    # score matrix: one row per observation
    score_mat = jacobian(logp_vec, value_vars)
    logp_vec, score_mat = graph_replace([logp_vec, score_mat], replace=replace, strict=False)
    return logp_vec, score_mat


def _core_prior_grad(
    particle_flat: pt.TensorVariable, model, mapper: PointMapper, *, jacobian_terms: bool
) -> pt.TensorVariable:
    """Prior score gradient for one flat particle."""
    value_vars = model.value_vars
    mapped_value_vars = flat_to_value_vars(particle_flat, mapper.point_map_info)
    replace = dict(zip(value_vars, mapped_value_vars, strict=True))
    # log prior gradient (prior term)
    logp_prior = model.logp(vars=model.free_RVs, jacobian=jacobian_terms, sum=True)
    prior_grad = gradient(logp_prior, value_vars)
    prior_grad = graph_replace(prior_grad, replace=replace, strict=False)
    # flatten prior gradient vector to 1-D
    return pt.flatten(prior_grad)


def _batched_observed_logp_score_graph(
    particles: pt.TensorVariable, model, mapper: PointMapper, *, use_scan: bool
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """Batch observed logp and score over particles."""
    if use_scan:
        (logp, score), _ = scan(
            fn=lambda particle: _core_observed_logp_score(particle, model, mapper),
            sequences=[particles],
        )
        return logp, score

    # vectorised evaluation over the particle batch
    vec_fn = pt.vectorize(
        lambda particle: _core_observed_logp_score(particle, model, mapper),
        signature="(d)->(n),(n,d)",
    )
    logp, score = vec_fn(particles)
    return logp, score


def _batched_prior_grad_graph(
    particles: pt.TensorVariable,
    model,
    mapper: PointMapper,
    *,
    jacobian_terms: bool,
    use_scan: bool,
) -> pt.TensorVariable:
    """Batch prior score gradients over particles."""
    if use_scan:
        prior_grad, _ = scan(
            fn=lambda particle: _core_prior_grad(
                particle, model, mapper, jacobian_terms=jacobian_terms
            ),
            sequences=[particles],
        )
        return prior_grad

    # vectorised evaluation over the particle batch
    vec_fn = pt.vectorize(
        lambda particle: _core_prior_grad(
            particle, model, mapper, jacobian_terms=jacobian_terms
        ),
        signature="(d)->(d)",
    )
    return vec_fn(particles)


def compile_batched_observed_logp_score(model=None, mapper: PointMapper | None = None) -> BatchedLogpScoreFunc:
    """Compile batched elementwise observed logp and score.

    Uses vectorized batching when possible; otherwise ``scan`` at compile time.
    """
    model = modelcontext(model)
    if mapper is None:
        raise ValueError("`mapper` is required for batched compilation.")
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")
    if model.discrete_value_vars:
        raise ValueError("Predictive score requires continuous model parameters.")

    particles = pt.matrix("particles")
    # Compile-time only: batch particle rows with pt.vectorize. If graph construction
    # fails (often flat_to_value_vars reshapes in the logp/score subgraph), rebuild
    # with scan over rows (slower, identical output shapes).
    try:
        logp, score = _batched_observed_logp_score_graph(particles, model, mapper, use_scan=False)
        return model.compile_fn(
            inputs=[particles],
            outs=[logp, score],
            point_fn=False,
            on_unused_input="ignore",
        )
    except Exception:  # vectorize path failed at graph build; use scan
        logp, score = _batched_observed_logp_score_graph(particles, model, mapper, use_scan=True)
        return model.compile_fn(
            inputs=[particles],
            outs=[logp, score],
            point_fn=False,
            on_unused_input="ignore",
        )


def compile_batched_prior_grad(
    mapper: PointMapper,
    model=None,
    *,
    jacobian: bool = True,
) -> BatchedGradFunc:
    """Compile batched prior score gradients.

    Uses vectorized batching when possible; otherwise ``scan`` at compile time.
    """
    model = modelcontext(model)
    if model.discrete_value_vars:
        raise ValueError("Prior gradient requires continuous value variables.")
    if not model.free_RVs:
        raise ValueError("Model has no free random variables.")

    particles = pt.matrix("particles")
    # Compile-time only: batch particle rows with pt.vectorize. If graph construction
    # fails (often flat_to_value_vars reshapes in the logp/score subgraph), rebuild
    # with scan over rows (slower, identical output shapes).
    try:
        prior_grad = _batched_prior_grad_graph(
            particles, model, mapper, jacobian_terms=jacobian, use_scan=False
        )
        return model.compile_fn(
            inputs=[particles],
            outs=prior_grad,
            point_fn=False,
            on_unused_input="ignore",
        )
    except Exception:  # vectorize path failed at graph build; use scan
        prior_grad = _batched_prior_grad_graph(
            particles, model, mapper, jacobian_terms=jacobian, use_scan=True
        )
        return model.compile_fn(
            inputs=[particles],
            outs=prior_grad,
            point_fn=False,
            on_unused_input="ignore",
        )


def compile_drift_for_logscore(
    mapper: PointMapper,
    model=None,
    *,
    log_ratio_clip: float = 10.0,
    eps: float = 1e-300,
    jacobian: bool = True,
) -> DriftFunc:
    """Compile log-score interaction and prior drift for one time step.

    .. math::

        \\mathrm{d}\\vartheta_t^{(j)}
        = -\\Bigl\\{
          \\lambda_n \\sum_i
          \\frac{\\nabla_\\vartheta\\, p_{\\vartheta^{(j)}}(x_i)}
          {\\frac{1}{p-1} \\sum_{\\ell\\neq j} p_{\\vartheta^{(\\ell)}}(x_i)}
          - \\nabla_\\vartheta \\log \\pi(\\vartheta^{(j)})
        \\Bigr\\}\\,\\mathrm{d}t + \\sqrt{2}\\,\\mathrm{d}B_t^{(j)}.

    where:

    - :math:`\\vartheta_t^{(j)}` is the :math:`j`-th particle at time :math:`t`
    - :math:`p_{\\vartheta^{(j)}}(x_i)` is the model density of particle :math:`j`
      evaluated at data point :math:`x_i`
    - :math:`\\log \\pi(\\vartheta^{(j)})` is the log prior density of the :math:`j`-th
      particle at time :math:`t`
    - :math:`\\mathrm{d}B_t^{(j)}` is a realisation from a Brownian motion at
      time :math:`t`

    **Implementation details**

    The numerator :math:`\\nabla_\\vartheta p_{\\vartheta^{(j)}}(x_i)` is computed by 
    the chain rule:

    .. math::

        \\nabla_\\vartheta p_{\\vartheta^{(j)}}(x_i)
        = p_{\\vartheta^{(j)}}(x_i)
          \\cdot \\nabla_\\vartheta \\log p_{\\vartheta^{(j)}}(x_i)

    so the particle-dependent term in the above algorithm reduces to an importance weighted score:

    .. math::

        \\frac{\\nabla_\\vartheta p_{\\vartheta^{(j)}}(x_i)}{q_{-j}(x_i)}
        = \\underbrace{\\frac{p_{\\vartheta^{(j)}}(x_i)}{q_{-j}(x_i)}}_{w_j(x_i)}
          \\cdot \\nabla_\\vartheta \\log p_{\\vartheta^{(j)}}(x_i).

    Computing :math:`q_{-j}(x_i) = \\frac{1}{p-1} \\sum_{\\ell \\neq j} p_{\\vartheta^{(\\ell)}}(x_i)` 
    directly in probability space is numerically unstable so instead we work in log-space and use the 
    log-sum-exp trick. Define the particle-wise maximum density as

    .. math::

        m(x_i) = \\max_j \\log p_{\\vartheta^{(j)}}(x_i)

    then the leave-one-out log-mixture is

    .. math::

        \\log q_{-j}(x_i)
        = m(x_i)
          + \\log\\!\\Bigl(\\sum_{\\ell \\neq j}
            e^{\\log p_{\\vartheta^{(\\ell)}}(x_i)\\,-\\,m(x_i)}\\Bigr)
          - \\log(p - 1)

    where the sum :math:`\\sum_{\\ell \\neq j}` is computed by forming the full sum
    over all particles and subtracting particle :math:`j`'s own contribution. The raw 
    log importance weight

    .. math::

        \\log w_j(x_i) = \\log p_{\\vartheta^{(j)}}(x_i) - \\log q_{-j}(x_i)

    can be large when particle :math:`j` has high density under the target but low
    density under the mixture of other particles, causing exploding gradients.
    We therefore clip in log-space before exponentiating for numerical stability:

    .. math::

        \\log \\tilde{w}_j(x_i)
        = \\mathrm{clip}\\bigl(\\log w_j(x_i),\\,
          -c,\\, c\\bigr)

    where :math:`c` = ``log_ratio_clip``. The lower bound avoids negligible
    weights from large negative log-ratios; the upper bound caps importance
    weights when particle :math:`j` dominates the LOO mixture on :math:`y_i`,
    which can otherwise explode ``ratio * score`` in the drift. Clipping in
    log-space is preferable to clipping weights directly.

    Parameters
    ----------
    mapper
        Point mapper for the model.
    model
        PyMC model.
    log_ratio_clip
        Clip log likelihood ratios before exponentiating (stability only).
    eps
        Floor for leave-one-particle-out normalising sums in the compiled graph.
    jacobian
        Whether to use the Jacobian of the log prior.
    
    Returns
    -------
    wgf_grad
        Interaction drift, shape ``(n_particles, n_params)``.
    prior_grad
        Prior score gradient, shape ``(n_particles, n_params)``.

    Shapes (``p`` = particles, ``d`` = raveled ``value_vars``, ``n_obs`` observations):

    - Input ``particles``: ``(p, d)``
    - Internal ``logp``: ``(p, n_obs)``; ``score``: ``(p, n_obs, d)``
    """
    model = modelcontext(model)
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")
    if model.discrete_value_vars:
        raise ValueError("Log-score requires continuous value variables.")
    if not model.free_RVs:
        raise ValueError("Model has no free random variables.")

    # Model guard: reject implicit transforms on free RVs.
    require_unconstrained_free_rvs(model)

    particles = pt.matrix("particles")  # compile input (p, d): one row per particle in flat value_vars space
    # Compile-time only: batch particle rows with pt.vectorize. If graph construction
    # fails (often flat_to_value_vars reshapes in the logp/score subgraph), rebuild
    # with scan over rows (slower, identical output shapes).
    try:
        # logp: p_{ϑ^{(j)}}(x_i); score: ∇_ϑ log p(y_i|ϑ^{(j)}); prior_grad: ∇_ϑ log π 
        # per particle: log p(y_i), score rows, log prior gradient
        logp, score = _batched_observed_logp_score_graph(particles, model, mapper, use_scan=False)
        prior_grad = _batched_prior_grad_graph(
            particles, model, mapper, jacobian_terms=jacobian, use_scan=False
        )
    except Exception:  # vectorize path failed at graph build; use scan
        logp, score = _batched_observed_logp_score_graph(particles, model, mapper, use_scan=True)
        prior_grad = _batched_prior_grad_graph(
            particles, model, mapper, jacobian_terms=jacobian, use_scan=True
        )

    # subtract particle-wise max before exponentiating to prevent overflow/underflow
    logp_max = pt.max(logp, axis=0, keepdims=True)
    exp_shifted = pt.exp(logp - logp_max)

    # leave-one-particle-out density: Σ_{ℓ≠j} p_{ϑ^{(ℓ)}}
    sum_all = pt.sum(exp_shifted, axis=0, keepdims=True)
    sum_excl = pt.maximum(sum_all - exp_shifted, eps)

    # restore the log p_max shift and divide by (p-1)
    denom = pt.cast(particles.shape[0] - 1, logp.dtype)
    log_mix = logp_max + pt.log(sum_excl) - pt.log(denom)  # (1/(p-1)) Σ_{ℓ≠j} p_{ϑ^{(ℓ)}}
    
    # log importance weights vs mixture for chain rule, clipped for stability
    log_ratio_raw = logp - log_mix
    # symmetric cap on log w_{i,j} before exp (stability)
    log_ratio = pt.clip(log_ratio_raw, -log_ratio_clip, log_ratio_clip)
    ratio = pt.exp(log_ratio)  # exponentiate for importance weights
    
    # particle interaction term computed by chain rule
    # − Σ_i w_{i,j} ∇_ϑ log p(y_i|ϑ^{(j)})  (λ_n applied in time_step)
    wgf_grad = -pt.sum(ratio[:, :, None] * score, axis=1)
    
    # return compiled functions
    return model.compile_fn(
        inputs=[particles],
        outs=[wgf_grad, prior_grad],
        point_fn=False,
        on_unused_input="ignore",
    )
