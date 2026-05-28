"""Compilation helpers for logp and gradients."""

from __future__ import annotations

from typing import Callable, Sequence

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


def _sum_logp_terms(logp_terms: Sequence[pt.TensorVariable]) -> pt.TensorVariable:
    logp = pt.as_tensor(logp_terms[0])
    for term in logp_terms[1:]:
        logp = logp + pt.as_tensor(term)
    return logp


def compile_observed_logp(model=None) -> PointFunc:
    model = modelcontext(model)
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")

    logp_terms = model.logp(vars=model.observed_RVs, sum=False)
    if not isinstance(logp_terms, (list, tuple)):
        logp_terms = [logp_terms]

    logp_vec = _sum_logp_terms(logp_terms)
    logp_vec = pt.reshape(logp_vec, (-1,))

    return model.compile_fn(inputs=model.value_vars, outs=logp_vec, on_unused_input="ignore")


def compile_observed_score(model=None) -> PointFunc:
    model = modelcontext(model)
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")
    if model.discrete_value_vars:
        raise ValueError("Predictive score requires continuous model parameters.")
    require_unconstrained_free_rvs(model)

    value_vars = model.value_vars

    logp_terms = model.logp(vars=model.observed_RVs, sum=False)
    if not isinstance(logp_terms, (list, tuple)):
        logp_terms = [logp_terms]

    logp_vec = _sum_logp_terms(logp_terms)
    logp_vec = pt.reshape(logp_vec, (-1,))

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

    # elementwise observed log-probability
    logp_terms = model.logp(vars=model.observed_RVs, sum=False)
    if not isinstance(logp_terms, (list, tuple)):
        logp_terms = [logp_terms]

    logp_vec = pt.reshape(_sum_logp_terms(logp_terms), (-1,))
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
    logp_prior = model.logp(vars=model.free_RVs, jacobian=jacobian_terms, sum=True)
    prior_grad = gradient(logp_prior, value_vars)
    prior_grad = graph_replace(prior_grad, replace=replace, strict=False)
    return pt.reshape(prior_grad, (-1,))


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
    """Compile batched elementwise observed logp and score."""
    model = modelcontext(model)
    if mapper is None:
        raise ValueError("`mapper` is required for batched compilation.")
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")
    if model.discrete_value_vars:
        raise ValueError("Predictive score requires continuous model parameters.")

    particles = pt.matrix("particles")
    try:
        # prefer vectorised batching; fall back to scan if needed
        logp, score = _batched_observed_logp_score_graph(particles, model, mapper, use_scan=False)
        return model.compile_fn(
            inputs=[particles],
            outs=[logp, score],
            point_fn=False,
            on_unused_input="ignore",
        )
    except Exception:
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
    """Compile batched prior score gradients."""
    model = modelcontext(model)
    if model.discrete_value_vars:
        raise ValueError("Prior gradient requires continuous value variables.")
    if not model.free_RVs:
        raise ValueError("Model has no free random variables.")

    particles = pt.matrix("particles")
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
    except Exception:
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
    """Compile fused batched log-score drift for one EM step.

    For particle :math:`j`, evaluates the log-score Wasserstein term
    :math:`\\mathcal{W}(Q^{(j)})` (McLatchie et al., 2025, appendix
    “Examples: MMD and logarithmic score”) using the finite-:math:`p`
    leave-one-out mixture

    .. math::

        Q_t^{(j)} \\approx \\frac{1}{p-1}\\sum_{\\ell\\neq j}
        \\delta_{\\vartheta^{(\\ell)}},

    with mixture log-density computed via log-sum-exp over the batch.
    Per-observation log-likelihood ratios are clipped before exponentiating;
    the interaction drift is ``wgf_grad = -mean(ratio * score, axis=obs)``.

    Also compiles batched prior score gradients
    :math:`\\nabla_{\\vartheta}\\log\\pi(\\vartheta)` for the same particles.

    Returns
    -------
    wgf_grad
        Shape ``(n_particles, n_params)``.
    prior_grad
        Shape ``(n_particles, n_params)``.
    """
    model = modelcontext(model)
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")
    if model.discrete_value_vars:
        raise ValueError("Log-score requires continuous value variables.")
    if not model.free_RVs:
        raise ValueError("Model has no free random variables.")
    require_unconstrained_free_rvs(model)

    particles = pt.matrix("particles")
    try:
        logp, score = _batched_observed_logp_score_graph(particles, model, mapper, use_scan=False)
        prior_grad = _batched_prior_grad_graph(
            particles, model, mapper, jacobian_terms=jacobian, use_scan=False
        )
    except Exception:
        logp, score = _batched_observed_logp_score_graph(particles, model, mapper, use_scan=True)
        prior_grad = _batched_prior_grad_graph(
            particles, model, mapper, jacobian_terms=jacobian, use_scan=True
        )

    # leave-one-out mixture log density (log-space, numerically stable)
    logp_max = pt.max(logp, axis=0, keepdims=True)
    exp_shifted = pt.exp(logp - logp_max)
    sum_all = pt.sum(exp_shifted, axis=0, keepdims=True)
    sum_excl = pt.maximum(sum_all - exp_shifted, eps)

    denom = pt.cast(particles.shape[0] - 1, logp.dtype)
    log_mix = logp_max + pt.log(sum_excl) - pt.log(denom)
    log_ratio_raw = logp - log_mix
    log_ratio = pt.clip(log_ratio_raw, -log_ratio_clip, log_ratio_clip)
    ratio = pt.exp(log_ratio)
    # Wasserstein interaction term: shape (num_particles, dim)
    wgf_grad = -pt.mean(ratio[:, :, None] * score, axis=1)

    return model.compile_fn(
        inputs=[particles],
        outs=[wgf_grad, prior_grad],
        point_fn=False,
        on_unused_input="ignore",
    )


def count_observations(logp_fn: PointFunc, mapper: PointMapper) -> int:
    logp_vec = logp_fn(mapper.start_point)
    logp_vec = np.asarray(logp_vec)
    if logp_vec.ndim != 1:
        logp_vec = logp_vec.reshape(-1)
    return int(logp_vec.shape[0])
