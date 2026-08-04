"""Compilation helpers for logp and gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.replace import graph_replace
from pytensor.scan import scan

from pymc.model import modelcontext
from pymc.pytensorf import gradient, jacobian

from pymc_prop.points import (
    PointMapper,
    flat_to_value_vars,
    require_mirror_compatible_transforms,
)


PointFunc = Callable[[dict[str, np.ndarray]], np.ndarray]
FlatGradFunc = Callable[[np.ndarray], np.ndarray]
BatchedLogpFunc = Callable[[np.ndarray], np.ndarray]
BatchedLogpScoreFunc = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
BatchedGradFunc = Callable[[np.ndarray], np.ndarray]
DriftFunc = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

# Compile-time only: graph construction failures that trigger scan fallback.
_VECTORIZE_FALLBACK_ERRORS = (TypeError, ValueError, AttributeError, NotImplementedError)


def _observed_logp_vector(model, observed_rvs: Sequence) -> pt.TensorVariable:
    """Elementwise observed logp vector (not yet mapped to flat particles)."""
    logp_terms = model.logp(vars=observed_rvs, sum=False)
    return pt.flatten(pt.add(*logp_terms))


def _try_vectorize_then_scan(build: Callable[[bool], Any]) -> Any:
    """Build a batched graph with vectorize, falling back to scan at compile time."""
    try:
        return build(use_scan=False)
    except _VECTORIZE_FALLBACK_ERRORS:
        return build(use_scan=True)


def _compile_particle_batch(model, particles: pt.TensorVariable, outs: Any) -> Any:
    """Compile a batched particle function."""
    if isinstance(outs, tuple):
        outs_list = list(outs)
    else:
        outs_list = outs
    return model.compile_fn(
        inputs=[particles],
        outs=outs_list,
        point_fn=False,
        on_unused_input="ignore",
    )


def compile_observed_logp(model=None) -> PointFunc:
    """Elementwise observed logp; output shape ``(n_obs,)`` (not summed)."""
    model = modelcontext(model)
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")

    logp_vec = _observed_logp_vector(model, model.observed_RVs)

    return model.compile_fn(inputs=model.value_vars, outs=logp_vec, on_unused_input="ignore")


def compile_observed_score(model=None) -> PointFunc:
    """Per-observation score rows via ``jacobian``; shape ``(n_obs, n_params)``.

    Returns the gradient w.r.t. unconstrained ``value_vars``. For
    mirror-mapped Euler–Maruyama drift use :func:`compile_drift_for_logscore`.
    """
    model = modelcontext(model)
    if not model.observed_RVs:
        raise ValueError("Model has no observed variables.")
    if model.discrete_value_vars:
        raise ValueError("Predictive score requires continuous model parameters.")
    require_mirror_compatible_transforms(model)

    value_vars = model.value_vars
    logp_vec = _observed_logp_vector(model, model.observed_RVs)
    scores = jacobian(logp_vec, value_vars)
    return model.compile_fn(inputs=value_vars, outs=scores, on_unused_input="ignore")


def compile_prior_gradient(model=None, *, jacobian: bool | None = None) -> PointFunc:
    """Compile prior score gradient w.r.t. unconstrained ``value_vars``.

    When free RVs use elementwise transforms, default ``jacobian=False`` so
    ``model.logp`` is the constrained-space potential :math:`f` (mirror Langevin
    dynamics; Gu & Kim, 2025, §2.1), not the change-of-variables
    density on unconstrained coordinates. For particle-step prior drift use
    :func:`compile_flat_prior_grad`, :func:`compile_batched_prior_grad`, or
    :func:`compile_drift_for_logscore`.
    """
    model = modelcontext(model)
    if model.discrete_value_vars:
        raise ValueError("Prior gradient requires continuous value variables.")
    if not model.free_RVs:
        raise ValueError("Model has no free random variables.")
    require_mirror_compatible_transforms(model)

    has_transforms = any(
        model.rvs_to_transforms.get(rv) is not None for rv in model.free_RVs
    )
    if jacobian is None:
        jacobian = not has_transforms

    # prior term: free RVs only (not the joint logp)
    logp_prior = model.logp(vars=model.free_RVs, jacobian=jacobian, sum=True)
    return model.compile_fn(
        inputs=model.value_vars,
        outs=gradient(logp_prior, model.value_vars),
        on_unused_input="ignore",
    )


def compile_prior_grad(model=None, *, jacobian: bool | None = None) -> PointFunc:
    """Alias for :func:`compile_prior_gradient`."""
    return compile_prior_gradient(model, jacobian=jacobian)


def compile_flat_prior_grad(
    mapper: PointMapper, model=None, *, jacobian: bool | None = None
) -> FlatGradFunc:
    """Prior gradient on a flat unconstrained particle for :func:`~pymc_prop.particles.time_step`.

    Compiles the dual (unconstrained) prior gradient, then scales at runtime by
    :meth:`~pymc_prop.points.PointMapper.primal_scale` (identity → ones) so the
    returned layout matches the constrained-space gradient used in the time step.
    """
    grad_fn = compile_prior_gradient(model, jacobian=jacobian)

    def flat_grad(particle: np.ndarray) -> np.ndarray:
        unconstrained_grad = np.asarray(grad_fn(mapper.unravel(particle)), dtype=float)
        return unconstrained_grad * mapper.primal_scale(particle)

    return flat_grad


def _core_observed_logp(
    particle_flat: pt.TensorVariable,
    model,
    mapper: PointMapper,
    observed_rvs: Sequence,
) -> pt.TensorVariable:
    """Elementwise observed logp for one flat particle (no jacobian).

    Split from :func:`_core_observed_logp_score` because post-sample groups
    (``log_likelihood``, ``mixture_log_predictive``) only need
    :math:`\\log p(y_i \\mid \\theta)`. Building the full observation
    ``jacobian`` there duplicated drift-path work and inflated compile and eval
    time with no change to the stored ArviZ arrays. Drift still uses
    :func:`compile_batched_observed_logp_score` / :func:`compile_drift_for_logscore`.
    """
    value_vars = model.value_vars
    mapped_value_vars = flat_to_value_vars(particle_flat, mapper.point_map_info)
    replace = dict(zip(value_vars, mapped_value_vars, strict=True))
    logp_vec = _observed_logp_vector(model, observed_rvs)
    return graph_replace(logp_vec, replace=replace, strict=False)


def _core_observed_logp_score(
    particle_flat: pt.TensorVariable, model, mapper: PointMapper
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """Elementwise observed logp and score for one flat particle."""
    value_vars = model.value_vars
    mapped_value_vars = flat_to_value_vars(particle_flat, mapper.point_map_info)
    replace = dict(zip(value_vars, mapped_value_vars, strict=True))

    logp_vec = _observed_logp_vector(model, model.observed_RVs)
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
    return pt.flatten(prior_grad)


def _batched_observed_logp_graph(
    particles: pt.TensorVariable,
    model,
    mapper: PointMapper,
    observed_rvs: Sequence,
    *,
    use_scan: bool,
) -> pt.TensorVariable:
    """Batch elementwise observed logp over particles; signature ``(d)->(n)``."""
    if use_scan:
        logp, _ = scan(
            fn=lambda particle: _core_observed_logp(particle, model, mapper, observed_rvs),
            sequences=[particles],
        )
        return logp

    vec_fn = pt.vectorize(
        lambda particle: _core_observed_logp(particle, model, mapper, observed_rvs),
        signature="(d)->(n)",
    )
    return vec_fn(particles)


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


def compile_batched_observed_logp(
    model=None,
    mapper: PointMapper | None = None,
    *,
    observed_rvs: Sequence | None = None,
) -> BatchedLogpFunc:
    """Compile batched elementwise observed logp (no jacobian).

    Post-sample packaging path only; log-score drift uses
    :func:`compile_batched_observed_logp_score`. Output shape
    ``(n_particles, n_obs)``. Uses vectorized batching when possible; otherwise
    ``scan`` at compile time.
    """
    model = modelcontext(model)
    if mapper is None:
        raise ValueError("`mapper` is required for batched compilation.")
    if observed_rvs is None:
        observed_rvs = model.observed_RVs
    if not observed_rvs:
        raise ValueError("Model has no observed variables.")

    particles = pt.matrix("particles")

    def build(use_scan: bool) -> pt.TensorVariable:
        return _batched_observed_logp_graph(
            particles, model, mapper, observed_rvs, use_scan=use_scan
        )

    outs = _try_vectorize_then_scan(build)
    return _compile_particle_batch(model, particles, outs)


def compile_batched_observed_logp_for_rv(
    model=None,
    mapper: PointMapper | None = None,
    rv=None,
) -> BatchedLogpFunc:
    """Compile batched elementwise logp for one observed RV."""
    model = modelcontext(model)
    if mapper is None:
        raise ValueError("`mapper` is required for batched compilation.")
    if rv is None:
        raise ValueError("`rv` is required.")
    return compile_batched_observed_logp(model, mapper, observed_rvs=[rv])


def compile_batched_observed_logp_score(model=None, mapper: PointMapper | None = None) -> BatchedLogpScoreFunc:
    """Compile batched elementwise observed logp and score w.r.t. unconstrained ``value_vars``.

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

    def build(use_scan: bool) -> tuple[pt.TensorVariable, pt.TensorVariable]:
        return _batched_observed_logp_score_graph(particles, model, mapper, use_scan=use_scan)

    outs = _try_vectorize_then_scan(build)
    return _compile_particle_batch(model, particles, outs)


def compile_batched_prior_grad(
    mapper: PointMapper,
    model=None,
    *,
    jacobian: bool | None = None,
) -> BatchedGradFunc:
    """Compile batched prior gradients.

    Compiles the raw dual (unconstrained) prior-gradient graph, then scales at
    runtime by :meth:`~pymc_prop.points.PointMapper.primal_scale` (identity →
    ones) so the returned layout matches the constrained-space gradient used in
    the time step. Uses vectorized batching when possible; otherwise ``scan`` at
    compile time. When ``jacobian`` is ``None``, defaults to ``False`` under
    transforms (constrained-space potential for mirror Langevin dynamics) and
    ``True`` otherwise.
    """
    model = modelcontext(model)
    if model.discrete_value_vars:
        raise ValueError("Prior gradient requires continuous value variables.")
    if not model.free_RVs:
        raise ValueError("Model has no free random variables.")
    require_mirror_compatible_transforms(model)
    if jacobian is None:
        jacobian = not mapper.has_transforms

    particles = pt.matrix("particles")

    def build(use_scan: bool) -> pt.TensorVariable:
        return _batched_prior_grad_graph(
            particles, model, mapper, jacobian_terms=jacobian, use_scan=use_scan
        )

    outs = _try_vectorize_then_scan(build)
    raw_fn = _compile_particle_batch(model, particles, outs)

    def batched(particles_np: np.ndarray) -> np.ndarray:
        g = np.asarray(raw_fn(particles_np), dtype=float)
        return g * mapper.primal_scale(particles_np)

    return batched


def compile_drift_for_logscore(
    mapper: PointMapper,
    model=None,
    *,
    log_ratio_clip: float = 10.0,
    eps: float = 1e-300,
    jacobian: bool | None = None,
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
    weights when particle :math:`j` dominates the leave-one-out mixture on
    :math:`y_i`, which can otherwise explode ``ratio * score`` in the drift.
    Clipping in log-space is preferable to clipping weights directly.

    **Mirror-mapped parameters.** Particles stay in unconstrained ``value_vars``.
    Leave-one-out reduction runs on dual (unconstrained) scores; the returned
    ``wgf_grad`` / ``prior_grad`` are then scaled at runtime by
    :meth:`~pymc_prop.points.PointMapper.primal_scale`
    (:math:`\\exp(-\\texttt{log\\_jac\\_det})` via the mapper's ``ljd_fn``;
    identity → ones) so the Wasserstein gradient flow uses constrained-space
    gradients :math:`\\nabla_\\theta` laid out in unconstrained flat coordinates
    (Gu & Kim 2025, §2.1). Prior ``jacobian`` then defaults to ``False`` so
    ``model.logp`` is the constrained-space potential, not the change-of-variables
    density. Diffusion scaling is applied in
    :func:`~pymc_prop.particles.time_step`.

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
        PyMC change-of-variables switch on the prior logp. ``None`` (default)
        uses ``False`` when any free RV has a transform, else ``True``.

    Returns
    -------
    wgf_grad
        Interaction drift, shape ``(n_particles, n_params)`` (constrained-space
        gradient laid out in unconstrained flat coordinates).
    prior_grad
        Prior score gradient, same layout.

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

    require_mirror_compatible_transforms(model)
    if jacobian is None:
        # jacobian=False: constrained-space prior potential (mirror Langevin)
        jacobian = not mapper.has_transforms

    particles = pt.matrix("particles")

    def build(use_scan: bool) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
        logp, score = _batched_observed_logp_score_graph(particles, model, mapper, use_scan=use_scan)
        prior_grad = _batched_prior_grad_graph(
            particles, model, mapper, jacobian_terms=jacobian, use_scan=use_scan
        )
        return logp, score, prior_grad

    logp, score, prior_grad = _try_vectorize_then_scan(build)

    # subtract particle-wise max before exponentiating to prevent overflow/underflow
    logp_max = pt.max(logp, axis=0, keepdims=True)
    exp_shifted = pt.exp(logp - logp_max)

    # leave-one-particle-out density: Σ_{ℓ≠j} p_{ϑ^{(ℓ)}}
    sum_all = pt.sum(exp_shifted, axis=0, keepdims=True)
    sum_excl = pt.maximum(sum_all - exp_shifted, eps)

    denom = pt.cast(particles.shape[0] - 1, logp.dtype)
    log_mix = logp_max + pt.log(sum_excl) - pt.log(denom)  # (1/(p-1)) Σ_{ℓ≠j} p_{ϑ^{(ℓ)}}

    # log importance weights vs mixture for chain rule, clipped for stability
    log_ratio_raw = logp - log_mix
    # symmetric cap on log w_{i,j} before exp (stability)
    log_ratio = pt.clip(log_ratio_raw, -log_ratio_clip, log_ratio_clip)
    ratio = pt.exp(log_ratio)

    # − Σ_i w_{i,j} ∇_ϑ log p(y_i|ϑ^{(j)}) on dual scores; primal scale at runtime
    wgf_grad = -pt.sum(ratio[:, :, None] * score, axis=1)

    raw_fn = model.compile_fn(
        inputs=[particles],
        outs=[wgf_grad, prior_grad],
        point_fn=False,
        on_unused_input="ignore",
    )

    def drift(particles_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wgf_out, prior_out = raw_fn(particles_np)
        scale = mapper.primal_scale(particles_np)
        return np.asarray(wgf_out, dtype=float) * scale, np.asarray(
            prior_out, dtype=float
        ) * scale

    return drift
