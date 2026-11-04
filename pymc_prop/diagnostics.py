from typing import Callable, Optional, Tuple

import numpy as np


def get_bandwidth(
    sq_dists: np.ndarray, bandwidth: Optional[float] = None
) -> float:
    """Return the kernel denominator ``h``.

    An explicit ``bandwidth`` is used directly. Otherwise, ``h`` is selected
    with the median heuristic ``median(||x_i - x_j||^2) / log(N)`` using the
    off-diagonal pairwise squared distances.
    """
    if bandwidth is not None:
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be finite and positive.")
        return float(bandwidth)

    n_particles = sq_dists.shape[0]
    off_diag = sq_dists[np.triu_indices(n_particles, k=1)]
    median_sq = np.median(off_diag) if off_diag.size else 0.0
    bandwidth = median_sq / np.log(n_particles) if median_sq > 0 else 1.0
    return max(float(bandwidth), 1e-12)

def rbf_kernel(
    sq_dists: np.ndarray, d: int, bandwidth: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes components for the Gaussian RBF kernel
    ``K(x, y) = exp(-||x-y||^2 / bandwidth)``.

    Returns: (K, cross_coeff, trace_term)
    """
    K = np.exp(-sq_dists / bandwidth)
    
    # -2 * f'(r^2)
    cross_coeff = (2.0 / bandwidth) * K
    # -2d * f'(r^2) - 4r^2 * f''(r^2)
    trace_term = K * (
        (2.0 * d / bandwidth) - (4.0 / bandwidth**2) * sq_dists
    )
    
    return K, cross_coeff, trace_term

def imq_kernel(
    sq_dists: np.ndarray, d: int, bandwidth: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes components for the inverse multi-quadric kernel
    ``K(x, y) = (1 + ||x-y||^2 / bandwidth)^(-1/2)``.

    Returns: (K, cross_coeff, trace_term)
    """
    K = (1.0 + sq_dists / bandwidth) ** (-0.5)
    
    # Precompute powers of K to avoid repeated expensive operations
    K2 = K*K
    K3 = K2*K
    K5 = K2*K3
    
    # -2 * f'(r^2)
    cross_coeff = (1.0 / bandwidth) * K3
    # -2d * f'(r^2) - 4r^2 * f''(r^2)
    trace_term = (d / bandwidth) * K3 - (3.0 / bandwidth**2) * K5 * sq_dists
    
    return K, cross_coeff, trace_term

def compute_kgd(
    particles: np.ndarray,
    score: np.ndarray,
    kernel_fn: Callable = imq_kernel,
    *,
    bandwidth: float,
) -> float:
    """Estimate KGD from particles and their target score.

    ``bandwidth`` is the positive denominator used directly by ``kernel_fn``.
    Bandwidth-selection policy belongs to the caller; use
    :func:`get_bandwidth` before calling this function when the median
    heuristic is desired. ``score`` must be ``grad(log q)`` evaluated at the
    corresponding rows of ``particles``.
    """
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive.")

    N, d = particles.shape
    if N <= 1:
        return 0.0
        
    # diffs shape: (N, N, d) -> diffs[j, l] = x_j - x_l
    diffs = particles[:, None, :] - particles[None, :, :]  
    sq_dists = np.sum(diffs ** 2, axis=-1)                
    
    K, cross_coeff, trace_term = kernel_fn(sq_dists, d, bandwidth)
    
    term1 = (score @ score.T) * K
    
    # The Cross Terms
    score_diffs = score[:, None, :] - score[None, :, :]
    cross_dot = np.sum(score_diffs * diffs, axis=-1)
    
    term23 = cross_coeff * cross_dot 
    
    #  Combine into the full Stein Kernel matrix (H)
    H = term1 + term23 + trace_term

    kgd_sq = np.mean(H)
    
    return float(np.sqrt(max(kgd_sq, 0.0)))
