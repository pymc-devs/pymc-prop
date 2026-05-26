"""Diagnostics helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np


def pairwise_distance_stats(particles: np.ndarray) -> Dict[str, float]:
    diff = particles[:, None, :] - particles[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    dists = dists[np.triu_indices_from(dists, k=1)]
    return {
        "pairwise_min": float(np.min(dists)),
        "pairwise_med": float(np.median(dists)),
        "pairwise_max": float(np.max(dists)),
    }
