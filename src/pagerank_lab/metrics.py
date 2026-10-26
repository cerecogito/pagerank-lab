"""Comparing two PageRank vectors: numerical error vs ranking agreement.

* mean absolute difference (MAD) measures how far the probabilities are apart;
* Spearman's rho measures whether the two vectors ORDER the pages the same way.
They answer different questions. Many pages with nearly equal rank can be
reordered by tiny numerical noise (low rho, tiny MAD), and a vector can be far
off numerically yet keep the same order (rho = 1, large MAD).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


def aligned_vectors(a: dict[str, float], b: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    if set(a) != set(b):
        raise ValueError("the two rank vectors cover different pages")
    names = sorted(a)
    return np.array([a[n] for n in names]), np.array([b[n] for n in names])


def mean_absolute_difference(a: dict[str, float], b: dict[str, float]) -> float:
    x, y = aligned_vectors(a, b)
    return float(np.mean(np.abs(x - y)))


def max_absolute_difference(a: dict[str, float], b: dict[str, float]) -> float:
    x, y = aligned_vectors(a, b)
    return float(np.max(np.abs(x - y)))


def spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Spearman's rho with ties given their average rank; None when undefined.

    rho is the Pearson correlation of the two rank vectors. If either vector is
    constant (all pages tied) its ranks have zero variance and rho is undefined.
    We return None then. We never return 0 or 1 as a substitute.
    """
    x, y = aligned_vectors(a, b)
    if x.size < 2:
        return None
    rank_x, rank_y = rankdata(x, method="average"), rankdata(y, method="average")
    centred_x, centred_y = rank_x - rank_x.mean(), rank_y - rank_y.mean()
    denominator = math.sqrt(float(np.dot(centred_x, centred_x)) * float(np.dot(centred_y, centred_y)))
    if denominator == 0.0:
        return None
    return float(np.dot(centred_x, centred_y) / denominator)


def summarise_defined(values: list[float | None]) -> dict:
    """Median/min/max over the defined values only; counts the undefined ones separately."""
    defined = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return {
        "defined": len(defined), "undefined": len(values) - len(defined),
        "median": float(np.median(defined)) if defined else None,
        "min": float(np.min(defined)) if defined else None,
        "max": float(np.max(defined)) if defined else None,
    }
