"""Uncertainty for retrieval metrics.

Queries are the unit of sampling. Two systems are always compared on the same
queries, so the tests are paired: they work on per-query differences.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Confidence interval for mean(a - b), resampling queries."""
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)


def paired_randomization_test(
    a: np.ndarray, b: np.ndarray, *, n_perm: int = 10_000, seed: int = 0
) -> float:
    """Two-sided paired randomization (sign-flip) test for mean(a) != mean(b).

    Under the null hypothesis the two systems are exchangeable on each query,
    so the sign of each per-query difference is a coin flip. The p-value is the
    share of random sign assignments whose mean difference is at least as far
    from zero as the observed one. The +1 terms keep the p-value above zero
    (Phipson and Smyth, 2010).
    """
    diff = a - b
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(diff)))
    perm = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(perm >= observed - 1e-12) + 1) / (n_perm + 1))


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment for a family of p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted
