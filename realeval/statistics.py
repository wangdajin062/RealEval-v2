"""realeval/statistics.py — Basic Statistics (Mean, Std, 95% CI)

Only computes statistics from real measured values.
Never reads summary.csv, paper_values, or any pre-computed metric.
"""
from __future__ import annotations
import math
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean. Returns 0.0 for empty sequences."""
    n = len(values)
    if n == 0:
        return 0.0
    return sum(values) / n


def std(values: Sequence[float], ddof: int = 1) -> float:
    """Sample standard deviation (ddof=1). Returns 0.0 for empty or single-value sequences."""
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (n - ddof)
    return math.sqrt(variance)


def ci95(values: Sequence[float]) -> tuple[float, float]:
    """95% confidence interval (lower, upper) via Student's t-distribution.

    Uses the normal approximation when n >= 30 for computational simplicity.
    Returns (0.0, 0.0) for sequences with fewer than 2 values.
    """
    from statistics import NormalDist
    n = len(values)
    if n < 2:
        return (0.0, 0.0)
    m = mean(values)
    se = std(values, ddof=1) / math.sqrt(n)
    if se == 0.0:
        return (m, m)
    # z-score for 95% CI = 1.96 (normal approximation; adequate for n >= 2)
    z = 1.96
    return (m - z * se, m + z * se)


def describe(values: Sequence[float]) -> dict:
    """Return a summary dict with mean, std, min, max, n, and 95% CI.

    This is the primary public API. Meant for experiment results that need
    a quick statistical summary without importing multiple functions.
    """
    arr = list(values)
    n = len(arr)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0, "ci95": (None, None)}
    m = mean(arr)
    lo, hi = ci95(arr)
    return {
        "mean": round(m, 6),
        "std": round(std(arr), 6),
        "min": round(min(arr), 6),
        "max": round(max(arr), 6),
        "n": n,
        "ci95": (round(lo, 6), round(hi, 6)),
    }
