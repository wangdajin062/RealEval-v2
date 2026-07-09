"""realeval/statistics.py — Basic Statistics (Mean, Std, 95% CI) + research-grade functions

Only computes statistics from real measured values.
Never reads summary.csv, paper_values, or any pre-computed metric.
"""
from __future__ import annotations
import math
import numpy as np
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
    For n < 30, uses Student's t critical value (requires scipy; falls back to z=1.96).
    Returns (0.0, 0.0) for sequences with fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return (0.0, 0.0)
    m = mean(values)
    se = std(values, ddof=1) / math.sqrt(n)
    if se == 0.0:
        return (m, m)
    if n >= 30:
        z = 1.96  # normal approximation
    else:
        try:
            from scipy.stats import t as students_t
            z = float(students_t.ppf(0.975, n - 1))
        except ImportError:
            z = 1.96  # fallback if scipy unavailable
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


# ── Research-grade statistics (from V4) ──


def summarize(samples, n_boot=2000, ci=0.95, seed=0) -> dict:
    """Mean/std/n + bootstrap CI for a 1-D sample."""
    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    rng = np.random.RandomState(seed)
    if x.size == 1:
        m = float(x[0])
        return {"n": 1, "mean": m, "std": 0.0, "ci_low": m, "ci_high": m}
    boot = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "ci_low": float(lo), "ci_high": float(hi)}


def cohens_d(a, b) -> float:
    """Standardised mean difference (pooled SD) — effect size for two independent samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1] (probability of superiority)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (a.size * b.size))


def compare(a, b, paired=False) -> dict:
    """Compare two conditions: summaries, difference, p-value (t-test/Wilcoxon), and effect sizes."""
    from scipy import stats as st
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"a": summarize(a), "b": summarize(b),
           "mean_diff": float(a.mean() - b.mean()) if a.size and b.size else None}
    try:
        if paired and a.size == b.size and a.size >= 2:
            out["t_p"] = float(st.ttest_rel(a, b).pvalue)
            try:
                out["wilcoxon_p"] = float(st.wilcoxon(a, b).pvalue)
            except Exception:
                out["wilcoxon_p"] = None
        elif a.size >= 2 and b.size >= 2:
            out["t_p"] = float(st.ttest_ind(a, b).pvalue)
            out["mannwhitney_p"] = float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        out["t_p"] = None
    out["cohens_d"] = cohens_d(a, b)
    out["cliffs_delta"] = cliffs_delta(a, b)
    return out
