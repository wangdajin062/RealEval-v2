"""statistics/stats.py — Statistical Analysis for Experimental Results

Every experiment outputs:
  - mean, std
  - 95% bootstrap confidence interval
  - effect size (Cohen's d, Cliff's delta)
  - significance test (paired t-test, Wilcoxon)
  - p-value

Not just a single F1 number.
"""
from __future__ import annotations
import numpy as np
from typing import Any


def bootstrap_ci(samples: list[float], n_bootstrap: int = 10000, ci: float = 0.95,
                 seed: int = 42) -> dict:
    """Bootstrap confidence interval for the mean.

    Args:
        samples: Raw measurements.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95).

    Returns:
        {mean, std, ci_lower, ci_upper, n_samples}
    """
    arr = np.asarray(samples)
    n = len(arr)
    if n < 3:
        return {"mean": float(np.mean(arr)) if n > 0 else 0.0, "std": float(np.std(arr)) if n > 1 else 0.0,
                "ci_lower": None, "ci_upper": None, "n": n, "note": "insufficient samples for bootstrap"}
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        means.append(float(np.mean(arr[idx])))
    means = np.sort(means)
    alpha = (1 - ci) / 2
    lo = int(alpha * n_bootstrap)
    hi = int((1 - alpha) * n_bootstrap)
    return {"mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr, ddof=1)), 6),
            "ci_lower": round(float(means[lo]), 6),
            "ci_upper": round(float(means[hi]), 6),
            "ci_level": ci,
            "n": n}


def cohens_d(x: list[float], y: list[float]) -> dict:
    """Cohen's d effect size between two groups. |d| < 0.2: negligible, 0.5: medium, 0.8: large."""
    xa, ya = np.asarray(x), np.asarray(y)
    nx, ny = len(xa), len(ya)
    pooled_std = np.sqrt(((nx - 1) * np.var(xa, ddof=1) + (ny - 1) * np.var(ya, ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return {"cohens_d": 0.0, "magnitude": "none", "note": "zero variance"}
    d = (np.mean(xa) - np.mean(ya)) / pooled_std
    mag = "negligible" if abs(d) < 0.2 else ("small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large"))
    return {"cohens_d": round(float(d), 4), "magnitude": mag}


def cliffs_delta(x: list[float], y: list[float]) -> dict:
    """Cliff's delta: non-parametric effect size. Ranges [-1, 1], 0 = complete overlap."""
    xa, ya = np.asarray(x), np.asarray(y)
    nx, ny = len(xa), len(ya)
    greater = sum(1 for xi in xa for yj in ya if xi > yj)
    lesser = sum(1 for xi in xa for yj in ya if xi < yj)
    delta = (greater - lesser) / (nx * ny)
    mag = "negligible" if abs(delta) < 0.147 else ("small" if abs(delta) < 0.33 else ("medium" if abs(delta) < 0.474 else "large"))
    return {"cliffs_delta": round(float(delta), 4), "magnitude": mag}


def paired_ttest(x: list[float], y: list[float]) -> dict:
    """Paired t-test. Returns {statistic, p_value, significant (at 0.05)}."""
    from scipy import stats
    result = stats.ttest_rel(x, y)
    return {"test": "paired_t_test",
            "statistic": round(float(result.statistic), 4),
            "p_value": round(float(result.pvalue), 6),
            "significant_005": result.pvalue < 0.05}


def wilcoxon(x: list[float], y: list[float]) -> dict:
    """Wilcoxon signed-rank test (non-parametric paired test)."""
    from scipy import stats
    result = stats.wilcoxon(x, y)
    return {"test": "wilcoxon_signed_rank",
            "statistic": round(float(result.statistic), 4),
            "p_value": round(float(result.pvalue), 6),
            "significant_005": result.pvalue < 0.05}


def describe(samples: list[float]) -> dict:
    """Descriptive statistics for a single group."""
    arr = np.asarray(samples)
    n = len(arr)
    if n == 0:
        return {"n": 0}
    ci = bootstrap_ci(samples)
    return {
        "n": n,
        "mean": ci["mean"],
        "std": ci["std"],
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "ci_95": [ci["ci_lower"], ci["ci_upper"]],
    }


def compare_groups(baseline: list[float], treatment: list[float],
                   label_baseline: str = "baseline", label_treatment: str = "treatment") -> dict:
    """Full comparison: descriptive stats + effect size + significance test.

    Returns a comprehensive dict suitable for paper tables and evidence evaluation.
    """
    return {
        label_baseline: describe(baseline),
        label_treatment: describe(treatment),
        "cohens_d": cohens_d(baseline, treatment),
        "cliffs_delta": cliffs_delta(baseline, treatment),
        "paired_ttest": paired_ttest(baseline, treatment),
        "wilcoxon": wilcoxon(baseline, treatment),
    }


def summary_table(results: dict[str, list[float]]) -> str:
    """Generate a formatted summary table from {group_name: [samples]} dict."""
    lines = [f"{'Group':<20} {'n':>6} {'mean':>10} {'std':>10} {'CI 95%':>22}"]
    lines.append("-" * 72)
    for name, samples in results.items():
        d = describe(samples)
        if d.get("n", 0) > 0:
            ci = f"[{d['ci_95'][0]:.4f}, {d['ci_95'][1]:.4f}]" if d.get("ci_95") and d["ci_95"][0] else "N/A"
            lines.append(f"{name:<20} {d['n']:>6} {d['mean']:>10.4f} {d['std']:>10.4f} {ci:>22}")
    return "\n".join(lines)
