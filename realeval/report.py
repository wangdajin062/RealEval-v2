"""realeval/report.py — Export Only (CSV, LaTeX, PNG)

This module does NOT compute metrics or statistics.
It only reads pre-computed result dicts and formats them for output.
"""
from __future__ import annotations
import csv
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("report")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
RESULTS = OUT / "results"
METRICS = OUT / "metrics"


def _latest_results() -> dict[str, dict]:
    """Read latest result for each experiment (sorted by timestamp)."""
    if not RESULTS.is_dir():
        return {}
    by_exp = {}
    for f in sorted(RESULTS.glob("*.json")):
        exp = f.stem.rsplit("_", 2)[0]
        try:
            by_exp[exp] = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping corrupt result file %s: %s", f.name, e)
            continue
    return by_exp


def _fmt(v):
    """Format for table display."""
    return "-" if v is None else str(v)


def _flatten_metrics(r: dict, prefix: str = "", depth: int = 0):
    """Recursively extract scalar metrics from nested result dicts (max depth 3)."""
    if depth > 3:
        return
    for k, v in r.items():
        if k in ("experiment", "computation", "path", "is_synthetic"):
            continue
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (dict, list)):
            if isinstance(v, dict):
                yield from _flatten_metrics(v, full_key, depth + 1)
        else:
            yield (full_key, v)


def build_summary_csv() -> Path:
    """Export summary.csv from pre-computed results. No metric computation."""
    METRICS.mkdir(parents=True, exist_ok=True)
    results = _latest_results()
    rows = []
    for exp in sorted(results):
        r = results[exp]
        base = {"experiment": r.get("experiment", exp), "computation": r.get("computation", "?")}
        found = False
        for metric_name, value in _flatten_metrics(r):
            row = dict(base)
            row["metric"] = metric_name
            row["value"] = value
            rows.append(row)
            found = True
        if not found:
            # No scalar metrics found; emit a placeholder so the experiment is not missing
            row = dict(base)
            row["metric"] = "-"
            row["value"] = "-"
            rows.append(row)
    with open(METRICS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "computation", "metric", "value"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return METRICS / "summary.csv"


def build_paper_tables() -> Path:
    """Export tables.md from pre-computed results."""
    TABLES.mkdir(parents=True, exist_ok=True)
    res = _latest_results()
    lines = ["# Paper Tables (auto-generated from real experiment results)\n"]
    for exp in sorted(res):
        r = res[exp]
        lines.append(f"\n## {exp}: {r.get('experiment', exp)}\n")
        lines.append(f"- Computation: {r.get('computation', '?')}\n")
        scalars = {k: v for k, v in r.items()
                   if not isinstance(v, (dict, list)) and k not in ("experiment", "computation", "path")}
        if scalars:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in scalars.items():
                lines.append(f"| {k} | {_fmt(v)} |")
    (TABLES / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    return TABLES / "tables.md"
def build_latex_tables() -> Path:
    """Export LaTeX table (Table2.tex) from pre-computed results."""
    TABLES.mkdir(parents=True, exist_ok=True)
    res = _latest_results()
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{TAF-28k main results}", r"\label{tab:main}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Method & F1 & Computation \\", r"\midrule",
    ]
    e1, e4 = res.get("exp1", {}), res.get("exp4", {})
    for name, v in e4.get("classifiers", {}).items():
        f1 = v.get("f1") if isinstance(v, dict) else v
        lines.append(f"{name} & {_fmt(f1)} & real \\\\")
    if e1.get("f1") is not None:
        lines.append(f"QAD+OV-Freeze & {_fmt(e1['f1'])} & real \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    (TABLES / "Table2.tex").write_text(tex + "\n", encoding="utf-8")
    return TABLES / "Table2.tex"
def build_paper_figures(fmt: str = "png") -> list:
    """Build figures from pre-computed results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figures")
        return []

    def _save(fig, name: str, fmt: str = "png"):
        """Save a matplotlib figure."""
        FIGS.mkdir(parents=True, exist_ok=True)
        path = FIGS / f"{name}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    res = _latest_results()
    made = []
    e8 = res.get("exp8", {})
    lat = e8.get("latencies", {})
    if lat:
        fig, ax = plt.subplots(figsize=(6, 4))
        schemes = list(lat.keys())
        vals = [lat[s] for s in schemes]
        ax.bar(schemes, vals, color=["steelblue", "indianred", "green"][:len(schemes)])
        ax.set_ylabel("Latency (s)")
        ax.set_title("Inference Latency by Quantization")
        for s, v in zip(schemes, vals):
            ax.text(s, v, f"{v:.3f}s", ha="center", va="bottom")
        made.append(str(_save(fig, "fig_latency_benchmark", fmt)))
    return made


def build_pdf_figures() -> list:
    """Vector (PDF) versions of paper figures."""
    return build_paper_figures(fmt="pdf")


def build_all() -> list:
    """Run all export generators."""
    made = []
    made.append(str(build_summary_csv()))
    made.append(str(build_paper_tables()))
    made.extend(build_paper_figures())
    made.append(str(build_latex_tables()))
    try:
        pdfs = build_pdf_figures()
        made.extend(pdfs)
    except Exception as e:
        logger.warning("PDF figure generation failed: %s", e)
    return made
