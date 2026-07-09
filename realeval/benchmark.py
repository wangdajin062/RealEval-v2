"""realeval/benchmark.py — Thin Entry for Forward Benchmarking (under 100 lines)

Delegates to:
  - runner.py   (inference, GPU monitoring, CUDA Graph)
  - metrics.py  (metric computation)
  - statistics.py (mean, std, ci95)
  - report.py  (CSV/LaTeX/PNG export)

Only orchestrates, never computes metrics or statistics.
"""
from __future__ import annotations
import csv
import logging
from pathlib import Path

from realeval.runner import run_forward_benchmark, best_batch_size

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "metrics"

logger = logging.getLogger("benchmark")


def benchmark(model, sample_input, *, warmup=10, repeat=100, batch_sizes=(1,),
              use_cuda_graph=False, device=None, save_csv=True):
    """Run forward benchmark, optionally save CSV. Delegates all measurement to runner.py."""
    results = run_forward_benchmark(
        model, sample_input,
        warmup=warmup, repeat=repeat, batch_sizes=batch_sizes,
        use_cuda_graph=use_cuda_graph, device=device)

    if save_csv and results:
        _write_benchmark_csv(results)

    return results


def _write_benchmark_csv(results: dict):
    """Save benchmark results to outputs/metrics/benchmark.csv."""
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["batch_size", "throughput_sps", "latency_p50_ms",
            "latency_p90_ms", "latency_p99_ms", "peak_mem_mb", "gpu_util_pct",
            "gpu_power_w", "energy_j", "wall_s", "cuda_graph", "device"]
    with open(OUT / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for bs, r in results.items():
            w.writerow({"batch_size": bs, **{k: r.get(k) for k in keys[1:]}})
    logger.info("Benchmark CSV saved to %s", OUT / "benchmark.csv")


def summary(results: dict) -> dict:
    """Return best batch size summary. Delegates to runner.best_batch_size()."""
    return best_batch_size(results)
