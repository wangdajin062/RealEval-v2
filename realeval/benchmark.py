"""realeval/benchmark.py — Full-Metric Benchmark (Latency/Throughput/GPU Util/Mem/Power/Energy) + benchmark.csv

Real measurements:
  - latency P50/P90/P99, throughput (samples/sec)
  - peak_memory (torch), GPU util/power (nvidia-smi concurrent sampling), energy (power x time integral)
  - optional CUDA Graph capture for forward pass
Under sandbox CPU, GPU-related metrics degrade to None, latency/throughput remain real.
Generates outputs/benchmark.csv.
"""
from __future__ import annotations
import csv
import logging
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

logger = logging.getLogger("benchmark")


def _sync(dev):
    if getattr(dev, "type", str(dev)) == "cuda":
        import torch
        torch.cuda.synchronize()


def _sample_gpu():
    """Instantaneous GPU util/power (nvidia-smi). Returns None without GPU."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"], timeout=2).decode().strip()
        utils, powers, mems = [], [], []
        for line in out.splitlines():
            u, p, m = [x.strip() for x in line.split(",")]
            utils.append(float(u)); powers.append(float(p)); mems.append(float(m))
        if not utils:
            return None
        return {"util_pct": sum(utils) / len(utils), "power_w": sum(powers), "mem_used_mb": sum(mems)}
    except Exception:
        return None


def benchmark_forward(model, sample_input, *, warmup=10, repeat=100, batch_sizes=(1,),
                      use_cuda_graph=False, device=None):
    """Full-metric forward pass benchmark with optional CUDA Graph."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    results = {}

    for bs in batch_sizes:
        x = sample_input.to(dev)
        x = x.unsqueeze(0).repeat(bs, *([1] * x.dim()))
        if getattr(dev, "type", str(dev)) == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
        _sync(dev)

        graphed = _try_cuda_graph(model, x) if (use_cuda_graph and getattr(dev, "type", str(dev)) == "cuda") else None

        gpu_samples = []
        lat = []
        t_wall0 = time.perf_counter()
        with torch.no_grad():
            for i in range(repeat):
                t0 = time.perf_counter()
                (graphed["replay"]() if graphed else model(x))
                _sync(dev)
                lat.append((time.perf_counter() - t0) * 1000)
                if getattr(dev, "type", str(dev)) == "cuda" and i % max(1, repeat // 10) == 0:
                    g = _sample_gpu()
                    if g:
                        gpu_samples.append(g)
        wall_s = time.perf_counter() - t_wall0

        lat = np.array(lat)
        total_samples = bs * repeat
        peak_mem = (torch.cuda.max_memory_allocated() / 1e6) if getattr(dev, "type", str(dev)) == "cuda" else None
        util = (np.mean([s["util_pct"] for s in gpu_samples]) if gpu_samples else None)
        power = (np.mean([s["power_w"] for s in gpu_samples]) if gpu_samples else None)
        energy = (power * wall_s if power is not None else None)  # Joules = W x s
        results[bs] = {
            "throughput_sps": round(total_samples / wall_s, 2) if wall_s > 0 else None,
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 3),
            "latency_p90_ms": round(float(np.percentile(lat, 90)), 3),
            "latency_p99_ms": round(float(np.percentile(lat, 99)), 3),
            "peak_mem_mb": round(peak_mem, 1) if peak_mem is not None else None,
            "gpu_util_pct": round(float(util), 1) if util is not None else None,
            "gpu_power_w": round(float(power), 1) if power is not None else None,
            "energy_j": round(float(energy), 1) if energy is not None else None,
            "wall_s": round(wall_s, 3),
            "cuda_graph": graphed is not None,
            "device": dev,
        }
    _write_csv(results)
    return results


def _write_csv(results: dict):
    if not results:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["batch_size", "throughput_sps", "latency_p50_ms",
            "latency_p90_ms", "latency_p99_ms", "peak_mem_mb", "gpu_util_pct",
            "gpu_power_w", "energy_j", "wall_s", "cuda_graph", "device"]
    with open(OUT / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for bs, r in results.items():
            w.writerow({"batch_size": bs, **{k: r.get(k) for k in keys[1:]}})


def _try_cuda_graph(model, x):
    """Capture CUDA Graph for the forward pass. Returns None on failure."""
    import torch
    try:
        static_x = x.clone().to("cuda") if x.device.type != "cuda" else x.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    model(static_x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                model(static_x)
        return {"graph": g, "static_x": static_x, "replay": lambda: g.replay()}
    except Exception:
        return None


def benchmark_summary(results: dict) -> dict:
    """Summarise benchmark results: best batch size by throughput."""
    if not results:
        return {}
    best_bs = max(results, key=lambda b: results[b].get("throughput_sps") or 0)
    return {"best_batch_size": best_bs, **results[best_bs], "all_batch_sizes": results}
