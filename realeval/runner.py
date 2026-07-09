"""realeval/runner.py — GPU Benchmark Runner (Inference, CUDA Graph, GPU Monitoring)

Responsibilities:
  - Load model and run inference
  - Measure latency with time.perf_counter()
  - Sample GPU utilization/power via nvidia-smi
  - Capture CUDA Graph for forward pass
  - Return real measured results (no fallback, no mock)

This module does NOT compute metrics, statistics, or generate reports.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("runner")


def sync_device(dev):
    """CUDA synchronize if device is CUDA."""
    if getattr(dev, "type", str(dev)) == "cuda":
        import torch
        torch.cuda.synchronize()


def sample_gpu():
    """Instantaneous GPU util/power (nvidia-smi). Returns None without GPU."""
    import subprocess
    try:
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


def try_cuda_graph(model, x):
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


def run_forward_benchmark(model, sample_input, *, warmup=10, repeat=100, batch_sizes=(1,),
                          use_cuda_graph=False, device=None):
    """Full-metric forward pass benchmark with real measurements.

    Args:
        model: PyTorch model (in eval mode).
        sample_input: Single sample tensor.
        warmup: Number of warmup iterations.
        repeat: Number of measured iterations.
        batch_sizes: Tuple of batch sizes to test.
        use_cuda_graph: If True, attempt CUDA Graph capture.
        device: Device string (e.g. 'cuda', 'cpu').

    Returns:
        dict mapping batch_size -> result dict with real measured values.
    """
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
        sync_device(dev)

        # CUDA Graph capture
        graphed = try_cuda_graph(model, x) if use_cuda_graph else None

        gpu_samples = []
        lat = []
        t_wall0 = time.perf_counter()
        with torch.no_grad():
            for i in range(repeat):
                t0 = time.perf_counter()
                (graphed["replay"]() if graphed else model(x))
                sync_device(dev)
                lat.append((time.perf_counter() - t0) * 1000)
                if getattr(dev, "type", str(dev)) == "cuda" and i % max(1, repeat // 10) == 0:
                    g = sample_gpu()
                    if g:
                        gpu_samples.append(g)
        wall_s = time.perf_counter() - t_wall0

        lat = np.array(lat)
        total_samples = bs * repeat
        peak_mem = (torch.cuda.max_memory_allocated() / 1e6) if getattr(dev, "type", str(dev)) == "cuda" else None
        util = (np.mean([s["util_pct"] for s in gpu_samples]) if gpu_samples else None)
        power = (np.mean([s["power_w"] for s in gpu_samples]) if gpu_samples else None)
        energy = (power * wall_s if power is not None else None)

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
    return results


def best_batch_size(results: dict) -> dict:
    """Return the batch size with highest throughput from benchmark results."""
    if not results:
        return {}
    best_bs = max(results, key=lambda b: results[b].get("throughput_sps") or 0)
    return {"best_batch_size": best_bs, **results[best_bs], "all_batch_sizes": results}
