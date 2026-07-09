"""realeval/envreport.py — Environment Report (Hardware/Driver/Torch Version)

Collects and writes environment information to outputs/env_report.json + environment_report.md
for reproduction records. Output path is configurable via the REALEVAL_OUTPUT_ROOT environment variable.
"""
from __future__ import annotations
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_ROOT = Path(os.environ.get("REALEVAL_OUTPUT_ROOT", str(ROOT / "outputs")))


def output_root() -> Path:
    """Get the configured output root directory."""
    return _OUTPUT_ROOT


def _run(cmd) -> str:
    try:
        return subprocess.check_output(cmd, timeout=5, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _cpu_count():
    try:
        return os.cpu_count()
    except Exception:
        return None


def _cudnn_version():
    try:
        import torch
        v = torch.backends.cudnn.version()
        return str(v) if v else None
    except Exception:
        return None


def _nvidia_driver():
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0] if out else None


def _nvidia_smi_brief():
    return _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader"]) or None


def _pkg_versions(names):
    import importlib.metadata as md
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = "not_installed"
    return out


def collect() -> dict:
    """Collect environment information into a structured dict (JSON + Markdown compatible)."""
    from realeval import hwenv
    env = hwenv.detect(verbose=False)
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "os": {"system": platform.system(), "release": platform.release(),
               "machine": platform.machine(), "platform": platform.platform()},
        "cpu": {"processor": platform.processor() or _run(["uname", "-p"]),
                "count": _cpu_count()},
        "torch": {"version": env.get("torch_version"),
                  "cuda_version": env.get("cuda_version"),
                  "cudnn_version": _cudnn_version()},
        "gpu": {"cuda_available": env.get("cuda_available"), "n_gpus": env.get("gpu_count"),
                "names": env.get("gpu_names"), "is_h100": env.get("is_h100"),
                "compute_capability": env.get("compute_capability"),
                "driver_version": _nvidia_driver(), "nvidia_smi": _nvidia_smi_brief()},
        "distributed": {"nccl_available": env.get("nccl_available"),
                        "nccl_version": env.get("nccl_version")},
        "optimizations": {"bf16_supported": env.get("bf16_supported"),
                          "torch_compile_available": env.get("torch_compile_available"),
                          "flash_attn_available": env.get("flash_attn_available")},
        "key_packages": _pkg_versions(["transformers", "accelerate", "bitsandbytes",
                                       "scikit-learn", "numpy", "matplotlib"]),
    }
    return info


def write_report() -> dict:
    """Collect environment info and write to env_report.json + environment_report.md."""
    info = collect()
    out_dir = output_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    (out_dir / "env_report.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Markdown output
    g = info["gpu"]
    cc = g.get("compute_capability") or []
    cc_str = ",".join(f"{c[0]}.{c[1]}" if isinstance(c, (list, tuple)) and len(c) == 2 else str(c) for c in cc)
    md_lines = [
        "# Environment Report\n",
        f"Generated: {info['timestamp']}\n",
        "## Hardware\n",
        f"- CPU: {info['cpu']['processor']} ({info['cpu']['count']} cores)",
        f"- GPU: {'/'.join(set(g['names'])) if g['names'] else 'None'} "
        f"({g['n_gpus']} devices, H100={g['is_h100']}, CC={cc_str or '—'})",
        f"- NVIDIA Driver: {g['driver_version'] or '—'}",
        "\n## Software Stack\n",
        f"- OS: {info['os']['platform']}",
        f"- Python: {info['python']['version']}",
        f"- PyTorch: {info['torch']['version']} (CUDA {info['torch']['cuda_version']}, "
        f"cuDNN {info['torch']['cudnn_version']})",
        "\n## Optimization Capabilities\n",
        f"- BF16: {info['optimizations']['bf16_supported']}",
        f"- torch.compile: {info['optimizations']['torch_compile_available']}",
        f"- FlashAttention: {info['optimizations']['flash_attn_available']}",
        "\n## Key Package Versions\n",
        "| Package | Version |", "|---|---|",
    ]
    for k, v in info["key_packages"].items():
        md_lines.append(f"| {k} | {v} |")
    (out_dir / "environment_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    return info
