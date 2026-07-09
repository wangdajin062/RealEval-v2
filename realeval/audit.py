"""realeval/audit.py — Minimal Audit Log for Reproducibility

Only records:
  - GPU model/name
  - CUDA version
  - Torch version
  - NVIDIA Driver version
  - Dataset name(s)
  - Model name(s)
  - Random seed(s)

Writes to outputs/logs/audit.log (append). Does NOT record config details,
experiment results, or internal events — those have no reproduction value.
"""
from __future__ import annotations
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG = ROOT / "outputs" / "logs" / "audit.log"

_logger: logging.Logger | None = None


def reset_audit_logger():
    """Reset the audit logger singleton. Useful for test isolation."""
    global _logger
    if _logger is not None:
        for h in _logger.handlers[:]:
            _logger.removeHandler(h)
            h.close()
    _logger = None


def get_audit_logger() -> logging.Logger:
    """Get or create the audit logger singleton."""
    global _logger
    if _logger is not None:
        return _logger
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("audit")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = logging.FileHandler(AUDIT_LOG, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [AUDIT] %(message)s"))
    lg.addHandler(fh)
    lg.propagate = False
    _logger = lg
    return lg


def log_environment(config: dict = None):
    """Record reproduction-critical environment fields.

    Only logs: GPU, CUDA, Torch, Driver, Dataset, Model, Seed.
    """
    import os
    import platform
    import subprocess
    import sys

    lg = get_audit_logger()
    lg.info("=== Reproduction Environment ===")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            lg.info("GPU=%s count=%d", gpu_name, gpu_count)
        else:
            lg.info("GPU=None")
    except Exception:
        lg.info("GPU=unknown")

    # CUDA
    try:
        import torch
        cuda_ver = torch.version.cuda
        lg.info("CUDA=%s", cuda_ver or "unknown")
    except Exception:
        lg.info("CUDA=unknown")

    # Torch
    try:
        import torch
        lg.info("Torch=%s", torch.__version__)
    except Exception:
        lg.info("Torch=unknown")

    # NVIDIA Driver
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()
        if out:
            lg.info("Driver=%s", out[0])
    except Exception:
        lg.info("Driver=unknown")

    # Dataset
    if config and "data" in config:
        source = config["data"].get("source", "unknown")
        max_s = config["data"].get("max_samples", "all")
        lg.info("Dataset=source=%s max_samples=%s", source, max_s)
    else:
        lg.info("Dataset=unknown")

    # Model
    if config and "models" in config:
        teacher = config["models"].get("teacher", "unknown")
        student = config["models"].get("student", "unknown")
        lg.info("Model=teacher=%s student=%s", teacher, student)
    else:
        lg.info("Model=unknown")

    # Seed
    seed = os.environ.get("REALEVAL_SEED", "42")
    lg.info("Seed=%s", seed)
    lg.info("Python=%s", sys.version.split()[0])
    lg.info("Platform=%s", platform.platform())
    lg.info("=== End Reproduction Environment ===")
