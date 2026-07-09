"""realeval/runlog.py — Experiment Logging with Automatic Provenance Recording

Logs experiment start/end, parameters, and auto-records reproducibility metadata:
  - git commit hash
  - environment (Python, torch, CUDA, driver versions)
  - config hash (SHA256 of sorted config dict)
  - seed
  - timestamp
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RUNLOG = OUT / "runlog.jsonl"

logger = logging.getLogger("runlog")


def _git_commit() -> str | None:
    """Return the current git commit SHA, or None if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], timeout=5, stderr=subprocess.DEVNULL,
            cwd=str(ROOT)
        ).decode().strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], timeout=5, stderr=subprocess.DEVNULL,
            cwd=str(ROOT)
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return None


def _env_snapshot() -> dict:
    """Capture environment: Python, torch, CUDA, driver versions."""
    info = {}
    import sys
    info["python_version"] = sys.version.split()[0]
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        info["torch_version"] = "not_installed"
    # NVIDIA driver
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()
        info["nvidia_driver"] = drv[0] if drv else None
    except Exception:
        info["nvidia_driver"] = None
    return info


def _config_hash(config: dict) -> str:
    """SHA256 of sorted, JSON-serialized config (stable across runs)."""
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _data_hash(data_paths: list[str] | None = None) -> dict:
    """SHA256 of dataset files. Returns {path: hexdigest[:16]}."""
    if not data_paths:
        return {}
    hashes = {}
    for p in data_paths:
        path = Path(p)
        if not path.exists():
            hashes[str(path)] = None
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        hashes[str(path)] = h.hexdigest()[:16]
    return hashes


def collect_provenance(config: dict | None = None) -> dict:
    """Collect full reproducibility provenance record.

    Returns a dict with: timestamp, git_commit, git_dirty, env, config_hash, seed.
    """
    prov = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "env": _env_snapshot(),
    }
    if config:
        prov["config_hash"] = _config_hash(config)
        prov["seed"] = config.get("reproducibility", {}).get("seed")
    return prov


def log_experiment_start(exp_name: str, config: dict = None):
    """Log the start of an experiment with provenance."""
    logger.info("=== Experiment %s started ===", exp_name)
    prov = collect_provenance(config)
    logger.info("Provenance: git=%s dirty=%s config_hash=%s seed=%s",
                prov.get("git_commit", "?")[:8] if prov.get("git_commit") else "?",
                prov.get("git_dirty", "?"),
                prov.get("config_hash", "?"),
                prov.get("seed", "?"))
    if config:
        logger.info("Config: %s", config)


def log_experiment_end(exp_name: str, result: dict = None):
    """Log the completion of an experiment."""
    logger.info("=== Experiment %s completed ===", exp_name)
    if result:
        logger.info("Result keys: %s", list(result.keys()))


def log_run(experiment: str, config: dict, result: dict, status: str = "completed",
            data_paths: list[str] | None = None):
    """Append a run record to runlog.jsonl with full provenance.

    Args:
        experiment: Experiment name (e.g. 'exp1').
        config: Experiment configuration dict.
        result: Experiment result dict.
        status: 'completed' or 'failed'.
        data_paths: Optional list of dataset file paths for hashing.
    """
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    prov = collect_provenance(config)
    if data_paths:
        prov["data_hashes"] = _data_hash(data_paths)
    record = {
        "provenance": prov,
        "experiment": experiment,
        "status": status,
        "config": config,
        "result": result,
    }
    with open(RUNLOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
