"""realeval/io.py — Config Loading + Result Saving

Config loading: YAML config with environment variable override (REALEVAL_DATA__KEY__SUBKEY).
Result saving: JSON to outputs/results/{exp_short}_{timestamp}.json.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results"


def _ensure_results_dir() -> Path:
    """Lazily create the results directory on first use."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def _resolve_env_overrides(config: dict, prefix: str = "REALEVAL_") -> dict:
    """Apply REALEVAL_ prefixed environment variable overrides (RealEval-v2 contract).
    Values are parsed via yaml.safe_load for proper type inference."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        keys = env_key[len(prefix):].lower().split("__")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            d[keys[-1]] = yaml.safe_load(env_val)
        except Exception:
            d[keys[-1]] = env_val
    return config


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` onto `base` (over wins on leaf conflicts)."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = None) -> dict:
    """Load YAML config, apply env overrides (REALEVAL_DATA__KEY__SUBKEY).

    The base is always config/experiments.yaml. If `path` points to a different file (e.g.
    config/h100.yaml), it is deep-merged ON TOP of the base so overlays only need to specify the
    keys they change (hardware/runtime/output) without duplicating models/data.
    """
    base_path = ROOT / "config" / "experiments.yaml"
    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path is not None and Path(path).resolve() != base_path.resolve():
        with open(path, encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overlay)
    return _resolve_env_overrides(cfg)


def save_results(exp_short: str, result: dict) -> Path:
    """Save result JSON to outputs/results/{exp_short}_{timestamp}.json. Returns Path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _ensure_results_dir() / f"{exp_short}_{ts}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
