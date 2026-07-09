"""realeval/paths.py — Data/Model/Cache Root Path Resolution (RunPod /workspace Persistent Volume Aware)

RunPod persistent volume is mounted at /workspace. This module resolves root directories with the following priority:
  1. Environment variables REALEVAL_DATA_ROOT / REALEVAL_MODELS_ROOT / HF_HOME (explicit)
  2. /workspace if it exists (RunPod persistent volume) -> /workspace/{data,models,hf_cache}
  3. Package-relative directories (sandbox/local fallback)

This way, placing data at /workspace/data and models at /workspace/models on RunPod is automatically discovered,
no config changes needed; REALEVAL_DATA_ROOT can also override to any mount point.
"""
from __future__ import annotations
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/workspace")


def _root(env_key: str, sub: str) -> Path:
    v = os.environ.get(env_key)
    if v:
        return Path(v)
    if WORKSPACE.is_dir():                      # RunPod persistent volume
        return WORKSPACE / sub
    return PKG_ROOT / sub


def data_root() -> Path:
    return _root("REALEVAL_DATA_ROOT", "data")


def models_root() -> Path:
    return _root("REALEVAL_MODELS_ROOT", "models")


def hf_cache() -> Path:
    v = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if v:
        return Path(v)
    if WORKSPACE.is_dir():
        return WORKSPACE / "hf_cache"
    return PKG_ROOT / ".hf_cache"


def resolve_data(rel_or_abs: str) -> Path:
    """Resolve config data relative path to actual root directory.

    Config values like 'data/TAF28k/...'; strip leading 'data/' then join to data_root().
    Path traversal protection: result must fall within data_root / workspace / package directory,
    otherwise rejected.
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        resolved = p
    else:
        parts = p.parts
        if parts and parts[0] == "data":
            resolved = data_root().joinpath(*parts[1:])
        else:
            resolved = data_root() / rel_or_abs
    try:
        from realeval.validation import safe_path, ValidationError
        allowed = [data_root(), WORKSPACE, PKG_ROOT]
        return safe_path(resolved, [a for a in allowed if a], must_exist=False)
    except ImportError:
        return resolved


def _has_model_files(d: Path) -> bool:
    """Check if directory looks like a real model (contains config.json or weight files)."""
    if not d.is_dir():
        return False
    if (d / "config.json").exists():
        return True
    for pat in ("*.safetensors", "*.bin", "*.gguf", "*.pt"):
        if any(d.glob(pat)):
            return True
    return False


def resolve_model(name_or_path: str) -> str:
    """Resolve model path: local directory or HF repo id.

    Real storage layout compatible (try in order, use first hit with weight files):
      <models_root>/<name_or_path>            e.g. models/Qwen/Qwen2.5-0.5B-Instruct
      <models_root>/<basename>                e.g. models/Qwen2.5-0.5B-Instruct
      <models_root>/<basename>.lower()        case-insensitive
      <models_root>/ case-insensitive scan matching basename
    No hit -> return as-is (let transformers load online as HF repo id).
    """
    if not name_or_path:
        return name_or_path
    p = Path(name_or_path)
    if p.is_absolute() and _has_model_files(p):
        return str(p)

    root = models_root()
    base = Path(name_or_path).name
    candidates = [root / name_or_path, root / base, root / base.lower()]
    for c in candidates:
        if _has_model_files(c):
            return str(c)
    if root.is_dir():
        for child in root.iterdir():
            if child.name.lower() == base.lower() and _has_model_files(child):
                return str(child)
    return name_or_path


def list_local_models() -> dict:
    """List model directories already in place under storage server models_root (for storage-check)."""
    root = models_root()
    found = []
    if root.is_dir():
        for child in root.iterdir():
            if _has_model_files(child):
                found.append(child.name)
            elif child.is_dir():
                for gc in child.iterdir():
                    if _has_model_files(gc):
                        found.append(f"{child.name}/{gc.name}")
    return {"models_root": str(root), "found": sorted(found), "count": len(found)}


def storage_report() -> dict:
    """RunPod storage mount and available space report."""
    import shutil
    out = {"workspace_mounted": WORKSPACE.is_dir(),
           "data_root": str(data_root()), "models_root": str(models_root()),
           "hf_cache": str(hf_cache())}
    for label, path in (("data_root", data_root()), ("models_root", models_root())):
        try:
            if path.exists():
                du = shutil.disk_usage(path)
                out[f"{label}_free_gb"] = round(du.free / 1e9, 1)
                out[f"{label}_exists"] = True
            else:
                out[f"{label}_exists"] = False
        except Exception:
            out[f"{label}_exists"] = False
    return out


def apply_hf_env():
    """Point HF cache to persistent volume, avoiding lost downloads on container restart.

    Call before real model loading. Sets HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE.
    """
    cache = hf_cache()
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))
    return str(cache)
