"""realeval/validation.py — Input Validation (Whitelist)

Whitelist-based input validation: rejects illegal source names, model IDs, experiment selections, path traversals.
"""
from __future__ import annotations
import string
from pathlib import Path

# Whitelist: allowed data sources
ALLOWED_SOURCES = {"auto", "taf28k", "synthetic", "chifraud"}
# Whitelist: allowed experiment name prefix
ALLOWED_EXP_PREFIX = "exp"
# Whitelist: allowed model ID characters (alphanumeric, slash, hyphen, underscore, dot)
ALLOWED_MODEL_CHARS = set(string.ascii_letters + string.digits + "/-_.")


class ValidationError(ValueError):
    """Input validation error."""


def validate_config(cfg: dict):
    """Validate config dict. Raises ValidationError on illegal values."""
    data = cfg.get("data", {})
    source = data.get("source", "auto")
    if source not in ALLOWED_SOURCES:
        raise ValidationError(f"Illegal data source: {source!r}. Allowed: {ALLOWED_SOURCES}")
    max_samples = data.get("max_samples", 4000)
    if not isinstance(max_samples, int) or max_samples <= 0 or max_samples > 10_000_000:
        raise ValidationError(f"Illegal max_samples: {max_samples}")
    models = cfg.get("models", {})
    for key, val in models.items():
        if not isinstance(val, str):
            continue
        if not set(val).issubset(ALLOWED_MODEL_CHARS):
            raise ValidationError(f"Illegal model ID: {val!r} (contains disallowed characters)")


def validate_experiment_selection(selection: str, available: list[tuple[str, str]]) -> list[str]:
    """Validate experiment selection string (e.g. '1,3,6'). Returns list of experiment names."""
    if not selection.replace(",", "").isdigit():
        raise ValidationError(f"Illegal experiment selection: {selection!r} (must be comma-separated numbers)")
    names = []
    for s in selection.split(","):
        name = f"exp{s.strip()}"
        if not any(name == a[0] for a in available):
            raise ValidationError(f"Unknown experiment: {name}")
        names.append(name)
    return names


def validate_experiment_name(name: str) -> str:
    """Validate experiment name (e.g. 'exp7'). Returns name if valid."""
    if not name.startswith(ALLOWED_EXP_PREFIX) or not name[len(ALLOWED_EXP_PREFIX):].isdigit():
        raise ValidationError(f"Illegal experiment name: {name!r}")
    return name


def safe_path(path: Path, allowed_roots: list[Path], must_exist: bool = True) -> Path:
    """Resolve path and verify it falls within one of the allowed root directories.

    Raises ValidationError on path traversal escape.
    """
    p = path.resolve()
    if must_exist and not p.exists():
        raise ValidationError(f"Path does not exist: {p}")
    for root in allowed_roots:
        try:
            root_resolved = root.resolve()
            if root_resolved in p.parents or p == root_resolved:
                return p
        except Exception:
            continue
    raise ValidationError(f"Path traversal rejected: {p} not in allowed roots {allowed_roots}")
