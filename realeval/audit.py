"""realeval/audit.py — Audit Log for Critical Operations and Exceptions

Centralized recording: config validation, data source resolution, model resolution, experiment start/end,
resource limits, errors/exceptions.
Writes to outputs/audit.log (append), facilitating post-hoc investigation and reproduction audit.

Uses a module-level logger dict to support test isolation (each test can call reset_audit_logger()).
"""
from __future__ import annotations
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG = ROOT / "outputs" / "audit.log"

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


def log_event(event: str, **fields):
    """Record a structured audit event.

    Args:
        event: Event name (e.g. "config_validated", "experiment_start").
        **fields: Key-value pairs describing the event context.
    """
    lg = get_audit_logger()
    detail = " ".join(f"{k}={v}" for k, v in fields.items())
    lg.info("%s %s", event, detail)


def log_error(where: str, exc: Exception):
    """Record an error event.

    Args:
        where: Context string describing where the error occurred.
        exc: The exception instance to record.
    """
    lg = get_audit_logger()
    lg.error("ERROR in %s: %s: %s", where, type(exc).__name__, exc)
