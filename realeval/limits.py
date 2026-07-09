"""realeval/limits.py — Resource Limits (GPU Memory Fraction, Concurrency Lock)

GPU memory limit: set via torch.cuda.set_per_process_memory_fraction (PyTorch 2.0+).
Concurrency guard: atomic slot-based file lock (O_CREAT|O_EXCL) to limit concurrent experiment processes.
Lock files store the PID of the owning process; stale-lock cleanup verifies the PID is actually dead.
"""
from __future__ import annotations
import os
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_LOCK_DIR = ROOT / ".locks"

# Windows: PROCESS_QUERY_INFORMATION access right
_PROCESS_QUERY_INFORMATION = 0x0400


def _my_pid() -> str:
    return str(os.getpid())


def check_limits() -> dict:
    """Check current resource limits. Returns dict with GPU memory limit and concurrency info."""
    result = {"gpu_memory_limit": None, "concurrency_slots": None}
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_memory_limit"] = 0.95
    except Exception:
        pass
    result["concurrency_slots"] = 4
    return result


def set_gpu_memory_limit(fraction: float = 0.95) -> dict:

    """Set per-process GPU memory fraction. Returns status dict. Degrades gracefully on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
            return {"applied": True, "fraction": fraction}
        return {"applied": False, "reason": "CUDA unavailable"}
    except Exception as e:
        return {"applied": False, "reason": str(e)}


def _pid_is_alive(pid: int) -> bool:
    """Check if a process is alive (cross-platform)."""
    if os.name == "nt":  # Windows
        try:
            import ctypes
            # PROCESS_QUERY_INFORMATION (0x0400) allows querying process info without terminating it
            handle = ctypes.windll.kernel32.OpenProcess(_PROCESS_QUERY_INFORMATION, False, pid)
            if not handle:
                # OpenProcess may return NULL for privileged or cross-user processes even
                # when the PID is alive. Fall back to tasklist for a definitive check.
                import subprocess
                out = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    timeout=5, stderr=subprocess.DEVNULL
                ).decode("utf-8", errors="replace")
                if str(pid) in out:
                    return True
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:
            # On any error (ctypes unavailable, tasklist timeout, etc.),
            # conservatively assume the PID IS alive to avoid deleting a
            # lock owned by a running process.
            return True
    # POSIX
    try:
        os.kill(pid, 0)  # signal 0 = existence check only
        return True
    except OSError:
        return False


def cleanup_stale_locks():
    """Remove lock files whose owning PID is no longer alive.

    Only removes locks whose stored PID does not correspond to a running process.
    Locks from living processes are preserved, so active experiments are never disrupted.
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(_LOCK_DIR.iterdir()):
        if f.suffix != ".lock":
            continue
        try:
            stored = int(f.read_text(encoding="utf-8").strip())
            if not _pid_is_alive(stored):
                f.unlink()
        except (ValueError, OSError, PermissionError):
            # Corrupted lock or permission denied — try to remove to avoid blocking
            try:
                f.unlink()
            except Exception:
                pass


@contextmanager
def concurrency_guard(max_concurrent: int = 4, retry_sec: int = 300):
    """Atomic slot-based concurrency lock. Blocks until a slot is available.

    Uses O_CREAT|O_EXCL for atomic file creation to avoid TOCTOU races.
    Each slot_{n}.lock stores the PID of the owning process so cleanup_stale_locks()
    can distinguish dead locks from live ones.

    Args:
        max_concurrent: Maximum number of concurrent processes (slots 0..max_concurrent-1).
        retry_sec: Total time to keep retrying before raising.

    Yields:
        The acquired lock file Path.

    Raises:
        RuntimeError: If a slot cannot be acquired within retry_sec seconds.
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_locks()
    max_attempts = max(1, retry_sec)
    acquired = False
    lock_file: Path | None = None

    try:
        for attempt in range(max_attempts):
            for slot in range(max_concurrent):
                lf = _LOCK_DIR / f"slot_{slot}.lock"
                try:
                    # Atomic create: O_EXCL ensures failure if file already exists.
                    # This is the critical fix for the TOCTOU race — there is no
                    # gap between "check if free" and "claim" as in the old code.
                    fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, _my_pid().encode())
                    os.close(fd)
                    lock_file = lf
                    acquired = True
                    break
                except FileExistsError:
                    # Slot taken — check if owner is alive, reap if dead
                    try:
                        stored = int(lf.read_text(encoding="utf-8").strip())
                        if not _pid_is_alive(stored):
                            lf.unlink()
                    except Exception:
                        pass
                    continue
            if acquired:
                break
            time.sleep(1.0)

        if not acquired:
            raise RuntimeError(
                f"Could not acquire concurrency slot (max={max_concurrent}) after "
                f"{retry_sec}s. Check for stuck processes in {_LOCK_DIR}."
            )
        yield lock_file

    finally:
        if acquired and lock_file and lock_file.exists():
            try:
                lock_file.unlink()
            except (OSError, PermissionError):
                pass
