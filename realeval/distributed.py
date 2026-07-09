"""realeval/distributed.py — Multi-GPU (DDP) Support

When launched via torchrun, reads RANK/LOCAL_RANK/WORLD_SIZE, initializes NCCL process group, wraps model as DDP.
Single process (no torchrun): all operations degrade to no-ops, code runs normally on single GPU/CPU.

Usage (in training experiments):
    from realeval import distributed as dist
    dist.init()
    model = dist.wrap(model)          # DDP wrap (returns as-is in single process)
    ...
    if dist.is_main(): save(...)      # Only main process writes results
    dist.cleanup()
"""
from __future__ import annotations
import logging
import os

logger = logging.getLogger("distributed")


def is_distributed() -> bool:
    """Check if running under torchrun (WORLD_SIZE > 1)."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank() -> int:
    """Return global rank (0 in single process)."""
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    """Return local rank (0 in single process)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    """Return world size (1 in single process)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    """True on rank 0 (the only rank in single process)."""
    return rank() == 0


_INITED = False


def init(backend: str = "nccl", timeout_s: int = 600) -> dict:
    """Initialize distributed (if under torchrun). Returns status dict. No-op in single process.

    Args:
        backend: NCCL backend for GPU communication.
        timeout_s: Timeout for process group initialization.

    Returns:
        Dict with keys: distributed, rank, world_size, initialized, error (if any).
    """
    global _INITED
    info = {"distributed": is_distributed(), "rank": rank(), "world_size": world_size(),
            "local_rank": local_rank(), "backend": backend}
    if not is_distributed() or _INITED:
        return info
    try:
        import torch.distributed as td
        if not td.is_initialized():
            td.init_process_group(backend=backend,
                                  timeout=__import__("datetime").timedelta(seconds=timeout_s))
        _INITED = True
        info["initialized"] = True
        logger.info("DDP init: rank=%d world=%d backend=%s", rank(), world_size(), backend)
    except Exception as e:
        logger.warning("DDP init failed (single-GPU fallback): %s", e)
        info["error"] = str(e)
    return info


def wrap(model, sync_bn: bool = False):
    """Wrap model in DDP if distributed. No-op in single process.

    Args:
        model: PyTorch model to wrap.
        sync_bn: If True, convert batch norm to SyncBatchNorm before wrapping.

    Returns:
        Wrapped model (or original in single process).
    """
    if not is_distributed():
        return model
    import torch.nn as nn
    if sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    from torch.nn.parallel import DistributedDataParallel
    return DistributedDataParallel(model, device_ids=[local_rank()])


def cleanup():
    """Destroy distributed process group."""
    global _INITED
    if _INITED:
        try:
            import torch.distributed as td
            td.destroy_process_group()
        except Exception:
            logger.debug("DDP process group destroy failed (may already be destroyed)")
        _INITED = False


def all_reduce_mean(value: float) -> float:
    """All-reduce mean across all processes. Single process returns value unchanged."""
    if not is_distributed():
        return value
    import torch
    import torch.distributed as td
    t = torch.tensor([value], device="cuda" if torch.cuda.is_available() else "cpu")
    td.all_reduce(t, op=td.ReduceOp.SUM)
    return float(t.item() / world_size())
