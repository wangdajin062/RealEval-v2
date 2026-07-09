"""realeval/hwenv.py — Hardware Environment Detection (H100/CUDA/BF16/FlashAttention/torch.compile)

8 independent detection functions (①) each return dict, aggregated by detect().
check() returns (ok, issues) for paper-grade requirement verification.
"""
from __future__ import annotations
import logging
from contextlib import contextmanager

logger = logging.getLogger("hwenv")


def detect_gpu() -> dict:
    """Detect GPU availability, count, and name."""
    try:
        import torch
        return {"cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception:
        return {"cuda_available": False, "gpu_count": 0, "gpu_name": None}


def detect_cuda() -> dict:
    """Detect CUDA version from torch."""
    try:
        import torch
        return {"cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None}
    except Exception:
        return {"cuda_version": None}


def detect_nccl() -> dict:
    """Detect NCCL availability via torch.distributed."""
    try:
        import torch.distributed as td
        return {"nccl_available": td.is_nccl_available()}
    except Exception:
        return {"nccl_available": False}


def detect_flashattention() -> dict:
    """FlashAttention detection: try import flash_attn, fall back to sdpa."""
    fa = False
    try:
        import flash_attn  # noqa
        fa = True
    except Exception:
        pass
    return {"flash_attn_available": fa, "flash_attn_version": "2" if fa else None}


def detect_compile() -> dict:
    """Detect torch.compile availability."""
    try:
        import torch
        return {"torch_compile_available": hasattr(torch, "compile")}
    except Exception:
        return {"torch_compile_available": False}


def detect_bf16() -> dict:
    """Detect BF16 support (requires CUDA + compatible GPU)."""
    try:
        import torch
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return {"bf16_supported": bf16}
    except Exception:
        return {"bf16_supported": False}


def detect_tf32() -> dict:
    """Detect TF32 support (Ampere+ architecture)."""
    try:
        import torch
        return {"tf32_available": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8}
    except Exception:
        return {"tf32_available": False}


def detect_distributed() -> dict:
    """Detect distributed environment variables (set by torchrun)."""
    import os
    return {"world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0"))}


def detect(verbose: bool = True) -> dict:
    """Aggregate all 8 detectors + is_h100 and torch_version."""
    env = {}
    for fn in (detect_gpu, detect_cuda, detect_nccl, detect_flashattention,
               detect_compile, detect_bf16, detect_tf32, detect_distributed):
        env.update(fn())
    env["is_h100"] = env.get("gpu_name") is not None and "H100" in env["gpu_name"]
    try:
        import torch
        env["torch_version"] = torch.__version__
    except Exception:
        pass
    # Build gpu_names list for compatibility with B's envreport
    if env.get("cuda_available"):
        try:
            import torch
            env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(env.get("gpu_count", 0))]
            env["compute_capability"] = [list(torch.cuda.get_device_capability(i)) for i in range(env.get("gpu_count", 0))]
        except Exception:
            env["gpu_names"] = [env.get("gpu_name")] if env.get("gpu_name") else []
            env["compute_capability"] = []
    else:
        env["gpu_names"] = []
        env["compute_capability"] = []
    if verbose:
        for k, v in sorted(env.items()):
            logger.info("  %s: %s", k, v)
    return env


def check(strict: bool = False) -> tuple[bool, list[str]]:
    """Check paper-grade requirements. strict=True requires H100."""
    env = detect(verbose=False)
    issues = []
    if not env.get("cuda_available"):
        issues.append("CUDA unavailable (paper-grade requires H100)")
    if strict and not env.get("is_h100"):
        issues.append("Not H100 (paper-grade requires H100 GPU)")
    if not env.get("bf16_supported"):
        issues.append("BF16 unsupported (paper-grade requires BF16)")
    if not env.get("flash_attn_available"):
        issues.append("FlashAttention unavailable (paper-grade recommends FlashAttention)")
    if not env.get("torch_compile_available"):
        issues.append("torch.compile unavailable (paper-grade recommends torch.compile)")
    ok = len(issues) == 0
    return ok, issues


def autocast_context():
    """Return an autocast context manager (BF16 if CUDA available, else a no-op nullcontext).

    NOTE: this is a plain factory that RETURNS a context-manager object; it is NOT decorated with
    @contextmanager (that decorator is for generator functions that yield). Use as:
        with hwenv.autocast_context():
            ...
    Uses the modern torch.autocast API (torch.cuda.amp.autocast is deprecated since PyTorch 2.4).
    The CPU fallback is a genuine no-op (nullcontext), not no_grad(), so it never disables gradients.
    """
    import torch
    from contextlib import nullcontext
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def apply_optimizations(env: dict = None) -> dict:
    """Apply global optimizations: TF32, matmul precision. Returns applied dict."""
    if env is None:
        env = detect(verbose=False)
    applied = {"tf32": False, "matmul_precision": "highest"}
    try:
        import torch
        if env.get("cuda_available"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            applied["tf32"] = True
        torch.set_float32_matmul_precision("high")
        applied["matmul_precision"] = "high"
    except Exception:
        pass
    return applied
