"""gguf_backend.py — Score a GGUF-quantised model on the SAME data via llama.cpp.

The edge student in the paper is a 0.5B Qwen2.5 quantised to Q4_K_M (GGUF, ~240 MB). GGUF cannot be
loaded by transformers, so head-to-head comparison against the BF16 teacher requires a separate
llama.cpp inference path. This module wraps llama-cpp-python and degrades gracefully when the runtime
or the model file is unavailable (e.g. in the sandbox), so the rest of the pipeline never breaks.

Honesty: results carry runtime="llama_cpp" so downstream reports can label the GGUF scores as produced
by a different runtime than the transformers (safetensors) baseline.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger("gguf_backend")


class GGUFUnavailable(RuntimeError):
    """Raised when llama-cpp-python or the GGUF model file is not available."""


def _find_gguf(model_ref: str) -> Path:
    """Resolve a GGUF file from a model reference (local dir, file, or staged models root)."""
    from realeval.paths import resolve_model
    p = Path(resolve_model(model_ref))
    if p.is_file() and p.suffix == ".gguf":
        return p
    if p.is_dir():
        ggufs = sorted(p.glob("*.gguf"))
        # Prefer a merged single-file GGUF over shards.
        merged = [g for g in ggufs if "-of-" not in g.name]
        if merged:
            return merged[0]
        if ggufs:
            return ggufs[0]
    raise GGUFUnavailable(f"No .gguf file found for {model_ref!r} (looked at {p})")


def gguf_classify(model_ref: str, texts, labels, *, n_ctx: int = 2048,
                  n_gpu_layers: int = -1, max_tokens: int = 4) -> dict:
    """Classify fraud/normal texts with a GGUF model via llama.cpp; return real metrics.

    Uses a minimal instruction prompt and reads the model's first-token decision. Returns a dict with
    f1/accuracy, the per-sample latency, and runtime="llama_cpp". Raises GGUFUnavailable if the
    llama_cpp package or the model file is missing (caller decides how to handle).
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise GGUFUnavailable("llama-cpp-python is not installed (pip install llama-cpp-python)") from e

    gguf_path = _find_gguf(model_ref)
    logger.info("Loading GGUF %s via llama.cpp", gguf_path)
    llm = Llama(model_path=str(gguf_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)

    from realeval.metrics import classification_metrics
    preds, lat = [], []
    for txt in texts:
        prompt = ("You are a fraud detector. Answer with a single digit: 1 if the message is fraud, "
                  "0 if normal.\nMessage: " + str(txt)[:1000] + "\nAnswer:")
        t0 = time.perf_counter()
        out = llm(prompt, max_tokens=max_tokens, temperature=0.0)
        lat.append((time.perf_counter() - t0) * 1000)
        ans = out["choices"][0]["text"].strip()
        # Robust regex-based extraction: match digit or fraud/normal keywords.
        # Handles "1", "0", "1.", "fraud", "Fraud", "normal", "answer: 1", etc.
        import re
        m = re.search(r"\b(1|0)\b", ans) or re.search(r"\b(fraud|normal)\b", ans, re.IGNORECASE)
        if m:
            token = m.group(1).lower()
            preds.append(1 if token in ("1", "fraud") else 0)
        else:
            preds.append(0)  # fallback: default to non-fraud

    import numpy as np
    m = classification_metrics(np.asarray(labels), np.asarray(preds))
    return {
        "f1": m["f1"],
        "accuracy": m["accuracy"],
        "latency_ms_p50": round(float(np.median(lat)), 2),
        "runtime": "llama_cpp",
        "gguf_file": gguf_path.name,
        "n": len(texts),
    }
