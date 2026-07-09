"""realeval/models.py — Real Model Loading (H100 Path)

Loads Qwen2.5 teacher/student/draft, Whisper on H100 with real weights, automatically applies BF16 /
torch.compile / TF32 / (optional) quantization. Gracefully returns None in sandbox without transformers/weights,
experiments fall back to small model path.

Model paths from config.experiments.yaml models: section. Quantization via bitsandbytes (4-bit) or native dtype.
"""
from __future__ import annotations
import logging
from pathlib import Path

from realeval import hwenv

logger = logging.getLogger("models")


def _have_transformers() -> bool:
    try:
        import transformers  # noqa
        return True
    except Exception:
        return False


def _resolve(path: str) -> str | None:
    """Local directory (including /workspace/models) or HF repo id. Resolved via paths.resolve_model."""
    if path is None:
        return None
    from realeval import paths
    paths.apply_hf_env()                        # HF cache points to persistent volume /workspace/hf_cache
    resolved = paths.resolve_model(path)
    p = Path(resolved)
    if p.exists():
        return str(p)
    # Not found on local storage — pass through as HF repo id.
    # path.resolve_model returns name_or_path unchanged when not found locally,
    # so resolved is the original string.  Slash is NOT required — repo ids like
    # "gpt2" (no org prefix) are valid HF identifiers.
    return resolved


def load_causal_lm(path: str, *, quantize: str = None, bf16: bool = True,
                   compile_model: bool = True, device: str = None):
    """Load causal LM. Returns (model, tokenizer) or (None, None) (when unavailable).

    quantize: None | 'int4' (bitsandbytes 4-bit) | 'int8'
    """
    if not _have_transformers():
        logger.warning("transformers unavailable, cannot load real model %s", path)
        return None, None
    resolved = _resolve(path)
    if resolved is None:
        logger.warning("Model path does not exist and is not a repo id: %s", path)
        return None, None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    env = hwenv.detect(verbose=False)
    dev = device or ("cuda" if env["cuda_available"] else "cpu")
    dtype = torch.bfloat16 if (bf16 and env["bf16_supported"] and getattr(dev, "type", str(dev)) == "cuda") else torch.float32

    kwargs = {"torch_dtype": dtype}
    # FlashAttention: transformers supports attn_implementation
    if env["flash_attn_available"]:
        kwargs["attn_implementation"] = "flash_attention_2" if env.get("flash_attn_version") else "sdpa"
    else:
        kwargs["attn_implementation"] = "sdpa"

    # fp16 loads in explicit half precision (no bitsandbytes); nf4/int4/int8 use bitsandbytes.
    if quantize == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif quantize in ("int4", "int8", "nf4"):
        try:
            from transformers import BitsAndBytesConfig
            # nf4 and int4 are both 4-bit; nf4 uses the NF4 quant type, int4 uses FP4.
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quantize in ("int4", "nf4")), load_in_8bit=(quantize == "int8"),
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=("nf4" if quantize == "nf4" else "fp4"))
            kwargs["device_map"] = "auto"
        except Exception as e:
            logger.warning("bitsandbytes quantization unavailable (%s), falling back to full precision", e)
    elif quantize not in (None, "bf16", "fp32"):
        logger.warning("Unknown quantize scheme %r; loading at %s (no quantization applied)", quantize, dtype)

    try:
        tok = AutoTokenizer.from_pretrained(resolved)
        model = AutoModelForCausalLM.from_pretrained(resolved, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(dev)
        if compile_model and env["torch_compile_available"]:
            try:
                model = torch.compile(model)
            except Exception as e:
                logger.warning("torch.compile skipped: %s", e)
        model.eval()
        logger.info("Loaded %s: dtype=%s attn=%s quant=%s dev=%s",
                    resolved, dtype, kwargs.get("attn_implementation"), quantize, dev)
        return model, tok
    except Exception as e:
        logger.warning("Failed to load %s: %s", resolved, e)
        return None, None


def models_available(config: dict) -> bool:
    """Check if real Qwen paths are available (transformers + teacher weights resolvable)."""
    if not _have_transformers():
        return False
    teacher = config.get("models", {}).get("teacher")
    resolved = _resolve(teacher)
    if resolved is None:
        return False
    # Verify that the resolved path has actual model weights on disk,
    # not just a string that fell through to be treated as an HF repo id.
    p = Path(resolved)
    if p.exists() and p.is_dir():
        # Look for typical model weight file markers (safetensors, bin, or HF hub marker)
        has_weights = any(p.glob("model-*.safetensors")) or any(p.glob("pytorch_model*.bin")) or any(p.glob("model.safetensors"))
        if has_weights:
            return True
    # No local weights found — the resolved value is being treated as an HF repo id.
    # Return False so callers can distinguish "available locally" from "might work if HF accessible".
    return False
