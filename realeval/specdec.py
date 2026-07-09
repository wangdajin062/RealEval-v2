"""realeval/specdec.py — Speculative Decoding Diagnostics

Diagnostic tools for speculative decoding acceptance rate analysis.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("specdec")


def diagnostic_B(config: dict, texts: list[str], *, gamma=5, n_samples=20) -> dict:
    """Diagnostic B: compute alpha from token counts (paper Table 8 consistency check).

    Returns dict with alpha_from_tokens, paper_tokens, table8_alpha, table8_tokens_would_be, verdict.
    """
    from realeval import models, hwenv
    import torch
    import torch.nn.functional as F

    from realeval.real_backend import require_assets, AssetsUnavailable

    require_assets(models.models_available(config), "Real Qwen weights unavailable")
    target, tok = models.load_causal_lm(config["models"]["teacher"], bf16=True)
    draft_path = config["models"].get("draft_model")
    draft, _ = models.load_causal_lm(draft_path, bf16=True)
    require_assets(target is not None and draft is not None, "draft/target loading failed")
    dev = next(target.parameters()).device

    # Compute alpha from token counts (paper method)
    accepted, proposed = 0, 0
    for text in texts[:n_samples]:
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        seq = ids
        for _ in range(40 // gamma):
            dprobs, dtoks = [], []
            cur = seq
            with torch.no_grad(), hwenv.autocast_context():
                for _g in range(gamma):
                    p = F.softmax(draft(cur).logits[0, -1], -1)
                    tk = int(torch.argmax(p))
                    dtoks.append(tk); dprobs.append(float(p[tk]))
                    cur = torch.cat([cur, torch.tensor([[tk]], device=dev)], 1)
                proposed += gamma
                ext = torch.cat([seq, torch.tensor([dtoks], device=dev)], 1)
                tlog = target(ext).logits
            base = seq.shape[1]
            ok = 0
            for i, tk in enumerate(dtoks):
                pt = float(F.softmax(tlog[0, base + i - 1], -1)[tk])
                if pt / (dprobs[i] + 1e-9) >= 0.5:
                    ok += 1
                else:
                    break
            accepted += ok
            if ok == 0:
                break
            seq = torch.cat([seq, torch.tensor([dtoks[:ok]], device=dev)], 1)

    gen_alpha = round(accepted / max(1, proposed), 4)

    def _tokens(a):
        return round((1 - a ** (gamma + 1)) / (1 - a), 2) if a < 1 else gamma + 1

    # The generic draft's alpha is MEASURED from real tokens above.
    # The domain-tuned draft requires a separately fine-tuned model; when not loaded,
    # it is reported as not_measured (no hardcoded paper fallback).
    result = {
        "alpha_generic_measured": gen_alpha,
        "alpha_domain": None,  # must be measured with actual domain-tuned draft model
        "paper_tokens_generic": _tokens(gen_alpha),
        "gamma": gamma,
        "n_samples": n_samples,
        "accepted": accepted,
        "proposed": proposed,
    }
    return result
