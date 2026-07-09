"""realeval/specdec.py — Speculative Decoding Diagnostics

Diagnostic tools for speculative decoding acceptance rate analysis.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("specdec")


def diagnostic_B(config: dict, texts: list[str], *, gamma=5, n_samples=20) -> dict:
    """Diagnostic B: compute alpha from token counts (paper Table 8 consistency check).

    Returns dict with h100_measured, h100_tokens, v25_table8_alpha, v25_table8_tokens, verdict.
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

    # ── H100 measured values (authoritative) ──
    # The generic draft's alpha is measured from real H100 tokens above.
    # The domain-tuned draft would require separate fine-tuned model loading.
    # domain-tuned draft alpha NOT hardcoded
    h100_measured = {"generic": gen_alpha}
    h100_tokens = {"generic": _tokens(gen_alpha)}

    # ── v25 paper Table 8 values (historical reference only) ──
    # These values (0.85/0.91) are from the v25 manuscript and may differ from H100 measurements.
    v25_table8_alpha = {"generic": 0.85}
    v25_table8_tokens = {"generic": _tokens(0.85)}

    # Verdict: H100 measured alpha is ground truth.
    verdict_data = {}
    if gen_alpha is not None:
        v25_ref = v25_table8_alpha.get("generic", 0.85)
        diff = abs(gen_alpha - v25_ref)
        if diff >= 0.05:
            verdict_data["generic"] = (
                f"H100 measured generic alpha={gen_alpha} differs from "
                f"v25 Table 8's {v25_ref} (diff={diff:.3f})")
        else:
            verdict_data["generic"] = (
                f"H100 measured generic alpha={gen_alpha} is consistent "
                f"with v25 Table 8's {v25_ref}")
    verdict_data["domain"] = "NOT MEASURED"
    verdict = "; ".join(verdict_data.values())

    return {
        "h100_measured": h100_measured,
        "h100_tokens": h100_tokens,
        "v25_table8_alpha": v25_table8_alpha,
        "v25_table8_tokens": v25_table8_tokens,
        "verdict": verdict,
    }
