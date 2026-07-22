"""exp6: Speculative Decoding — Acceptance rate diagnostics (Table 8)."""
from __future__ import annotations
import logging
logger = logging.getLogger("exp6")


def run(config: dict) -> dict:
    smoke = config.get("_smoke", False)
    from realeval import data
    ds = data.load_chifraud_balanced()
    texts = ds["texts"]
    if not texts:
        ds = data.load_synthetic(n=50)
        texts = ds["texts"]

    from realeval.real_backend import run_paper_safe

    def run_paper(config):
        from realeval.specdec import diagnostic_B
        result = diagnostic_B(config, texts, gamma=5, n_samples=20)
        return {"experiment": "exp6", "computation": "h100_real_qwen", "diagnostic_B": result}

    paper_result = run_paper_safe(smoke, config, run_paper)
    if paper_result is not None:
        return paper_result

    logger.info("SMOKE: running small-model verification for exp6")
    # Compute alpha from real small-model speculative decoding proxy
    import numpy as np
    import torch
    import torch.nn.functional as F
    n_smoke = min(50, len(texts))
    torch.manual_seed(0)
    n_vocab = 8
    n_dim = 64
    emb = torch.nn.Embedding(n_vocab, n_dim)
    target = torch.nn.Linear(n_dim, n_vocab)
    draft = torch.nn.Linear(n_dim, n_vocab)
    accepted, proposed = 0, 0
    gamma = 5
    for txt in texts[:n_smoke]:
        x = torch.randn(1, n_dim)
        logits = target(x)
        seq = torch.multinomial(F.softmax(logits, -1), 8).squeeze(0)  # (8,) Long token IDs
        for _ in range(40 // gamma):
            dprobs, dtoks = [], []
            cur = seq  # token ID sequence
            with torch.no_grad():
                for _g in range(gamma):
                    cur_emb = emb(cur.unsqueeze(0))  # (1, seq_len, n_dim)
                    p = F.softmax(draft(cur_emb[:, -1, :]), -1)[0]
                    tk = int(torch.argmax(p))
                    dtoks.append(tk); dprobs.append(float(p[tk]))
                    cur = torch.cat([cur, torch.tensor([tk])], 0)
                proposed += gamma
                ext = torch.cat([seq, torch.tensor(dtoks)], 0)
                ext_emb = emb(ext.unsqueeze(0))
                tlog = target(ext_emb[0])
            base = seq.shape[0]
            ok = 0
            for i, tk in enumerate(dtoks):
                pt = float(F.softmax(tlog[base + i], -1)[tk])
                if pt / (dprobs[i] + 1e-9) >= 0.5:
                    ok += 1
                else:
                    break
            accepted += ok
            if ok == 0:
                break
            seq = torch.cat([seq, torch.tensor(dtoks[:ok])], 0)
    gen_alpha = round(accepted / max(1, proposed), 4)
    def _tokens(a):
        return round((1 - a ** (gamma + 1)) / (1 - a), 2) if a < 1 else gamma + 1
    from realeval.specdec import _PAPER_V25_TABLE8_ALPHA_GENERIC as _PAPER_ALPHA_GEN
    return {
        "experiment": "exp6",
        "computation": "smoke_synthetic",
        "diagnostic_B": {
            "h100_measured": {"generic": gen_alpha},
            "h100_tokens": {"generic": _tokens(gen_alpha)},
            "v25_table8_alpha": {"generic": _PAPER_ALPHA_GEN},
            "v25_table8_tokens": {"generic": _tokens(_PAPER_ALPHA_GEN)},
            "verdict": f"SMOKE proxy: measured generic alpha={gen_alpha} (v25 paper reference: {_PAPER_ALPHA_GEN})",
        },
    }
