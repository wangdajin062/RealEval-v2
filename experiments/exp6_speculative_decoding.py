"""exp6: Speculative Decoding — Acceptance rate diagnostics (Table 8)."""
from __future__ import annotations
import logging
logger = logging.getLogger("exp6")


def run(config: dict) -> dict:
    smoke = config.get("_smoke", False)
    from realeval import data
    ds = data.load_taf28k(max_samples=config.get("data", {}).get("max_samples", 100))
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
    return {
        "experiment": "exp6",
        "computation": "smoke_synthetic",
        "diagnostic_B": {
            "alpha_generic_measured": None,
            "alpha_domain": None,
            "gamma": 5,
            "n_samples": 0,
            "accepted": 0,
            "proposed": 0,
            "note": "sandbox: no real draft/target model available; all values set to None (not measured)",
        },
    }
