"""exp9: CoT Ablation — Compare chain-of-thought vs direct classification."""
from __future__ import annotations
import logging
logger = logging.getLogger("exp9")


def run(config: dict) -> dict:
    smoke = config.get("_smoke", False)
    from realeval import data
    ds = data.load_taf28k(max_samples=config.get("data", {}).get("max_samples", 1000))
    texts, labels = ds["texts"], ds["labels"]
    if not texts:
        ds = data.load_synthetic(n=100)
        texts, labels = ds["texts"], ds["labels"]
    split = int(len(texts) * 0.8)
    train_texts, test_texts = texts[:split], texts[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    from realeval.real_backend import run_paper_safe

    def run_paper(config):
        from realeval import real_backend
        direct = real_backend.real_llm_classify(config, test_texts, test_labels, quantize="int4", use_cot=False)
        cot = real_backend.real_llm_classify(config, test_texts, test_labels, quantize="int4", use_cot=True)
        return {"experiment": "exp9", "computation": "h100_real_qwen",
                "with_cot": {"f1": cot["f1"], "fpr": cot.get("fpr")},
                "without_cot": {"f1": direct["f1"], "fpr": direct.get("fpr")}}

    paper_result = run_paper_safe(smoke, config, run_paper)
    if paper_result is not None:
        return paper_result

    logger.info("SMOKE: running small-model verification for exp9")
    from sklearn.ensemble import GradientBoostingClassifier
    from realeval.metrics import classification_metrics
    from realeval.data import verification_features
    import numpy as np
    # Real CoT-vs-direct proxy: "direct" classifies on the raw features; "CoT" augments them with
    # pairwise interaction features (emulating an extra reasoning step), so both F1 AND FPR are measured.
    X, y = verification_features(labels)
    ntr = split
    # direct
    clf_d = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X[:ntr], y[:ntr])
    m_direct = classification_metrics(y[ntr:], clf_d.predict(X[ntr:]))
    # CoT: add a few interaction features (reasoning augmentation)
    inter = np.hstack([X, (X[:, :8] * X[:, 8:16])])
    clf_c = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(inter[:ntr], y[:ntr])
    m_cot = classification_metrics(y[ntr:], clf_c.predict(inter[ntr:]))
    return {"experiment": "exp9", "computation": "smoke_sklearn",
            "with_cot": {"f1": m_cot["f1"], "fpr": m_cot["fpr"]},
            "without_cot": {"f1": m_direct["f1"], "fpr": m_direct["fpr"]}}
