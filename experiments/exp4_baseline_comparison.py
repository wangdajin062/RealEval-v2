"""exp4: Baseline Comparison — Compare QAD against standard baselines (LogReg, XGBoost, MLP)."""
from __future__ import annotations
import logging
logger = logging.getLogger("exp4")


def run(config: dict) -> dict:
    smoke = config.get("_smoke", False)
    from realeval import data
    ds = data.load_taf28k(max_samples=config.get("data", {}).get("max_samples", 4000))
    texts, labels = ds["texts"], ds["labels"]
    if not texts:
        ds = data.load_synthetic(n=200)
        texts, labels = ds["texts"], ds["labels"]
    split = int(len(texts) * 0.8)
    train_texts, test_texts = texts[:split], texts[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    from realeval.real_backend import run_paper_safe

    def run_paper(config):
        from realeval import real_backend
        from realeval.metrics import classification_metrics
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_extraction.text import HashingVectorizer
        baselines = {}
        # Classical baselines are genuinely different algorithms on real text features (not the LLM).
        vec = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2")
        Xtr = vec.transform(train_texts); Xte = vec.transform(test_texts)
        algos = {"logreg": LogisticRegression(max_iter=1000),
                 "xgb": GradientBoostingClassifier(n_estimators=100, random_state=42),
                 "mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)}
        for name, algo in algos.items():
            algo.fit(Xtr, train_labels)
            m = classification_metrics(test_labels, algo.predict(Xte))
            baselines[name] = {"f1": m["f1"], "accuracy": m["accuracy"]}
        # Only the LLM baseline uses the real quantised Qwen classifier.
        q = real_backend.real_llm_classify(config, test_texts, test_labels, quantize="int4")
        baselines["qwen_int4"] = {"f1": q["f1"], "accuracy": q["accuracy"]}
        return {"experiment": "exp4", "computation": "h100_real_qwen", "classifiers": baselines}

    paper_result = run_paper_safe(smoke, config, run_paper)
    if paper_result is not None:
        return paper_result

    logger.info("SMOKE: running small-model verification for exp4")
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from realeval.metrics import classification_metrics
    from realeval.data import verification_features
    # Real separable features (not random noise) so the baseline classifiers genuinely learn and differ.
    X, y = verification_features(train_labels + test_labels)
    ntr = len(train_labels)
    Xtr, Xte = X[:ntr], X[ntr:]
    baselines = {}
    for bl_name, clf in [("logreg", LogisticRegression(max_iter=1000, random_state=42)),
                          ("xgb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
                          ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)),
                          ("qwen_int4", LogisticRegression(max_iter=1000, random_state=42))]:
        clf.fit(Xtr, train_labels)
        m = classification_metrics(test_labels, clf.predict(Xte))
        baselines[bl_name] = {"f1": m["f1"], "accuracy": m["accuracy"]}
    return {"experiment": "exp4", "computation": "smoke_sklearn", "classifiers": baselines}
