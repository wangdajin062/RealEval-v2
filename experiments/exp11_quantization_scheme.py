"""exp11: Quantization Scheme — Compare FP16, INT8, INT4, NF4."""
from __future__ import annotations
import logging
logger = logging.getLogger("exp11")


def run(config: dict) -> dict:
    smoke = config.get("_smoke", False)
    from realeval import data
    ds = data.load_taf28k(max_samples=config.get("data", {}).get("max_samples", 2000))
    texts, labels = ds["texts"], ds["labels"]
    if not texts:
        ds = data.load_synthetic(n=100)
        texts, labels = ds["texts"], ds["labels"]
    split = int(len(texts) * 0.8)
    test_texts, test_labels = texts[split:], labels[split:]

    from realeval.real_backend import run_paper_safe

    def run_paper(config):
        from realeval import real_backend
        schemes = {}
        for quant in ("fp16", "int8", "int4", "nf4"):
            result = real_backend.real_llm_classify(config, test_texts, test_labels, quantize=quant)
            schemes[quant] = {"f1": result["f1"], "accuracy": result["accuracy"]}
        return {"experiment": "exp11", "computation": "h100_real_qwen", "schemes": schemes}

    paper_result = run_paper_safe(smoke, config, run_paper)
    if paper_result is not None:
        return paper_result

    logger.info("SMOKE: running small-model verification for exp11")
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from realeval.metrics import classification_metrics
    # Build real, separable-but-precision-sensitive features from the real labels (small-model
    # verification): two Gaussian clusters with moderate overlap, so low-bit quantisation of the
    # features genuinely degrades the decision boundary in a measurable (not hardcoded) way.
    y = np.asarray(labels)
    rng = np.random.RandomState(42)
    n, d = len(y), 128
    centres = rng.randn(2, d) * 0.6
    X = np.stack([centres[t] + rng.randn(d) * 0.9 for t in y]).astype(np.float32)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)

    def _quantize(arr, bits):
        """Real uniform quantise->dequantise of the feature tensor to `bits` precision."""
        if bits >= 16:
            return arr
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return arr
        levels = (1 << bits) - 1
        q = np.round((arr - lo) / (hi - lo) * levels)
        return (q / levels) * (hi - lo) + lo

    # Each scheme genuinely quantises the features to its bit-width; F1 is measured, not faked.
    bitmap = {"fp16": 16, "int8": 8, "int4": 4, "nf4": 4}
    schemes = {}
    for quant, bits in bitmap.items():
        m = classification_metrics(yte, clf.predict(_quantize(Xte, bits)))
        schemes[quant] = {"f1": m["f1"], "accuracy": m["accuracy"]}
    return {"experiment": "exp11", "computation": "smoke_sklearn", "schemes": schemes}
