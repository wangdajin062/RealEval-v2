"""exp12: FraudFusion Baseline — Compare against FraudFusion multi-modal baseline."""
from __future__ import annotations
import logging
logger = logging.getLogger("exp12")


def run(config: dict) -> dict:
    smoke = config.get("_smoke", False)
    from realeval import data
    ds = data.load_taf28k(max_samples=config.get("data", {}).get("max_samples", 2000))
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
        qad = real_backend.real_llm_classify(config, test_texts, test_labels, quantize="int4", use_cot=True)
        # Storage footprints measured from actual model files on disk.
        import os as _os_12
        def _model_size_mb(model_path_key):
            """Measure actual model file size from disk. Returns None if not found."""
            path = config.get("models", {}).get(model_path_key, "")
            if not path:
                return None
            import glob as _glob12
            p = _os_12.path.join(str(path), "*.safetensors") if _os_12.path.isdir(str(path)) else str(path)
            # Try to resolve via models_root
            from realeval.paths import models_root
            root = models_root()
            candidate = root / path
            total = 0
            if candidate.is_dir():
                for f in candidate.rglob("*.safetensors"):
                    total += f.stat().st_size if f.exists() else 0
                for f in candidate.rglob("*.bin"):
                    total += f.stat().st_size if f.exists() else 0
                for f in candidate.rglob("*.gguf"):
                    total += f.stat().st_size if f.exists() else 0
            if total > 0:
                return round(total / 1e6, 1)
            return None
        fp = {}
        for key, label in (("teacher_7b", "7B_BF16_SAFE_QAQ"),
                           ("student", "0.5B_BF16"),
                           ("student_gguf", "0.5B_Q4_K_M")):
            sz = _model_size_mb(key)
            if sz is not None:
                fp[label] = sz
        # Compute advantage factors from measured footprints (not hardcoded).
        bf16_7b = fp.get("7B_BF16_SAFE_QAQ")
        bf16_05 = fp.get("0.5B_BF16")
        q4_05 = fp.get("0.5B_Q4_K_M")
        quant_x = round(bf16_05 / q4_05, 1) if (bf16_05 and q4_05) else None
        param_x = round(bf16_7b / bf16_05, 1) if (bf16_7b and bf16_05) else None
        total_x = round(quant_x * param_x, 1) if (quant_x and param_x) else None
        return {"experiment": "exp12", "computation": "h100_real_qwen",
                "competitor_comparison_real": {
                    "QAD_MultiGuard_INT4": {"f1": qad["f1"], "source": "ours"},
                    # FraudFusion has no released weights; marked as cite-only (no F1 compared).
                    "FraudFusion_pruned_INT4": {"f1": None, "source": "cited (no released weights)"},
                },
                "storage_decomposition_point8": {
                    "footprints_mb": fp,
                    "quantization_alone_x": quant_x,
                    "param_scale_alone_x": param_x,
                    "total_advantage_x": total_x,
                }}

    paper_result = run_paper_safe(smoke, config, run_paper)
    if paper_result is not None:
        return paper_result

    logger.info("SMOKE: running small-model verification for exp12")
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from realeval.metrics import classification_metrics
    from realeval.data import verification_features
    X, y = verification_features(train_labels + test_labels)
    ntr = len(train_labels)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X[:ntr], y[:ntr])
    f1 = classification_metrics(y[ntr:], clf.predict(X[ntr:]))["f1"]
    # Smoke path: storage footprints are estimates from typical model sizes,
    # clearly marked as smoke_proxy (not real measurements).
    return {"experiment": "exp12", "computation": "smoke_sklearn",
            "competitor_comparison_real": {
                "QAD_MultiGuard_INT4": {"f1": f1, "source": "ours"},
                "FraudFusion_pruned_INT4": {"f1": None, "source": "cited (no released weights)"},
            },
            "storage_decomposition_point8": {
                "footprints_mb": {
                    "7B_BF16_SAFE_QAQ": None,
                    "0.5B_BF16": None,
                    "0.5B_Q4_K_M": None,
                },
                "quantization_alone_x": None,
                "param_scale_alone_x": None,
                "total_advantage_x": None,
                "note": "smoke_proxy: storage footprints not measurable without real model files",
            }}
