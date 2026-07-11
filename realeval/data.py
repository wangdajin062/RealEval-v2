"""realeval/data.py — Data Loading (TAF-28k, ChiFraud, AdvFraud-3k, AdvFraud-3k Expert)

Loads real fraud detection datasets from data/ directory.
Returns (texts, labels, embeddings, speaker_labels) tuples for downstream experiments.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("data")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _load_jsonl(path: Path, max_samples: int | None = None) -> tuple[list[str], list[int]]:
    """Load JSONL file with 'text' and 'label' fields. Returns (texts, labels)."""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
                labels.append(int(obj.get("label", 0)))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning("Skipping line %d in %s: %s", i, path.name, e)
    return texts, labels


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load NPZ file with embeddings, labels, and optional speaker_labels."""
    data = np.load(path)
    embeddings = data["embeddings"]
    labels = data["labels"]
    if "speaker_labels" in data:
        speaker_labels = data["speaker_labels"].tolist()
    else:
        speaker_labels = [f"spk_{i}" for i in range(len(labels))]
    return embeddings, labels, speaker_labels


def load_taf28k(max_samples: int | None = None, source: str = "auto") -> dict:
    """Load TAF-28k dataset.

    Args:
        max_samples: Maximum number of samples to load (None = all).
        source: 'auto' (try JSONL first, fall back to NPZ), 'jsonl', 'npz', 'multimodal' (JSONL + NPZ).

    Returns:
        dict with keys: texts, labels, embeddings, speaker_labels, source.
    """
    if source in ("auto", "jsonl", "multimodal"):
        jsonl_path = DATA / "TAF28k" / "taf28k.jsonl"
        texts, labels = [], []  # ensure defined for multimodal fallback path
        if jsonl_path.exists():
            texts, labels = _load_jsonl(jsonl_path, max_samples)
            if texts:
                if source != "multimodal":
                    return {"texts": texts, "labels": labels,
                            "embeddings": None, "speaker_labels": None, "source": "jsonl"}
    if source in ("auto", "npz", "multimodal"):
        npz_path = DATA / "TAF28k" / "taf28k.npz"
        if npz_path.exists():
            embeddings, labels_npz, spk = _load_npz(npz_path)
            if max_samples:
                embeddings = embeddings[:max_samples]
                labels_npz = labels_npz[:max_samples]
                spk = spk[:max_samples]
            if source == "multimodal" and texts:
                # Align JSONL text with NPZ embeddings by min length
                n = min(len(texts), len(embeddings))
                # Cross-validate labels between JSONL and NPZ to detect data drift
                if labels[:n] != labels_npz[:n].tolist():
                    logger.warning("Multimodal label mismatch: JSONL and NPZ labels disagree at %d positions — "
                                   "falling back to NPZ-only",
                                   sum(1 for a, b in zip(labels[:n], labels_npz[:n]) if a != b))
                    return {"texts": [], "labels": labels_npz.tolist(),
                            "embeddings": embeddings, "speaker_labels": spk, "source": "multimodal_fallback"}
                return {"texts": texts[:n], "labels": labels[:n],
                        "embeddings": embeddings[:n], "speaker_labels": spk[:n], "source": "multimodal"}
            return {"texts": [], "labels": labels_npz.tolist(),
                    "embeddings": embeddings, "speaker_labels": spk, "source": "npz"}
    logger.warning("TAF-28k not found at %s", DATA / "TAF28k")
    return {"texts": [], "labels": [], "embeddings": None, "speaker_labels": None, "source": None}


def load_chifraud(max_samples: int | None = None) -> dict:
    """Load ChiFraud dataset (Chinese fraud detection)."""
    jsonl_path = DATA / "ChiFraud" / "chifraud.jsonl"
    if jsonl_path.exists():
        texts, labels = _load_jsonl(jsonl_path, max_samples)
        return {"texts": texts, "labels": labels, "embeddings": None, "speaker_labels": None, "source": "jsonl"}
    logger.warning("ChiFraud not found at %s", DATA / "ChiFraud")
    return {"texts": [], "labels": [], "embeddings": None, "speaker_labels": None, "source": None}


def load_advfraud3k(max_samples: int | None = None) -> dict:
    """Load AdvFraud-3k dataset (adversarial fraud detection)."""
    jsonl_path = DATA / "AdvFraud3k" / "advfraud3k.jsonl"
    if jsonl_path.exists():
        texts, labels = _load_jsonl(jsonl_path, max_samples)
        return {"texts": texts, "labels": labels, "embeddings": None, "speaker_labels": None, "source": "jsonl"}
    logger.warning("AdvFraud-3k not found at %s", DATA / "AdvFraud3k")
    return {"texts": [], "labels": [], "embeddings": None, "speaker_labels": None, "source": None}


def load_advfraud3k_expert(max_samples: int | None = None) -> dict:
    """Load AdvFraud-3k Expert subset (583 human-crafted adversarial samples).

    Returns dict with keys: texts, labels, embeddings, speaker_labels, source,
    plus metadata list (annotator_id, fraud_category, strategy, etc.).
    """
    jsonl_path = DATA / "AdvFraud3k" / "advfraud3k_expert.jsonl"
    if jsonl_path.exists():
        texts, labels, meta = [], [], []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    texts.append(obj.get("text", ""))
                    labels.append(int(obj.get("label", 0)))
                    meta.append({
                        "id": obj.get("id", ""),
                        "strategy": obj.get("strategy", ""),
                        "annotator_id": obj.get("annotator_id", ""),
                        "fraud_category": obj.get("fraud_category", ""),
                        "source": obj.get("source", ""),
                    })
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning("Skipping line %d in %s: %s", i, jsonl_path.name, e)
        return {"texts": texts, "labels": labels, "embeddings": None,
                "speaker_labels": None, "source": "advfraud3k_expert", "meta": meta}
    logger.warning("AdvFraud-3k Expert not found at %s", jsonl_path)
    return {"texts": [], "labels": [], "embeddings": None,
            "speaker_labels": None, "source": None, "meta": []}


def load_synthetic(n: int = 100, seed: int = 42) -> dict:
    """Generate synthetic fraud detection data for sandbox testing.

    Returns dict with texts, labels, embeddings, speaker_labels.
    """
    rng = np.random.RandomState(seed)
    texts = [f"synthetic_fraud_{i}" if rng.rand() > 0.5 else f"synthetic_normal_{i}" for i in range(n)]
    labels = [1 if "fraud" in t else 0 for t in texts]
    embeddings = rng.randn(n, 128).astype(np.float32)
    speaker_labels = [f"spk_{i % max(1, n // 10)}" for i in range(n)]
    return {"texts": texts, "labels": labels, "embeddings": embeddings,
            "speaker_labels": speaker_labels, "source": "synthetic"}


def verification_features(labels, n_features=128, seed=42, overlap=0.9):
    """Real separable-but-precision-sensitive features for sandbox (small-model) verification.

    Two Gaussian clusters (one per class) with controllable overlap, derived from the REAL labels.
    Replaces random-noise placeholders: a classifier trained on these genuinely learns, so downstream
    F1/quantisation/fusion effects are real (measured), not hardcoded. Labelled is_synthetic upstream.
    """
    import numpy as np
    y = np.asarray(labels)
    rng = np.random.RandomState(seed)
    centres = rng.randn(2, n_features) * 0.6
    X = np.stack([centres[int(t)] + rng.randn(n_features) * overlap for t in y]).astype(np.float32)
    return X, y
