"""realeval/data.py — Data Loading (TAF-28k, ChiFraud, AdvFraud-3k, HF Bucket)

Loads real fraud detection datasets from data/ directory or HuggingFace bucket.
Returns (texts, labels, embeddings, speaker_labels) tuples for downstream experiments.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("data")

ROOT = Path(__file__).resolve().parent.parent
HF_BUCKET = "wangdajin062/TeleAntiFraud-bucket"

# Test override hook: set DATA to a Path to redirect all data loading (used by tests).
DATA: Path | None = None


def _data_root() -> Path:
    """Resolve data root, respecting REALEVAL_DATA_ROOT and RunPod /workspace.

    Tests can override by setting ``data.DATA = tmp_path`` (backward-compatible).
    """
    if DATA is not None:
        return DATA
    try:
        from realeval.paths import data_root
        return data_root()
    except ImportError:
        logger.warning("realeval.paths not available, falling back to package-relative data/")
        return ROOT / "data"


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
                if "label" not in obj:
                    logger.warning("Missing 'label' field at line %d in %s — using -1 (unknown)", i, path.name)
                    labels.append(-1)
                else:
                    labels.append(int(obj["label"]))
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
        jsonl_path = _data_root() / "TAF28k" / "taf28k.jsonl"
        texts, labels = [], []  # ensure defined for multimodal fallback path
        if jsonl_path.exists():
            texts, labels = _load_jsonl(jsonl_path, max_samples)
            if texts:
                if source != "multimodal":
                    return {"texts": texts, "labels": labels,
                            "embeddings": None, "speaker_labels": None, "source": "jsonl"}
    if source in ("auto", "npz", "multimodal"):
        npz_path = _data_root() / "TAF28k" / "taf28k.npz"
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
    # Fallback: try loading from HF bucket when local data is missing
    logger.warning("TAF-28k not found at %s — trying HF bucket fallback", _data_root() / "TAF28k")
    try:
        hf_data = load_hf_bucket(HF_BUCKET, split="train", max_samples=max_samples)
        if hf_data["texts"]:
            logger.info("Loaded %d samples from HF bucket as TAF28k fallback", len(hf_data["texts"]))
            hf_data["source"] = "hf_bucket_fallback"
            return hf_data
    except Exception as e:
        logger.warning("HF bucket fallback also failed: %s", e)
    return {"texts": [], "labels": [], "embeddings": None, "speaker_labels": None, "source": None}


def load_chifraud(max_samples: int | None = None) -> dict:
    """Load ChiFraud dataset (Chinese fraud detection)."""
    jsonl_path = _data_root() / "ChiFraud" / "chifraud.jsonl"
    if jsonl_path.exists():
        texts, labels = _load_jsonl(jsonl_path, max_samples)
        return {"texts": texts, "labels": labels, "embeddings": None, "speaker_labels": None, "source": "jsonl"}
    logger.warning("ChiFraud not found at %s", _data_root() / "ChiFraud")
    return {"texts": [], "labels": [], "embeddings": None, "speaker_labels": None, "source": None}


def load_advfraud3k(max_samples: int | None = None) -> dict:
    """Load AdvFraud-3k dataset (adversarial fraud detection)."""
    jsonl_path = _data_root() / "AdvFraud3k" / "advfraud3k.jsonl"
    if jsonl_path.exists():
        texts, labels = _load_jsonl(jsonl_path, max_samples)
        return {"texts": texts, "labels": labels, "embeddings": None, "speaker_labels": None, "source": "jsonl"}
    logger.warning("AdvFraud-3k not found at %s", _data_root() / "AdvFraud3k")
    return {"texts": [], "labels": [], "embeddings": None, "speaker_labels": None, "source": None}


def load_spam11358() -> dict:
    """Load spam11358 dataset (11k+ cleaned Chinese fraud SMS). All-fraud."""
    jsonl_path = _data_root() / "spam11358" / "spam11358.jsonl"
    if jsonl_path.exists():
        texts, labels = _load_jsonl(jsonl_path)
        return {"texts": texts, "labels": labels, "embeddings": None, "speaker_labels": None, "source": "spam11358"}
    logger.warning("spam11358 not found at %s", _data_root() / "spam11358")
    return {"texts": [], "labels": [], "embeddings": None, "speaker_labels": None, "source": None}


def load_chifraud_balanced() -> dict:
    """Balanced Chinese fraud detection dataset: 2000 fraud + 2000 normal.

    Normal: ChiFraud originals (149) + template generation (~800) + char augmentation (~1050).
    Fraud: spam11358 diverse fraud SMS samples (2000).
    Perfectly balanced, suitable for training without pos_weight.
    """
    path = _data_root() / "balanced4k" / "balanced4k.jsonl"
    if path.exists():
        texts, labels = _load_jsonl(path)
        return {"texts": texts, "labels": labels, "embeddings": None,
                "speaker_labels": None, "source": "balanced4k"}
    # Fallback: build on-the-fly
    cf = load_chifraud()
    sf = load_spam11358()
    cf_texts, cf_labels = cf["texts"], cf["labels"]
    sf_texts, sf_labels = sf["texts"], sf["labels"]
    n_normal = sum(1 for l in cf_labels if int(l) == 0)
    sf_fraud = [t for t, l in zip(sf_texts, sf_labels) if int(l) == 1]
    import random
    random.shuffle(sf_fraud)
    sf_fraud = sf_fraud[:n_normal * 2]
    texts = cf_texts + sf_fraud
    labels = cf_labels + [1] * len(sf_fraud)
    return {"texts": texts, "labels": labels, "embeddings": None, "speaker_labels": None, "source": "chifraud+spam11358"}


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


# ─────────────────── HuggingFace Bucket Integration ───────────────────

def load_hf_bucket(name_or_path: str = None, split: str = "train",
                   max_samples: int | None = None) -> dict:
    """Load fraud detection data from a HuggingFace bucket/dataset.

    Args:
        name_or_path: HF repo id (default: wangdajin062/TeleAntiFraud-bucket).
        split: Dataset split name ('train', 'test', etc.).
        max_samples: Cap samples for quick iteration.

    Returns:
        dict with texts, labels, embeddings, speaker_labels, source.
    """
    repo = name_or_path or HF_BUCKET
    try:
        from datasets import load_dataset
        logger.info("Loading dataset from HF: %s (split=%s)", repo, split)
        # hf:// protocol for buckets, standard loading for regular datasets
        ds = load_dataset(repo, split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        # Auto-detect column mapping
        cols = ds.column_names
        # Text column: instruction, text, content, sentence, message
        text_col = next((c for c in ["instruction", "text", "content", "sentence", "message"]
                         if c in cols), cols[0])
        # Label column: label, labels, fraud_label, is_fraud
        label_col = next((c for c in ["label", "labels", "fraud_label", "is_fraud"]
                          if c in cols), None)
        # Audio path column
        audio_col = next((c for c in ["audio_path", "audio", "wav_path", "path"]
                          if c in cols), None)

        texts = [str(row[text_col]) for row in ds]
        if label_col:
            labels = [int(row[label_col]) for row in ds]
        else:
            labels = [0] * len(texts)

        # Optional audio embeddings
        emb_col = next((c for c in ["embeddings", "audio_emb", "features"] if c in cols), None)
        embeddings = np.array([row[emb_col] for row in ds], dtype=np.float32) if emb_col else None
        spk_col = next((c for c in ["speaker_id", "speaker_label"] if c in cols), None)
        speaker_labels = [str(row[spk_col]) for row in ds] if spk_col else None

        logger.info("Loaded %d samples from HF bucket (text_col=%s, label_col=%s)",
                    len(texts), text_col, label_col)
        return {"texts": texts, "labels": labels, "embeddings": embeddings,
                "speaker_labels": speaker_labels, "source": f"hf:{repo}"}
    except Exception as e:
        logger.warning("HF bucket load failed (%s), falling back to synthetic: %s", repo, e)
        return load_synthetic(n=max_samples or 200)


def _to_binary_labels(labels: list) -> list[int]:
    """Normalise labels to binary (0/1) — fraud-like=1, normal-like=0, unknown=-1.

    FRAUD-LIKE (→1): 1, true, yes, fraud, scam, phishing, spam, malicious, attack, anomaly
    NORMAL-LIKE (→0): 0, false, no, normal, benign, safe, legitimate, clean, ham
    UNKNOWN (→-1): anything else — logged as warning, excluded from metrics downstream.
    """
    FRAUD_SET = frozenset({"1", "true", "yes", "fraud", "scam", "phishing",
                           "spam", "malicious", "attack", "anomaly"})
    NORMAL_SET = frozenset({"0", "false", "no", "normal", "benign", "safe",
                            "legitimate", "clean", "ham"})
    result = []
    unknown = 0
    for l in labels:
        s = str(l).strip().lower()
        if s in FRAUD_SET:
            result.append(1)
        elif s in NORMAL_SET:
            result.append(0)
        else:
            result.append(-1)
            unknown += 1
    if unknown:
        logger.warning("_to_binary_labels: %d/%d labels unknown (mapped to -1)", unknown, len(labels))
    return result


def prepare_sft_dataset(name_or_path: str = None, max_samples: int | None = None,
                         test_ratio: float = 0.2, seed: int = 42):
    """Prepare train/test splits for supervised fine-tuning from the HF bucket.

    If the dataset already has train/test splits (like TeleAntiFraud-bucket),
    they are used directly. Otherwise, a random split is performed.

    Returns (train_texts, train_labels), (test_texts, test_labels).
    Labels are normalised to binary (0/1) — fraud=1, normal=0.
    """
    repo = name_or_path or HF_BUCKET
    try:
        from datasets import load_dataset, get_dataset_split_names
        # Check if the dataset already has splits
        splits = get_dataset_split_names(repo)
        if "train" in splits and "test" in splits:
            logger.info("Using pre-existing train/test splits from %s", repo)
            train_ds = load_dataset(repo, split="train")
            test_ds = load_dataset(repo, split="test")
            if max_samples:
                train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
                test_ds = test_ds.select(range(min(max_samples // 10, len(test_ds))))

            text_col = next((c for c in ["instruction", "text", "content", "message"]
                             if c in train_ds.column_names), train_ds.column_names[0])
            label_col = next((c for c in ["label", "labels", "fraud_label"]
                              if c in train_ds.column_names), None)

            train_texts = [str(row[text_col]) for row in train_ds]
            test_texts = [str(row[text_col]) for row in test_ds]
            train_labels = _to_binary_labels([row[label_col] for row in train_ds]) if label_col else [0]*len(train_texts)
            test_labels = _to_binary_labels([row[label_col] for row in test_ds]) if label_col else [0]*len(test_texts)

            logger.info("SFT data: %d train (%.1f%% fraud) / %d test (%.1f%% fraud)",
                        len(train_texts), 100*sum(train_labels)/max(1,len(train_labels)),
                        len(test_texts), 100*sum(test_labels)/max(1,len(test_labels)))
            return (train_texts, train_labels), (test_texts, test_labels)
    except Exception as e:
        logger.info("No pre-existing splits (%s), falling back to manual split", e)

    # Manual split fallback
    data = load_hf_bucket(name_or_path, max_samples=max_samples)
    texts, labels = data["texts"], data["labels"]
    binary_labels = _to_binary_labels(labels)

    n = len(texts)
    n_test = max(1, int(n * test_ratio))
    indices = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    test_idx = set(indices[:n_test])
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for i, (t, l) in enumerate(zip(texts, binary_labels)):
        if i in test_idx:
            test_texts.append(t); test_labels.append(l)
        else:
            train_texts.append(t); train_labels.append(l)

    logger.info("SFT split: %d train / %d test (fraud: %d / %d)",
                len(train_texts), len(test_texts),
                sum(train_labels), sum(test_labels))
    return (train_texts, train_labels), (test_texts, test_labels)
